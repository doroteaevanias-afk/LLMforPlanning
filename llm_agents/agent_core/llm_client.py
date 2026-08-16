"""
OpenAI-compatible LLM client.
Supports Ollama (local), OpenAI, Azure, or any OpenAI-compatible endpoint.
Reads config from environment variables. Logs all calls.
"""

import os
import json
import re
import threading
import time
from typing import Optional, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Anchored to this package's location (llm_agents/logs), not the caller's cwd —
# a relative default here previously scattered logs depending on where scripts were launched from.
_DEFAULT_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")

# Proactive pacing shared across all LLMClient instances in this process, so a
# benchmark that constructs one client per agent still respects one global
# rate limit against the same API key.
#
# REPLACED 2026-08-07: the previous approach (fixed 20s between calls,
# CORRECTED 2026-07-30 from an earlier RPM-only version) assumed every call
# costs ~1210 tokens on average. That's true across a whole benchmark, but
# coordinated mode's per-step token cost scales with active-EVSE count -- a
# single high-occupancy step (e.g. 25 active EVSEs) was measured pushing the
# economic + environmental + coordinator calls to ~8,800 tokens combined,
# fired within about one minute. That's already ~47% over Groq's 6000 TPM
# ceiling for llama-3.1-8b-instant from ONE step alone, no matter how the
# fixed interval is tuned -- a flat per-call delay can't distinguish a
# 3-active-EVSE step from a 40-active-EVSE one. Confirmed from
# llm_calls_2026-08-07.jsonl: once occupancy rose, calls that normally took
# ~20s took 100-116s instead (each retry re-entered the same 20s wait, so
# repeated 429s compounded rather than backing off).
#
# Fix: track actual (timestamp, token_count) usage in a trailing 60s window
# and wait only as long as needed to keep the window under a safety-margined
# TPM budget, using each call's own estimated cost (prompt size + requested
# max_tokens, the worst case for what the call could consume) rather than a
# fixed guess. A small call during a quiet step now waits far less than 20s;
# a large call during a high-occupancy step waits as long as it actually
# needs to, rather than firing anyway and eating a 429 that a flat interval
# couldn't have predicted. Requests-per-minute (RPM=30) is checked the same
# way, off the same window's entry count.
_TPM_BUDGET = float(os.getenv("LLM_TPM_BUDGET", "6000"))
_RPM_BUDGET = float(os.getenv("LLM_RPM_BUDGET", "30"))
# Headroom against estimation error: prompt token estimate is a cheap
# chars/4 heuristic (not the real tokenizer), and actual output is usually
# well under the requested max_tokens reserved up front -- 0.85 leaves room
# for both without being so conservative it re-creates the old problem.
_TPM_SAFETY_MARGIN = float(os.getenv("LLM_TPM_SAFETY_MARGIN", "0.85"))
_RATE_WINDOW_SECONDS = 60.0

_rate_limit_lock = threading.Lock()
# Each entry: [timestamp, token_count]. Mutable list (not tuple) so a
# reservation's token_count can be trued up in place once the real
# input_tokens + output_tokens are known after the call returns.
_token_window: list = []


def _estimate_tokens(text: str) -> float:
    """Cheap chars-per-token heuristic (~4 chars/token for English/JSON-ish
    text) -- good enough for a safety-margined budget check, not meant to
    match the real tokenizer exactly."""
    return len(text) / 4.0


# Matches this codebase's two resource-id schemes (see
# ev2gym_adapter.py's "port_{i}" and sustaingym_adapter.py's "evse_{i}"),
# bracketed as they appear in every prompt that lists resources: the
# economic/environmental "[evse_28] demand_remaining=..." lines, and the
# coordinator's indented "  [evse_28] kW=..." lines under each sub-agent's
# proposal (which repeats each id once per sub-agent -- deduplicated via the
# set() this is fed into below, so it still counts distinct active
# resources, not raw line count).
_RESOURCE_ID_RE = re.compile(r"\[(evse_\d+|port_\d+)\]")


def _estimate_output_tokens(prompt: str, max_tokens: int) -> float:
    """
    Estimate expected output tokens from how many distinct active resources
    this prompt actually lists, instead of assuming every call will use its
    full max_tokens ceiling.

    2026-08-07: reserving at flat max_tokens made the pacing non-adaptive in
    practice -- economic/environmental's max_tokens=3200 alone already
    dwarfs the difference between a quiet 1-EVSE prompt and a busy 25-EVSE
    one, so both ended up waiting almost the same amount despite one
    actually needing a fraction of the tokens. Response shape here is one
    action entry (~20 tokens each, generously rounded) per active resource
    plus a small fixed overhead for reasoning/wrapper JSON -- true-up
    corrects this estimate to the real count once the response is back, so
    under-estimating here costs nothing but a slightly optimistic wait; it's
    capped at max_tokens as the hard ceiling either way.
    """
    n_active = len(set(_RESOURCE_ID_RE.findall(prompt)))
    estimate = 40 + n_active * 20 if n_active > 0 else 150.0
    return min(estimate, max_tokens)


def _reserve_rate_limit_slot(prompt: str, max_tokens: int) -> list:
    """
    Block until firing a call estimated at (prompt tokens + expected output
    tokens) would keep the trailing 60s window under both the TPM and RPM
    budgets, then reserve that estimate in the window and return the
    reservation entry so the caller can true it up with the real usage once
    the response returns.
    """
    estimated = _estimate_tokens(prompt) + _estimate_output_tokens(prompt, max_tokens)
    tpm_budget = _TPM_BUDGET * _TPM_SAFETY_MARGIN
    rpm_budget = _RPM_BUDGET * _TPM_SAFETY_MARGIN

    # A single call's own worst-case cost can legitimately exceed the
    # safety-margined budget (e.g. SustainGym's _SUSTAINGYM_MAX_TOKENS=3200
    # plus a large high-occupancy prompt already totals ~5300-5700, above
    # 6000*0.85=5100) -- waiting for "under budget" would never terminate in
    # that case. Raise the effective ceiling to fit calls up to the raw
    # (non-margined) TPM budget so we still wait out any other in-flight
    # usage, but a lone big call isn't asked to wait for headroom that
    # doesn't exist.
    if estimated > tpm_budget:
        tpm_budget = min(_TPM_BUDGET, estimated)

    while True:
        with _rate_limit_lock:
            now = time.time()
            _token_window[:] = [e for e in _token_window if now - e[0] < _RATE_WINDOW_SECONDS]
            used_tokens = sum(e[1] for e in _token_window)
            if used_tokens + estimated <= tpm_budget and len(_token_window) < rpm_budget:
                entry = [now, estimated]
                _token_window.append(entry)
                return entry
            if not _token_window:
                # Nothing left to age out and still over budget: this one
                # call's own cost is simply larger than what we can protect
                # against. Let it through rather than spinning forever --
                # this is as much headroom as waiting can buy; the retry/
                # backoff in call() handles any 429 this still triggers.
                entry = [now, estimated]
                _token_window.append(entry)
                return entry
            oldest_t = _token_window[0][0]
        # Release the lock while sleeping so other threads can still check in
        # (and so a slot freed by aging-out gets picked up promptly).
        sleep_for = max(0.5, min(5.0, _RATE_WINDOW_SECONDS - (time.time() - oldest_t)))
        time.sleep(sleep_for)


def _true_up_reservation(reservation: list, actual_tokens: float) -> None:
    """Replace a reservation's worst-case estimate with the real token usage
    now that the call has returned, so later calls see accurate window state
    instead of the (usually higher) max_tokens reservation forever."""
    with _rate_limit_lock:
        reservation[1] = actual_tokens


class LLMClient:
    """OpenAI-compatible LLM provider."""

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "llama3.1:8b",
        temperature: float = 0.0,
        max_tokens: int = 1200,
        timeout: float = 180.0,
        log_dir: Optional[str] = None
    ):
        """
        Initialize LLM client.
        
        Args:
            provider: "ollama", "openai", "azure", etc. (for fallback behavior)
            model: Model name, e.g., "llama3.1:8b"
            temperature: Always 0 for deterministic behavior
            max_tokens: Max output length
            timeout: Request timeout (seconds)
            log_dir: Where to save JSON logs. Defaults to llm_agents/logs,
                resolved relative to this file rather than the caller's cwd.
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.log_dir = log_dir if log_dir is not None else _DEFAULT_LOG_DIR
        
        # Read API config from environment
        self.api_key = os.getenv("LLM_API_KEY", "ollama")  # Ollama doesn't require a key
        self.base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")  # Ollama default
        self.model_override = os.getenv("LLM_MODEL", model)  # Allow override
        
        # Use env var model if set
        if self.model_override != model:
            self.model = self.model_override
        
        # Ensure log dir exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        logger.info(
            f"LLMClient initialized: provider={self.provider}, "
            f"model={self.model}, base_url={self.base_url}, temp={self.temperature}"
        )
    
    def call(
        self,
        prompt: str,
        plan_id: str = "",
        episode_index: int = 0,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, Dict[str, Any]]:
        """
        Call LLM with prompt.

        Args:
            prompt: Prompt text
            plan_id: Unique identifier for this planning episode (for logging)
            episode_index: Which episode (within the caller's lifetime) this
                call belongs to (for logging/grouping calls by episode)
            max_tokens: Per-call override for this instance's max_tokens
                (e.g. the coordinator role's third call needs more room than
                economic/environmental's own calls, since it echoes both
                sub-proposals' full action lists). Defaults to the instance's
                max_tokens (set at construction) when not given.

        Returns:
            (response_text, metadata_dict)

        Raises:
            Exception: If LLM call fails
        """
        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required. Install: pip install openai")
        
        # Configure OpenAI client (works with any OpenAI-compatible API)
        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        max_transient_retries = 5
        t0 = time.time()

        # Reserved ONCE up front, not re-checked on every retry: a rejected
        # (429) request consumes no real quota, so re-waiting a full pacing
        # slot per retry (the old behavior) only compounded delay without
        # protecting the budget any further -- that compounding is what
        # turned ~20s calls into ~100s+ ones once occupancy rose (see this
        # module's 2026-08-07 pacing comment above). The 2-10s backoff below
        # already spaces retries out; the reservation just needs to hold the
        # worst case (effective_max_tokens) so the trailing-window budget
        # stays honest regardless of how many attempts this call takes.
        reservation = _reserve_rate_limit_slot(prompt, effective_max_tokens)

        try:
            for attempt in range(max_transient_retries + 1):
                try:
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a power grid optimization assistant. Be concise and structured."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=effective_max_tokens,
                        timeout=self.timeout,
                        # At temperature=0.0, small models can fall into a
                        # token-repetition loop (re-emitting the same action
                        # list over and over instead of stopping), which runs
                        # out the token budget mid-response and produces
                        # unrepairable truncated JSON. frequency_penalty
                        # directly discourages repeating already-used tokens,
                        # which breaks that loop.
                        frequency_penalty=0.3
                    )
                    break
                # RateLimitError (429), APIConnectionError (includes its
                # subclass APITimeoutError -- network drop or request timeout),
                # and InternalServerError (5xx) are all transient conditions
                # worth retrying with backoff. Confirmed 2026-07-29: a
                # mid-episode call that raised something in this set (not
                # further identified -- see llm_calls_2026-07-29.jsonl's 3
                # single-call replans) previously propagated straight through
                # uncaught, discarding that seed's entire episode-so-far (no
                # per-step retry existed anywhere in the pipeline). Anything
                # NOT in this set (auth errors, bad request, etc.) still
                # raises immediately -- retrying those would not help.
                except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError) as exc:
                    if attempt == max_transient_retries:
                        raise
                    backoff = 2.0 * (attempt + 1)
                    logger.warning(
                        f"Transient error from {self.provider} (attempt {attempt + 1}/"
                        f"{max_transient_retries}), backing off {backoff:.1f}s: {exc}"
                    )
                    time.sleep(backoff)

            latency_ms = (time.time() - t0) * 1000
            response_text = response.choices[0].message.content.strip()
            
            metadata = {
                "model": self.model,
                "provider": self.provider,
                "temperature": self.temperature,
                "latency_ms": latency_ms,
                "timestamp_iso": datetime.utcnow().isoformat() + "Z",
                "plan_id": plan_id,
                "episode_index": episode_index,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }

            # Now that real usage is known, replace the worst-case
            # (max_tokens) reservation with it so the trailing window
            # reflects actual consumption for later calls' budget checks --
            # most calls finish well under max_tokens, so leaving the
            # reservation at its worst-case estimate would make the window
            # look busier than it really is and cause unnecessary waiting.
            _true_up_reservation(reservation, metadata["input_tokens"] + metadata["output_tokens"])

            # Log to JSON
            self._log_call(prompt, response_text, metadata)

            return response_text, metadata

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def _log_call(self, prompt: str, response: str, metadata: Dict[str, Any]) -> None:
        """Save prompt + response to JSON log."""
        log_entry = {
            "timestamp_iso": metadata["timestamp_iso"],
            "plan_id": metadata.get("plan_id", ""),
            "episode_index": metadata.get("episode_index", 0),
            "model": metadata["model"],
            "provider": metadata["provider"],
            "temperature": metadata["temperature"],
            "latency_ms": metadata["latency_ms"],
            "input_tokens": metadata.get("input_tokens", 0),
            "output_tokens": metadata.get("output_tokens", 0),
            "prompt": prompt,
            "response": response,
        }
        
        # Log to dated file
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        log_file = os.path.join(self.log_dir, f"llm_calls_{date_str}.jsonl")
        
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        logger.debug(f"Logged LLM call to {log_file}")
