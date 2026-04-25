#!/usr/bin/env python3

import os
import random
import sys
import time

# Code style:
# - No type hinting
# - No doc strings
# - No triple quoted multi-line strings
# - No comments with repeated characters for visual page breaks like # ---
# - No non-ascii characters
# - No global variables unless making them local increases complexity
# - Yes strategic inline comments enhancing rapid code comprehension by real humans


# Default provider selection. Edit this single line to switch backends for
# testing. Supported values: "anthropic", "moonshot".
# DEFAULT_PROVIDER = ["anthropic", "moonshot", "google", "qwen", "xiaomi", "glm"][5]
DEFAULT_PROVIDER = ["anthropic", "qwen", "glm", "ds4_pro", "ds4_flash"][0]

MAX_RETRIES = 8

# Anthropic / Opus 4.7 specifics (as of 2026-04):
# - budget_tokens is removed; only thinking: {"type": "adaptive"} is valid.
# - thinking display defaults to "omitted" on 4.7, so thinking blocks are not
#   returned in the response unless we explicitly ask for "summarized". Since
#   we want plain text output, omitted is exactly what we want.
# - temperature/top_p/top_k return 400 if set to non-default values.
# - effort controls token spend. "medium" fits this task: synthesizing ~20
#   numeric datapoints against defined thresholds and picking from 4 enumerated
#   actions. "high" (default) over-elaborates; "low" skips thinking entirely.
ANTHROPIC_MODEL = "claude-opus-4-7"
ANTHROPIC_MAX_TOKENS = 8000
ANTHROPIC_EFFORT = "medium"


# Moonshot / Kimi K2.6 specifics (as of 2026-04):
# - OpenAI-compatible endpoint at https://api.moonshot.ai/v1.
# - Model id is "kimi-k2.6" (also has kimi-k2-thinking, kimi-k2.5, etc).
# - 256K context window; we cap output tokens to keep cost predictable.
# - Thinking mode is ENABLED by default on k2.6 and returns reasoning in a
#   separate "reasoning" field on the message (not inside content). We leave
#   thinking on - this task is exactly the kind of multi-input synthesis
#   K2.6 thinking was designed for - and just ignore the reasoning field.
# - Moonshot recommends temperature=1.0 for thinking mode, top_p=0.95.
MOONSHOT_MODEL = "kimi-k2.6"
MOONSHOT_BASE_URL = "https://api.moonshot.ai/v1"
MOONSHOT_MAX_TOKENS = 8000
MOONSHOT_TEMPERATURE = 1.0
MOONSHOT_TOP_P = 0.95


# Google / Gemini 3.1 Pro specifics (as of 2026-04):
# - Uses the new google-genai SDK (pip install google-genai). The older
#   google-generativeai package is deprecated - do not use it.
# - Model id "gemini-3.1-pro-preview". The "gemini-3-pro-preview" alias
#   was retired Mar 2026 and now also points here.
# - Three thinking levels: low / medium / high. Default is high, which
#   over-elaborates on this task the same way Opus high-effort does, so
#   we pin to medium - Google explicitly recommends medium for "code
#   review and data analysis", which closely matches our synthesis job.
# - Thinking cannot be disabled on 3.1 Pro; LOW is the floor.
# - response.text gives the final answer text; thought parts are filtered
#   out automatically (they'd only appear if include_thoughts were set).
# - The SDK auto-detects GEMINI_API_KEY or GOOGLE_API_KEY from env; we
#   check both explicitly so we can give a clean error if neither is set.
GOOGLE_MODEL = "gemini-3.1-pro-preview"
GOOGLE_THINKING_LEVEL = "medium"
GOOGLE_MAX_TOKENS = 8000


# Qwen3.6-Plus specifics (as of 2026-04):
# - Hosted by Alibaba Cloud Model Studio (DashScope). The Plus tier is a
#   proprietary API-only model; not self-hostable.
# - Alibaba's recommended Python integration is the OpenAI-compatible
#   endpoint, so we reuse the openai SDK (same as Moonshot) rather than
#   pulling in the dashscope package.
# - International endpoint is Singapore. Beijing / US-Virginia / HK exist
#   as alternatives but are not interchangeable - API keys are region-
#   specific. HRV_QWEN_BASE_URL env var lets you override without editing.
# - Hybrid thinking mode is enabled by default for qwen3.6-plus, but we
#   pass enable_thinking=True explicitly so behavior stays deterministic
#   if Alibaba flips the default later. Reasoning content comes back in
#   message.reasoning_content (not in .content), so the final answer is
#   already clean - same pattern as Kimi.
QWEN_MODEL = "qwen3.6-plus"
QWEN_BASE_URL = os.environ.get(
    "HRV_QWEN_BASE_URL",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)
QWEN_MAX_TOKENS = 8000
QWEN_ENABLE_THINKING = True


# Xiaomi MiMo-V2-Pro via OpenRouter specifics (as of 2026-04):
# - Closed-weight flagship; OpenRouter lists it with a single provider
#   (Xiaomi direct). That means the :exacto suffix is a no-op for this
#   model - :exacto is a routing shortcut that reorders *multiple*
#   providers by tool-calling quality signals, and there's only one
#   provider here. We use the plain slug.
# - OpenRouter's API is OpenAI-compatible, so we reuse the openai SDK
#   with a base_url override - same pattern as Moonshot/Qwen.
# - Reasoning is controlled via OpenRouter's unified "reasoning" param
#   (passed through extra_body). For single-turn synthesis tasks like
#   this one, on=true is right; Xiaomi recommends off for agentic tool
#   loops where latency matters more than deliberation.
# - HTTP-Referer / X-Title are optional attribution headers that put
#   your app on OpenRouter's leaderboards. Safe to leave as defaults.
XIAOMI_MODEL = "xiaomi/mimo-v2-pro"
XIAOMI_BASE_URL = "https://openrouter.ai/api/v1"
XIAOMI_MAX_TOKENS = 8000
XIAOMI_REASONING_ENABLED = True
XIAOMI_APP_URL = os.environ.get("HRV_OPENROUTER_APP_URL", "")
XIAOMI_APP_TITLE = os.environ.get("HRV_OPENROUTER_APP_TITLE", "hrv-trainer-prompt")


GLM_MODEL = "z-ai/glm-5.1:exacto"
GLM_BASE_URL = "https://openrouter.ai/api/v1"
GLM_MAX_TOKENS = 8000
GLM_REASONING_ENABLED = True
GLM_APP_URL = os.environ.get("HRV_OPENROUTER_APP_URL", "")
GLM_APP_TITLE = os.environ.get("HRV_OPENROUTER_APP_TITLE", "hrv-trainer-prompt")

# "ds4_pro", "ds4_flash"

DS4_PRO_MODEL = "deepseek/deepseek-v4-pro:exacto"
DS4_FLASH_MODEL = "deepseek/deepseek-v4-flash:exacto"
DS4_BASE_URL = "https://openrouter.ai/api/v1"
DS4_MAX_TOKENS = 8000
DS4_REASONING_ENABLED = True
DS4_APP_URL = os.environ.get("HRV_OPENROUTER_APP_URL", "")
DS4_APP_TITLE = os.environ.get("HRV_OPENROUTER_APP_TITLE", "hrv-trainer-prompt")


def _sleep_with_jitter(attempt):
    # Exponential backoff with jitter: 1-2s, 2-4s, 4-8s, 8-16s...
    delay = random.uniform(2 ** (attempt - 1), 2**attempt)
    print(f"  [retry {attempt}/{MAX_RETRIES - 1}] waiting {delay:.1f}s...", file=sys.stderr)
    time.sleep(delay)


def _call_anthropic(prompt):
    import anthropic  # deferred; only needed on this call path

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ERROR: provider='anthropic' but ANTHROPIC_API_KEY is not set")

    client = anthropic.Anthropic(api_key=api_key)

    print(
        f"# calling {ANTHROPIC_MODEL} (effort={ANTHROPIC_EFFORT}, " "adaptive thinking, display omitted)...",
        file=sys.stderr,
    )

    response = None
    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            _sleep_with_jitter(attempt)
        try:
            response = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=ANTHROPIC_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
                thinking={"type": "adaptive"},
                output_config={"effort": ANTHROPIC_EFFORT},
            )
            break
        except anthropic.RateLimitError:
            if attempt < MAX_RETRIES - 1:
                print("  [error] 429 rate limit, retrying...", file=sys.stderr)
                continue
            raise
        except anthropic.APIConnectionError as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  [error] connection error, retrying: {e}", file=sys.stderr)
                continue
            raise
        except anthropic.APIStatusError as e:
            if e.status_code >= 500 and attempt < MAX_RETRIES - 1:
                print(f"  [error] {e.status_code} server error, retrying...", file=sys.stderr)
                continue
            raise
    if response is None:
        raise RuntimeError("_call_anthropic: exhausted retries")

    # Extract only text blocks. Thinking blocks are omitted by default on
    # Opus 4.7, but we filter defensively in case that ever changes.
    text = "".join(b.text for b in response.content if b.type == "text").strip()

    usage = response.usage
    print(
        f"# input_tokens={usage.input_tokens} output_tokens={usage.output_tokens}",
        file=sys.stderr,
    )
    if getattr(response, "stop_reason", None) == "max_tokens":
        print("# WARNING: hit max_tokens; output likely truncated", file=sys.stderr)

    return text


def _call_moonshot(prompt):
    # Moonshot's API is OpenAI-compatible, so we use the openai SDK with a
    # base_url override. This avoids pulling in a second vendor SDK.
    import openai  # deferred; only needed on this call path

    api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        sys.exit("ERROR: provider='moonshot' but MOONSHOT_API_KEY is not set")

    client = openai.OpenAI(api_key=api_key, base_url=MOONSHOT_BASE_URL)

    print(
        f"# calling {MOONSHOT_MODEL} via {MOONSHOT_BASE_URL} " f"(thinking mode on, temp={MOONSHOT_TEMPERATURE})...",
        file=sys.stderr,
    )

    response = None
    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            _sleep_with_jitter(attempt)
        try:
            response = client.chat.completions.create(
                model=MOONSHOT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MOONSHOT_MAX_TOKENS,
                temperature=MOONSHOT_TEMPERATURE,
                top_p=MOONSHOT_TOP_P,
            )
            break
        except openai.RateLimitError:
            if attempt < MAX_RETRIES - 1:
                print("  [error] 429 rate limit, retrying...", file=sys.stderr)
                continue
            raise
        except openai.APIConnectionError as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  [error] connection error, retrying: {e}", file=sys.stderr)
                continue
            raise
        except openai.APIStatusError as e:
            if e.status_code >= 500 and attempt < MAX_RETRIES - 1:
                print(f"  [error] {e.status_code} server error, retrying...", file=sys.stderr)
                continue
            raise
    if response is None:
        raise RuntimeError("_call_moonshot: exhausted retries")

    # Standard OpenAI response shape. K2.6 returns reasoning in a separate
    # "reasoning" attribute on the message (not inside content), so the
    # content field is already the clean final answer - no filtering needed.
    choice = response.choices[0]
    text = (choice.message.content or "").strip()

    usage = response.usage
    print(
        f"# prompt_tokens={usage.prompt_tokens} completion_tokens={usage.completion_tokens}",
        file=sys.stderr,
    )
    if choice.finish_reason == "length":
        print("# WARNING: hit max_tokens; output likely truncated", file=sys.stderr)

    return text


def _call_google(prompt):
    # The google-genai SDK is the supported library (google-generativeai
    # is deprecated). Import deferred so users of other backends don't
    # need it installed.
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types as genai_types

    # The SDK prefers GOOGLE_API_KEY over GEMINI_API_KEY when both are
    # set. We accept either and pass it explicitly - keeps the error path
    # clean when neither is defined.
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("ERROR: provider='google' but neither GOOGLE_API_KEY nor GEMINI_API_KEY is set")

    client = genai.Client(api_key=api_key)

    config = genai_types.GenerateContentConfig(
        max_output_tokens=GOOGLE_MAX_TOKENS,
        thinking_config=genai_types.ThinkingConfig(thinking_level=GOOGLE_THINKING_LEVEL),
    )

    print(
        f"# calling {GOOGLE_MODEL} (thinking_level={GOOGLE_THINKING_LEVEL})...",
        file=sys.stderr,
    )

    response = None
    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            _sleep_with_jitter(attempt)
        try:
            response = client.models.generate_content(
                model=GOOGLE_MODEL,
                contents=prompt,
                config=config,
            )
            break
        except genai_errors.ClientError as e:
            # 429 rate limit is the one 4xx we retry; other 4xx are caller bugs.
            code = getattr(e, "code", None)
            if code == 429 and attempt < MAX_RETRIES - 1:
                print("  [error] 429 rate limit, retrying...", file=sys.stderr)
                continue
            raise
        except genai_errors.ServerError as e:
            if attempt < MAX_RETRIES - 1:
                code = getattr(e, "code", "5xx")
                print(f"  [error] {code} server error, retrying...", file=sys.stderr)
                continue
            raise
    if response is None:
        raise RuntimeError("_call_google: exhausted retries")

    # response.text is the convenience accessor that concatenates text parts
    # and excludes thought parts. Safe even though we didn't set
    # include_thoughts, because thought parts wouldn't appear anyway.
    text = (response.text or "").strip()

    usage = response.usage_metadata
    if usage is not None:
        # thoughts_token_count isn't included in candidates_token_count, so
        # surface it separately - thinking at "medium" level can be substantial
        # and affects cost even though it's not in the visible output.
        thoughts = getattr(usage, "thoughts_token_count", None) or 0
        print(
            f"# prompt_tokens={usage.prompt_token_count} " f"output_tokens={usage.candidates_token_count} " f"thoughts_tokens={thoughts}",
            file=sys.stderr,
        )

    # Finish reason lives on the candidate, not the response. "MAX_TOKENS"
    # is the truncation signal on this SDK (enum value, compared as string).
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        finish = getattr(candidates[0], "finish_reason", None)
        if finish is not None and str(finish).endswith("MAX_TOKENS"):
            print("# WARNING: hit max_tokens; output likely truncated", file=sys.stderr)

    return text


def _call_qwen(prompt):
    # Alibaba's Model Studio exposes an OpenAI-compatible endpoint, so we
    # reuse the openai SDK - same pattern as Moonshot, just a different
    # base_url and a different extra_body dial for thinking mode.
    import openai

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        sys.exit("ERROR: provider='qwen' but DASHSCOPE_API_KEY is not set")

    client = openai.OpenAI(api_key=api_key, base_url=QWEN_BASE_URL)

    print(
        f"# calling {QWEN_MODEL} via {QWEN_BASE_URL} " f"(thinking={'on' if QWEN_ENABLE_THINKING else 'off'})...",
        file=sys.stderr,
    )

    response = None
    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            _sleep_with_jitter(attempt)
        try:
            response = client.chat.completions.create(
                model=QWEN_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=QWEN_MAX_TOKENS,
                # Qwen-specific toggles ride in extra_body - the OpenAI
                # schema has no field for these. enable_thinking is
                # explicit to lock behavior; defaults on Alibaba's side
                # have changed before.
                extra_body={"enable_thinking": QWEN_ENABLE_THINKING},
            )
            break
        except openai.RateLimitError:
            if attempt < MAX_RETRIES - 1:
                print("  [error] 429 rate limit, retrying...", file=sys.stderr)
                continue
            raise
        except openai.APIConnectionError as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  [error] connection error, retrying: {e}", file=sys.stderr)
                continue
            raise
        except openai.APIStatusError as e:
            if e.status_code >= 500 and attempt < MAX_RETRIES - 1:
                print(f"  [error] {e.status_code} server error, retrying...", file=sys.stderr)
                continue
            raise
    if response is None:
        raise RuntimeError("_call_qwen: exhausted retries")

    # Reasoning goes to message.reasoning_content, final answer to
    # message.content - so .content is already clean (same as Kimi).
    choice = response.choices[0]
    text = (choice.message.content or "").strip()

    usage = response.usage
    print(
        f"# prompt_tokens={usage.prompt_tokens} completion_tokens={usage.completion_tokens}",
        file=sys.stderr,
    )
    if choice.finish_reason == "length":
        print("# WARNING: hit max_tokens; output likely truncated", file=sys.stderr)

    return text


def _call_xiaomi(prompt):
    # OpenRouter exposes an OpenAI-compatible API, so we reuse the openai
    # SDK with a base_url override. Same structural pattern as Moonshot
    # and Qwen - the differences are just base_url, the extra_body keys,
    # and two optional attribution headers.
    import openai

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("ERROR: provider='xiaomi' but OPENROUTER_API_KEY is not set")

    # OpenRouter's attribution headers are optional. Populating them lets
    # the request show up on their leaderboards if you care. Empty strings
    # are fine if you don't. Headers with empty values get dropped by httpx.
    default_headers = {}
    if XIAOMI_APP_URL:
        default_headers["HTTP-Referer"] = XIAOMI_APP_URL
    if XIAOMI_APP_TITLE:
        default_headers["X-Title"] = XIAOMI_APP_TITLE

    client = openai.OpenAI(
        api_key=api_key,
        base_url=XIAOMI_BASE_URL,
        default_headers=default_headers or None,
    )

    print(
        f"# calling {XIAOMI_MODEL} via openrouter " f"(reasoning={'on' if XIAOMI_REASONING_ENABLED else 'off'})...",
        file=sys.stderr,
    )

    response = None
    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            _sleep_with_jitter(attempt)
        try:
            response = client.chat.completions.create(
                model=XIAOMI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=XIAOMI_MAX_TOKENS,
                # OpenRouter's unified reasoning param: {"enabled": bool}
                # toggles thinking across providers without us having to
                # know each backend's native dial. Rides in extra_body
                # because the OpenAI schema has no "reasoning" field.
                extra_body={"reasoning": {"enabled": XIAOMI_REASONING_ENABLED}},
            )
            break
        except openai.RateLimitError:
            if attempt < MAX_RETRIES - 1:
                print("  [error] 429 rate limit, retrying...", file=sys.stderr)
                continue
            raise
        except openai.APIConnectionError as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  [error] connection error, retrying: {e}", file=sys.stderr)
                continue
            raise
        except openai.APIStatusError as e:
            if e.status_code >= 500 and attempt < MAX_RETRIES - 1:
                print(f"  [error] {e.status_code} server error, retrying...", file=sys.stderr)
                continue
            raise
    if response is None:
        raise RuntimeError("_call_xiaomi: exhausted retries")

    # OpenRouter puts reasoning in message.reasoning (distinct from the
    # final answer in message.content). So .content is already clean.
    choice = response.choices[0]
    text = (choice.message.content or "").strip()

    usage = response.usage
    print(
        f"# prompt_tokens={usage.prompt_tokens} completion_tokens={usage.completion_tokens}",
        file=sys.stderr,
    )
    # OpenRouter normalizes finish_reason to OpenAI values, so "length"
    # is the truncation signal here same as with direct OpenAI calls.
    if choice.finish_reason == "length":
        print("# WARNING: hit max_tokens; output likely truncated", file=sys.stderr)

    return text


def _call_glm(prompt):
    # OpenRouter exposes an OpenAI-compatible API, so we reuse the openai
    # SDK with a base_url override. Same structural pattern as Moonshot
    # and Qwen - the differences are just base_url, the extra_body keys,
    # and two optional attribution headers.
    import openai

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("ERROR: provider='glm' but OPENROUTER_API_KEY is not set")

    # OpenRouter's attribution headers are optional. Populating them lets
    # the request show up on their leaderboards if you care. Empty strings
    # are fine if you don't. Headers with empty values get dropped by httpx.
    default_headers = {}
    if GLM_APP_URL:
        default_headers["HTTP-Referer"] = GLM_APP_URL
    if GLM_APP_TITLE:
        default_headers["X-Title"] = GLM_APP_TITLE

    client = openai.OpenAI(
        api_key=api_key,
        base_url=GLM_BASE_URL,
        default_headers=default_headers or None,
    )

    print(
        f"# calling {GLM_MODEL} via openrouter " f"(reasoning={'on' if GLM_REASONING_ENABLED else 'off'})...",
        file=sys.stderr,
    )

    response = None
    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            _sleep_with_jitter(attempt)
        try:
            response = client.chat.completions.create(
                model=GLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=GLM_MAX_TOKENS,
                # OpenRouter's unified reasoning param: {"enabled": bool}
                # toggles thinking across providers without us having to
                # know each backend's native dial. Rides in extra_body
                # because the OpenAI schema has no "reasoning" field.
                extra_body={"reasoning": {"enabled": GLM_REASONING_ENABLED}},
            )
            break
        except openai.RateLimitError:
            if attempt < MAX_RETRIES - 1:
                print("  [error] 429 rate limit, retrying...", file=sys.stderr)
                continue
            raise
        except openai.APIConnectionError as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  [error] connection error, retrying: {e}", file=sys.stderr)
                continue
            raise
        except openai.APIStatusError as e:
            if e.status_code >= 500 and attempt < MAX_RETRIES - 1:
                print(f"  [error] {e.status_code} server error, retrying...", file=sys.stderr)
                continue
            raise
    if response is None:
        raise RuntimeError("_call_glm: exhausted retries")

    # OpenRouter puts reasoning in message.reasoning (distinct from the
    # final answer in message.content). So .content is already clean.
    choice = response.choices[0]
    text = (choice.message.content or "").strip()

    usage = response.usage
    print(
        f"# prompt_tokens={usage.prompt_tokens} completion_tokens={usage.completion_tokens}",
        file=sys.stderr,
    )
    # OpenRouter normalizes finish_reason to OpenAI values, so "length"
    # is the truncation signal here same as with direct OpenAI calls.
    if choice.finish_reason == "length":
        print("# WARNING: hit max_tokens; output likely truncated", file=sys.stderr)

    return text


def _call_ds4(prompt):
    # OpenRouter exposes an OpenAI-compatible API, so we reuse the openai
    # SDK with a base_url override. Same structural pattern as Moonshot
    # and Qwen - the differences are just base_url, the extra_body keys,
    # and two optional attribution headers.
    import openai

    if DEFAULT_PROVIDER == "ds4_pro":
        DS4_MODEL = DS4_PRO_MODEL
    if DEFAULT_PROVIDER == "ds4_flash":
        DS4_MODEL = DS4_FLASH_MODEL

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("ERROR: provider='ds4' but OPENROUTER_API_KEY is not set")

    # OpenRouter's attribution headers are optional. Populating them lets
    # the request show up on their leaderboards if you care. Empty strings
    # are fine if you don't. Headers with empty values get dropped by httpx.
    default_headers = {}
    if DS4_APP_URL:
        default_headers["HTTP-Referer"] = DS4_APP_URL
    if DS4_APP_TITLE:
        default_headers["X-Title"] = DS4_APP_TITLE

    client = openai.OpenAI(
        api_key=api_key,
        base_url=DS4_BASE_URL,
        default_headers=default_headers or None,
    )

    print(
        f"# calling {DS4_MODEL} via openrouter " f"(reasoning={'on' if DS4_REASONING_ENABLED else 'off'})...",
        file=sys.stderr,
    )

    response = None
    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            _sleep_with_jitter(attempt)
        try:
            response = client.chat.completions.create(
                model=DS4_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=DS4_MAX_TOKENS,
                # OpenRouter's unified reasoning param: {"enabled": bool}
                # toggles thinking across providers without us having to
                # know each backend's native dial. Rides in extra_body
                # because the OpenAI schema has no "reasoning" field.
                extra_body={"reasoning": {"enabled": DS4_REASONING_ENABLED}},
            )
            break
        except openai.RateLimitError:
            if attempt < MAX_RETRIES - 1:
                print("  [error] 429 rate limit, retrying...", file=sys.stderr)
                continue
            raise
        except openai.APIConnectionError as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  [error] connection error, retrying: {e}", file=sys.stderr)
                continue
            raise
        except openai.APIStatusError as e:
            if e.status_code >= 500 and attempt < MAX_RETRIES - 1:
                print(f"  [error] {e.status_code} server error, retrying...", file=sys.stderr)
                continue
            raise
    if response is None:
        raise RuntimeError("_call_ds4: exhausted retries")

    # OpenRouter puts reasoning in message.reasoning (distinct from the
    # final answer in message.content). So .content is already clean.
    choice = response.choices[0]
    text = (choice.message.content or "").strip()

    usage = response.usage
    print(
        f"# prompt_tokens={usage.prompt_tokens} completion_tokens={usage.completion_tokens}",
        file=sys.stderr,
    )
    # OpenRouter normalizes finish_reason to OpenAI values, so "length"
    # is the truncation signal here same as with direct OpenAI calls.
    if choice.finish_reason == "length":
        print("# WARNING: hit max_tokens; output likely truncated", file=sys.stderr)

    return text


# Dispatch table. Adding a provider = one entry here + one _call_* function.
_PROVIDERS = {
    "anthropic": _call_anthropic,
    "moonshot": _call_moonshot,
    "google": _call_google,
    "qwen": _call_qwen,
    "xiaomi": _call_xiaomi,
    "glm": _call_glm,
    "ds4_pro": _call_ds4,
    "ds4_flash": _call_ds4,
}


def call_llm(prompt, provider=None):
    # Main entry point. Caller passes the prompt and gets back response text.
    # Callers who don't specify a provider get DEFAULT_PROVIDER - flip that
    # constant at the top of this file to switch defaults for testing.
    name = provider or DEFAULT_PROVIDER
    if name not in _PROVIDERS:
        raise ValueError(f"unknown provider: {name!r} (known: {sorted(_PROVIDERS)})")
    return _PROVIDERS[name](prompt)
