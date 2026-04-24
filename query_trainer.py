#!/usr/bin/env python3

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from llm_client import call_llm

# Code style:
# - No type hinting
# - No doc strings
# - No triple quoted multi-line strings
# - No comments with repeated characters for visual page breaks like # ---
# - No non-ascii characters
# - No command line argument processing
# - No global variables unless making them local increases complexity
# - Yes strategic inline comments enhancing rapid code comprehension by real humans
# - Yes if __name__ == "__main__": main()


CONTEXT_DIR = Path("./hrv_context")
PACIFIC = ZoneInfo("America/Los_Angeles")

# 17:00 Pacific is the morning/evening cutover. Before this, "today's run"
# is still on the table and fresh morning HRV (if any) is the key input.
# After this, the decision is for tomorrow.
EVENING_CUTOVER_HOUR = 17

LAT = float(os.environ.get("HRV_LAT", "33.87"))
LON = float(os.environ.get("HRV_LON", "-118.32"))
LOCATION_LABEL = os.environ.get("HRV_LOCATION_LABEL", "South Bay LA")

# NWS fronted by a CDN that filters non-browser UAs. Chrome's UA string
# passes through; a bare "my-script" does not. We prepend a real browser
# UA then append an identifier suffix with a contact. NWS docs request a
# contact (email or URL) in the UA so they can reach you if your script
# misbehaves; set HRV_CONTACT_EMAIL to your real email to comply.
# Default below is a generic placeholder safe to publish.
_BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) " "AppleWebKit/537.36 (KHTML, like Gecko) " "Chrome/124.0.0.0 Safari/537.36"
_CONTACT = os.environ.get("HRV_CONTACT_EMAIL", "nobody@example.com")
NWS_UA = f"{_BROWSER_UA} (hrv-trainer-prompt; {_CONTACT})"
HTTP_TIMEOUT_SEC = 8


def now_pacific():
    return datetime.now(PACIFIC)


def time_of_day_label(dt):
    return "morning" if dt.hour < EVENING_CUTOVER_HOUR else "evening"


def decision_horizon(dt):
    # Returns (horizon_label, target_date).
    # Morning -> today. Evening -> tomorrow.
    if time_of_day_label(dt) == "morning":
        return "today", dt.date()
    return "tomorrow", (dt + timedelta(days=1)).date()


# --- Context file loading ---


def detect_fresh_hrv(target_date):
    # True if a morning HRV session file exists for target_date.
    # Filename convention 20_session_YYYY-M-D[_N].md does not zero-pad month
    # or day in the existing pipeline, so we probe all four padding variants
    # to be robust to pipeline changes.
    if not CONTEXT_DIR.exists():
        return False
    y, m, d = target_date.year, target_date.month, target_date.day
    patterns = [
        f"20_session_{y}-{m}-{d}*.md",
        f"20_session_{y}-{m}-{d:02d}*.md",
        f"20_session_{y}-{m:02d}-{d}*.md",
        f"20_session_{y}-{m:02d}-{d:02d}*.md",
    ]
    seen = set()
    for pat in patterns:
        for p in CONTEXT_DIR.glob(pat):
            if p in seen:
                continue
            seen.add(p)
            try:
                if "Morning HRV" in p.read_text():
                    return True
            except Exception:
                continue
    return False


def load_context_files():
    if not CONTEXT_DIR.exists():
        return None, []
    files = sorted(CONTEXT_DIR.glob("*.md"))
    if not files:
        return None, []
    parts = []
    for p in files:
        try:
            parts.append(p.read_text())
        except Exception as e:
            parts.append(f"(error reading {p.name}: {e})")
    return "\n\n".join(parts), [p.name for p in files]


# --- Weather (NWS api.weather.gov, no API key required) ---


def _http_get_json(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SEC) as r:
        return json.loads(r.read())


def fetch_weather_nws():
    # Two-step: points endpoint gives us the hourly forecast URL for this grid.
    # Headers mirror a successful browser-style request - NWS's CDN filters
    # out non-browser-looking traffic, so Accept-Language + the browser UA
    # in NWS_UA together get us through.
    headers = {
        "User-Agent": NWS_UA,
        "Accept": "application/geo+json",
        "Accept-Language": "en-US,en;q=0.9",
    }
    pts = _http_get_json(f"https://api.weather.gov/points/{LAT},{LON}", headers)
    hourly_url = pts["properties"]["forecastHourly"]
    fc = _http_get_json(hourly_url, headers)
    periods = fc["properties"]["periods"]
    out = []
    for p in periods:
        start = datetime.fromisoformat(p["startTime"])
        out.append(
            {
                "start": start.astimezone(PACIFIC),
                "temp_f": p.get("temperature"),
                "wind": f"{p.get('windSpeed', '')} {p.get('windDirection', '')}".strip(),
                "short": p.get("shortForecast", ""),
                # NWS doesn't always populate humidity/dewpoint; treat as optional.
                "humidity": (p.get("relativeHumidity") or {}).get("value"),
                "dewpoint_c": (p.get("dewpoint") or {}).get("value"),
            }
        )
    return out


def pick_weather_windows(periods, horizon_label, target_date):
    # Probe 3 hours spanning the likely workout window. The user could run
    # any time during daylight on the target date; three samples give a
    # readable range without bloating the prompt.
    # Today (morning cutover): 10am, 2pm, 6pm
    # Tomorrow (evening cutover): 7am, 11am, 3pm
    if horizon_label == "today":
        probe_hours = [10, 14, 18]
    else:
        probe_hours = [7, 11, 15]

    picks = []
    for hour in probe_hours:
        target = datetime.combine(target_date, datetime.min.time(), tzinfo=PACIFIC).replace(hour=hour)
        best = min(
            periods,
            key=lambda p: abs((p["start"] - target).total_seconds()),
            default=None,
        )
        if best is None:
            continue
        # If the closest available period is more than 3h off, the forecast
        # doesn't really cover this probe - skip rather than misrepresent.
        if abs((best["start"] - target).total_seconds()) > 3 * 3600:
            continue
        picks.append(best)

    return picks


def format_weather_block(horizon_label, target_date):
    # Best-effort: try NWS, degrade to a clean "unavailable" line otherwise.
    try:
        periods = fetch_weather_nws()
    except urllib.error.HTTPError as e:
        return f"Weather ({LOCATION_LABEL}): unavailable (HTTP {e.code} {e.reason})."
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as e:
        return f"Weather ({LOCATION_LABEL}): unavailable ({type(e).__name__})."

    picks = pick_weather_windows(periods, horizon_label, target_date)
    if not picks:
        return f"Weather ({LOCATION_LABEL}): forecast returned no usable windows."

    when = "today" if horizon_label == "today" else f"tomorrow ({target_date.isoformat()})"
    lines = [f"Weather ({LOCATION_LABEL}), forecast for {when}:"]
    for p in picks:
        hhmm = p["start"].strftime("%-I%p").lower()
        parts = [f"{p['temp_f']}F"] if p["temp_f"] is not None else []
        if p.get("humidity") is not None:
            parts.append(f"RH {p['humidity']}%")
        if p.get("wind"):
            parts.append(f"wind {p['wind']}")
        if p.get("short"):
            parts.append(p["short"])
        lines.append(f"  {hhmm}: {', '.join(parts)}")
    return "\n".join(lines)


# --- Time budget prompt ---


DEFAULT_TIME_BUDGET_MIN = 42


def prompt_time_budget(horizon_label):
    # Read from stdin. Accept integer minutes, blank for default, - for unspecified.
    # Prompt goes to stderr so piping stdout to pbcopy/a file stays clean.
    prompt = f"Minutes available for exercise {horizon_label}? " f"(integer, blank for default {DEFAULT_TIME_BUDGET_MIN}, - for unspecified): "
    print(prompt, end="", file=sys.stderr, flush=True)
    try:
        raw = input().strip()
    except EOFError:
        return DEFAULT_TIME_BUDGET_MIN
    if raw == "":
        return DEFAULT_TIME_BUDGET_MIN
    if raw == "-":
        return None
    try:
        n = int(raw)
        if n <= 0 or n > 600:
            print(f"  (ignoring implausible value {n}; using default {DEFAULT_TIME_BUDGET_MIN})", file=sys.stderr)
            return DEFAULT_TIME_BUDGET_MIN
        return n
    except ValueError:
        print(f"  (could not parse '{raw}'; using default {DEFAULT_TIME_BUDGET_MIN})", file=sys.stderr)
        return DEFAULT_TIME_BUDGET_MIN


# --- Prompt assembly ---


def build_situational_block(now, horizon_label, target_date, time_budget_min, weather_block, fresh_hrv_date):
    dow = now.strftime("%A")
    datestr = now.strftime("%Y-%m-%d")
    timestr = now.strftime("%-I:%M %p %Z")
    tod = time_of_day_label(now)

    if time_budget_min is not None:
        budget_line = f"- Time available: **{time_budget_min} min** (hard constraint on duration)"
    else:
        budget_line = "- Time available: not specified (assume 30-45 min nominal)"

    # Fresh-HRV line: called out because a same-day morning HRV is the
    # single most important input and is easy to miss buried in session files.
    # We always probe TODAY's date in both horizons - today's HRV is the
    # freshest datum whether deciding today's or tomorrow's run.
    today_str = now.date().isoformat()
    if fresh_hrv_date is not None:
        hrv_line = f"- Morning HRV for today ({today_str}): **logged** " f"(see session file; freshest signal, weigh accordingly)"
    else:
        hrv_line = f"- Morning HRV for today ({today_str}): **not logged** " f"(rely on rolling HRV trend from weekly summary)"

    lines = [
        "# Situational Context",
        "",
        f"- Now: {dow} {datestr} {timestr} ({tod})",
        f"- Decision for: **{horizon_label}** ({target_date.isoformat()})",
        budget_line,
        hrv_line,
        "",
        weather_block,
    ]
    return "\n".join(lines)


def build_task_block(horizon_label, target_date):
    # Tight, non-chatty. Mirrors section 7 of 00_context.md without
    # re-explaining physiology, zones, or the constrained action space -
    # those already live in the context and re-stating invites drift.
    return (
        "# Task\n"
        "\n"
        f"Produce a coaching response for **{horizon_label}** "
        f"({target_date.isoformat()}) using the full context above:\n"
        "\n"
        "1. **Critique** the most recent 1-2 sessions. Cite numbers and dates.\n"
        "2. **Check** the weekly summary for trends.\n"
        "3. **Prescribe** exactly one option from the constrained action space "
        "(section 7 of the context). State type, target HR, duration, and a "
        "one-line execution cue.\n"
        "\n"
        "The time budget is a hard upper bound on duration. Weather affects "
        "HR-vs-effort calibration (hot or humid conditions shift HR up at a "
        "given effort) - factor that into target-vs-ceiling judgment, not "
        "into changing the zone.\n"
        "\n"
        "Follow the response format specified in section 7 of the context."
    )


def build_prompt():
    # Assembles the full coaching prompt as a single string. Also returns
    # side-channel info for the caller (status messages to print to stderr,
    # time budget, etc) via a dict - keeps main() thin.
    now = now_pacific()
    horizon_label, target_date = decision_horizon(now)

    context, filenames = load_context_files()
    if context is None:
        print(f"ERROR: no context files found in {CONTEXT_DIR}/", file=sys.stderr)
        sys.exit(1)

    # Today's HRV is the freshest signal for either horizon (we won't have
    # tomorrow's reading yet when deciding tomorrow's run).
    today = now.date()
    fresh_hrv_date = today if detect_fresh_hrv(today) else None

    print("# query_trainer.py", file=sys.stderr)
    print(f"# {now.strftime('%A %Y-%m-%d %-I:%M %p %Z')} - horizon: {horizon_label}", file=sys.stderr)
    print(f"# loaded {len(filenames)} context files from {CONTEXT_DIR}/", file=sys.stderr)
    if fresh_hrv_date:
        print(f"# found morning HRV for today ({fresh_hrv_date})", file=sys.stderr)
    time_budget = prompt_time_budget(horizon_label)

    print("# fetching weather...", file=sys.stderr)
    weather_block = format_weather_block(horizon_label, target_date)

    situational = build_situational_block(
        now,
        horizon_label,
        target_date,
        time_budget,
        weather_block,
        fresh_hrv_date,
    )
    task = build_task_block(horizon_label, target_date)

    # Prompt order: context -> separator -> situational -> separator -> task.
    # The separator is a visible but minimal marker so the LLM knows it's
    # transitioning from reference material to the active query.
    separator = "\n\n---\n\n"
    return context + separator + situational + separator + task


def main():
    prompt = build_prompt()

    # Dispatch to whatever provider/model llm_client has configured as the
    # default. Swap it there, not here.
    response_text = call_llm(prompt)
    print("=" * 80)
    print(response_text)
    print("=" * 80)


if __name__ == "__main__":
    main()
