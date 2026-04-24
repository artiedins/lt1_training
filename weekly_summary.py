#!/usr/bin/env python3

import re
import statistics
from datetime import datetime, timedelta, date
from pathlib import Path

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
SUMMARY_PATH = CONTEXT_DIR / "10_weekly_summary.md"

# Rolling window durations (days)
SHORT_WINDOW = 7
LONG_WINDOW = 28
# Minimum samples to report a median at all. Set to 1 so early sparse data
# is not hidden. Sample counts are always shown alongside so the LLM can
# apply the correct weight to each number.
MIN_SAMPLES_SHORT = 1
MIN_SAMPLES_LONG = 1

# Below this sample count, a "median" over runs is reported as a single
# reading rather than as a trend, to avoid misleading a downstream LLM
# into treating N=1 or N=2 as a trend signal.
TREND_MIN_N = 3


# --- Parsing per-session markdown ---

# Filename format: 20_session_YYYY-M-DD[_N].md  (N is same-day suffix)
SESSION_FILE_RE = re.compile(r"20_session_(\d{4})-(\d{1,2})-(\d{1,2})(?:_(\d+))?\.md$")


def parse_date_from_filename(path):
    m = SESSION_FILE_RE.search(path.name)
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        return datetime(y, mo, d).date()
    except ValueError:
        return None


def extract_metric(text, pattern, cast=float):
    m = re.search(pattern, text)
    if not m:
        return None
    try:
        return cast(m.group(1))
    except (ValueError, IndexError):
        return None


def parse_session_file(path):
    try:
        text = path.read_text()
    except Exception:
        return None

    date = parse_date_from_filename(path)
    if date is None:
        return None

    rec = {"date": date, "path": path, "type": None}

    if "Morning HRV" in text:
        rec["type"] = "morning_hrv"
        rec["rhr"] = extract_metric(text, r"Resting HR \| ([\d.]+) bpm")
        rec["rmssd"] = extract_metric(text, r"RMSSD \| ([\d.]+) ms")
        rec["rmssd_med"] = extract_metric(text, r"RMedSSD \| ([\d.]+) ms")
        rec["stdrr"] = extract_metric(text, r"stdRR \(calm window\) \| ([\d.]+) ms")
        rec["breathing_rate"] = extract_metric(text, r"Breathing rate \(est\) \| ([\d.]+) /min")
    elif "Easy Aerobic Run" in text:
        rec["type"] = "easy_aerobic"
        rec["duration_min"] = extract_metric(text, r"Duration \| ([\d.]+) min")
        rec["aerobic_mean_hr"] = extract_metric(text, r"Aerobic-portion mean HR \| ([\d.]+) bpm")
        rec["drift"] = extract_metric(text, r"Cardiac drift:.*?([+-]?\d+\.\d+) bpm \(")
        rec["peak_hr"] = extract_metric(text, r"End push:\*\* peak (\d+) bpm", cast=int)
        rec["hrr_30"] = extract_metric(text, r"HRR_30 (\d+)", cast=int)
        rec["hrr_60"] = extract_metric(text, r"HRR_60 (\d+)", cast=int)
        rec["hrr_90"] = extract_metric(text, r"HRR_90 (\d+)", cast=int)
        rec["hrr_120"] = extract_metric(text, r"HRR_120 (\d+)", cast=int)
        rec["push_detected"] = "**End push:**" in text

        # Time-in-band %s. Compressed format suppresses zero-rows; missing
        # fields stay None and are treated as 0% when averaged.
        for band_key, pattern_desc in [
            ("pct_below_target", r"below target \(0-117\)"),
            ("pct_true_z2", r"true Z2 \(target\) \(118-125\)"),
            ("pct_mild_overshoot", r"mild overshoot \(126-140\)"),
            ("pct_sig_overshoot", r"significant overshoot \(VT2 territory\) \(141-152\)"),
            ("pct_above_vt2", r"above VT2 \(>=153\)"),
        ]:
            rec[band_key] = extract_metric(
                text,
                r"\| " + pattern_desc + r" \| \d+s \| (\d+)%",
                cast=int,
            )
    elif "Rest Day" in text:
        rec["type"] = "rest_day"

    return rec


def load_all_sessions():
    recs = []
    for p in sorted(CONTEXT_DIR.glob("20_session_*.md")):
        rec = parse_session_file(p)
        if rec and rec["type"]:
            recs.append(rec)
    recs.sort(key=lambda r: r["date"])
    return recs


# --- Rolling metrics ---


def values_in_window(recs, field, days_back, today, session_type=None):
    cutoff = today - timedelta(days=days_back)
    vals = []
    for r in recs:
        if r["date"] < cutoff or r["date"] > today:
            continue
        if session_type and r["type"] != session_type:
            continue
        v = r.get(field)
        if v is not None:
            vals.append(v)
    return vals


def median_or_none(vals, min_n):
    if len(vals) < min_n:
        return None
    return round(statistics.median(vals), 1)


def cv_pct(vals):
    if len(vals) < 2:
        return None
    m = statistics.mean(vals)
    if m == 0:
        return None
    return round(statistics.stdev(vals) / m * 100, 1)


def trend_arrow(short, long):
    if short is None or long is None:
        return "-"
    delta = round(short - long, 1)
    if abs(delta) < 0.5:
        return f"flat ({delta:+})"
    arrow = "up" if delta > 0 else "down"
    return f"{arrow} ({delta:+})"


# --- Autonomic drift flag ---


def autonomic_drift_flag(rmssd_short, rmssd_long, rhr_short, rhr_long, sessions_short=None, sessions_long=None):
    have_hrv = rmssd_short is not None and rmssd_long is not None
    have_rhr = rhr_short is not None and rhr_long is not None

    if not have_hrv and not have_rhr:
        return None, "deferred", "insufficient data"
    if not have_hrv:
        return None, "deferred", f"RHR trend available ({rhr_short} vs {rhr_long}), HRV not - cannot check drift"
    if not have_rhr:
        return None, "deferred", f"HRV trend available ({rmssd_short} vs {rmssd_long}), RHR not - cannot check drift"

    hrv_down = rmssd_short < rmssd_long
    rhr_down = rhr_short < rhr_long

    # Precondition: the overcook interpretation requires actual training.
    # Threshold: at least 3 easy-aerobic sessions in the short window.
    # Below that, HRV+RHR decline is probably not from training stress.
    MIN_SESSIONS_FOR_OVERCOOK = 3
    training_sufficient = sessions_short is not None and sessions_short >= MIN_SESSIONS_FOR_OVERCOOK

    if hrv_down and rhr_down:
        if not training_sufficient:
            # The both-down-but-not-from-training case. Worth explaining in
            # full because a downstream LLM pattern-matching "HRV+RHR both
            # declining = overtraining" is the exact failure mode to prevent.
            n = sessions_short or 0
            return (
                False,
                "concern",
                (
                    f"HRV short ({rmssd_short}) < long ({rmssd_long}) AND "
                    f"RHR short ({rhr_short}) < long ({rhr_long}), but only "
                    f"{n} training session{'s' if n != 1 else ''} in short window - "
                    f"HRV/RHR pattern not attributable to training overload. "
                    f"Consider life stress, illness, or detraining; do NOT deload."
                ),
            )
        return (
            True,
            "concern",
            (
                f"HRV short ({rmssd_short}) < long ({rmssd_long}) AND "
                f"RHR short ({rhr_short}) < long ({rhr_long}) with "
                f"{sessions_short} sessions in short window. "
                f"Autonomic-overcook pattern; consider deload."
            ),
        )

    if hrv_down and not rhr_down:
        return False, "normal", "HRV down but RHR steady - likely acute stress, not chronic overcook"
    return False, "normal", "no drift concern"


# --- Report generation ---


def generate_summary(recs, today=None):
    if today is None:
        today = datetime.now().date()  # TODO: CHANGE TO BE LOCAL TIME, not UTC

    if not recs:
        return "# Weekly Rolling Summary\n\nNo session data found.\n"

    last_session_date = recs[-1]["date"]
    days_since_any = (today - last_session_date).days

    # Transition-era caveat: surface only when the long window contains BOTH
    # pre- and post-transition sessions. Before the first post-transition
    # session, 00_context.md section 9 covers the "everything is pre-transition"
    # case. Once everything in the long window is post-transition, no caveat
    # needed.
    TRANSITION_DATE = date(2026, 4, 16)
    long_window_cutoff = last_session_date - timedelta(days=LONG_WINDOW)
    has_pre = any(r["date"] < TRANSITION_DATE and r["date"] >= long_window_cutoff for r in recs)
    has_post = any(r["date"] >= TRANSITION_DATE for r in recs)
    transition_note = None
    # if has_pre and has_post:
    #    transition_note = (
    #        f"**Transition caveat:** The {LONG_WINDOW}d window straddles "
    #        f"the {TRANSITION_DATE.isoformat()} protocol transition. Pre-transition "
    #        f"runs were at the wrong HR. Morning HRV/RHR from pre-transition is "
    #        f"still valid (measurement protocol unchanged). Run-execution and "
    #        f"HRR metrics from pre-transition should be treated as historical, "
    #        f"not as a baseline the new protocol should match."
    #    )

    # Staleness strategy: if last session was within SHORT_WINDOW days, use
    # calendar-based 7d/28d windows. Otherwise, anchor windows on the last
    # session date so the report remains informative during gaps.
    if days_since_any > SHORT_WINDOW:
        anchor = last_session_date
        anchor_note = f" (windows anchored to last session {last_session_date.isoformat()}, " f"{days_since_any} days ago)"
    else:
        anchor = today
        anchor_note = ""

    # --- Morning HRV rollups ---
    rhr_s = values_in_window(recs, "rhr", SHORT_WINDOW, anchor, "morning_hrv")
    rhr_l = values_in_window(recs, "rhr", LONG_WINDOW, anchor, "morning_hrv")

    rmssd_s = values_in_window(recs, "rmssd", SHORT_WINDOW, anchor, "morning_hrv")
    rmssd_l = values_in_window(recs, "rmssd", LONG_WINDOW, anchor, "morning_hrv")

    rmssd_med_s = values_in_window(recs, "rmssd_med", SHORT_WINDOW, anchor, "morning_hrv")
    rmssd_med_l = values_in_window(recs, "rmssd_med", LONG_WINDOW, anchor, "morning_hrv")

    stdrr_s = values_in_window(recs, "stdrr", SHORT_WINDOW, anchor, "morning_hrv")
    stdrr_l = values_in_window(recs, "stdrr", LONG_WINDOW, anchor, "morning_hrv")

    rhr_s_med = median_or_none(rhr_s, MIN_SAMPLES_SHORT)
    rhr_l_med = median_or_none(rhr_l, MIN_SAMPLES_LONG)

    rmssd_s_med = median_or_none(rmssd_s, MIN_SAMPLES_SHORT)
    rmssd_l_med = median_or_none(rmssd_l, MIN_SAMPLES_LONG)
    rmssd_l_cv = cv_pct(rmssd_l)

    rmssd_med_s_med = median_or_none(rmssd_med_s, MIN_SAMPLES_SHORT)
    rmssd_med_l_med = median_or_none(rmssd_med_l, MIN_SAMPLES_LONG)
    rmssd_med_l_cv = cv_pct(rmssd_med_l)

    stdrr_s_med = median_or_none(stdrr_s, MIN_SAMPLES_SHORT)
    stdrr_l_med = median_or_none(stdrr_l, MIN_SAMPLES_LONG)

    # --- Run rollups ---
    def runs_within(days):
        return [r for r in recs if r["type"] == "easy_aerobic" and 0 <= (anchor - r["date"]).days < days]

    runs_s = runs_within(SHORT_WINDOW)
    runs_l = runs_within(LONG_WINDOW)

    def sum_field(runs, field):
        vals = [r.get(field) for r in runs if r.get(field) is not None]
        return round(sum(vals), 1) if vals else 0

    def avg_pct(runs, field):
        vals = [r.get(field) or 0 for r in runs if r["type"] == "easy_aerobic"]
        return round(statistics.mean(vals), 0) if vals else None

    total_min_s = sum_field(runs_s, "duration_min")
    total_min_l = sum_field(runs_l, "duration_min")

    # HRR rollups. Collect samples with their dates so we can render a
    # clean single-point summary when N < TREND_MIN_N.
    def samples_with_dates(runs, field):
        return [(r["date"], r[field]) for r in runs if r.get(field) is not None]

    hrr_60_s_samples = samples_with_dates(runs_s, "hrr_60")
    hrr_60_l_samples = samples_with_dates(runs_l, "hrr_60")

    drift_s = [r["drift"] for r in runs_s if r.get("drift") is not None]
    drift_s_med = median_or_none(drift_s, MIN_SAMPLES_SHORT) if drift_s else None

    pct_true_z2_s = avg_pct(runs_s, "pct_true_z2")
    pct_mild_over_s = avg_pct(runs_s, "pct_mild_overshoot")
    pct_sig_over_s = avg_pct(runs_s, "pct_sig_overshoot")

    # Days since last "hard" session (peak >=150 with confirmed push)
    hard_sessions = [r for r in recs if r.get("peak_hr") is not None and r["peak_hr"] >= 150 and r.get("push_detected")]
    days_since_hard = (today - max(r["date"] for r in hard_sessions)).days if hard_sessions else None

    drift_flag, drift_severity, drift_note = autonomic_drift_flag(
        rmssd_s_med,
        rmssd_l_med,
        rhr_s_med,
        rhr_l_med,
        sessions_short=len(runs_s),
        sessions_long=len(runs_l),
    )

    # --- Render ---
    lines = [
        "# Weekly Rolling Summary",
        f"*Generated {today.isoformat()}.{anchor_note}*",
        "",
        f"**Last session:** {last_session_date.isoformat()} ({days_since_any}d ago). "
        f"**Last session with confirmed push (peak >=150):** "
        f"{(days_since_hard if days_since_hard is not None else 'none on record')}"
        f"{'d ago' if days_since_hard is not None else ''}.",
        "",
    ]
    if transition_note:
        lines += [transition_note, ""]
    lines += [
        "## Morning HRV",
        "",
        f"| Metric | {SHORT_WINDOW}d median (N={len(rhr_s)}) | {LONG_WINDOW}d median (N={len(rhr_l)}) | delta |",
        "|---|---|---|---|",
        f"| RHR | {_fmt(rhr_s_med)} bpm | {_fmt(rhr_l_med)} bpm | {trend_arrow(rhr_s_med, rhr_l_med)} |",
        f"| RMSSD | {_fmt(rmssd_s_med)} ms | {_fmt(rmssd_l_med)} ms | {trend_arrow(rmssd_s_med, rmssd_l_med)} (normal RMSSD) |",
        f"| RMedSSD | {_fmt(rmssd_med_s_med)} ms | {_fmt(rmssd_med_l_med)} ms | {trend_arrow(rmssd_med_s_med, rmssd_med_l_med)} (this is rt-median-ssd with scaling factors to approximate normal RMSSD)|",
        f"| stdRR | {_fmt(stdrr_s_med)} ms | {_fmt(stdrr_l_med)} ms | {trend_arrow(stdrr_s_med, stdrr_l_med)} |",
        "",
        f"RMSSD {LONG_WINDOW}d CV: {_fmt(rmssd_l_cv)}%. " "Target RMSSD: 65 ms (multi-year move; weekly changes are noise).",
        "",
        "## Autonomic Drift Check",
        "",
    ]
    # Severity-weighted display: "concern" gets a big block, "normal" gets a
    # one-liner, "deferred" gets a parenthetical. Prevents no-concern output
    # from reading as alarming.
    if drift_flag is True:
        lines.append(f"**FLAG RAISED:** {drift_note}")
    elif drift_severity == "concern":
        lines.append(f"**No flag, but note:** {drift_note}")
    elif drift_severity == "normal":
        lines.append(f"No flag: {drift_note}.")
    else:
        lines.append(f"*(deferred: {drift_note})*")
    lines.append("")

    lines += [
        "## Training Volume & Execution",
        "",
        f"| | Last {SHORT_WINDOW}d | Last {LONG_WINDOW}d |",
        "|---|---|---|",
        f"| Sessions | {len(runs_s)} | {len(runs_l)} |",
        f"| Total minutes | {total_min_s} | {total_min_l} |",
        "",
    ]

    if runs_s:
        lines += [
            f"**Execution (last {SHORT_WINDOW}d, avg % of aerobic portion):** "
            f"true Z2: {_fmt(pct_true_z2_s)}%, "
            f"mild overshoot: {_fmt(pct_mild_over_s)}%, "
            f"sig overshoot: {_fmt(pct_sig_over_s)}%. Target: true Z2 >=80%.",
            "",
        ]

    lines += ["## HRR", ""]
    lines += _format_hrr_section(hrr_60_s_samples, hrr_60_l_samples)
    lines += [
        f"Within-run cardiac drift {SHORT_WINDOW}d median: {_fmt(drift_s_med)} bpm.",
        "",
        "Lab baseline HRR_60: 29 bpm (rested likely 32-36+). Rising trend = parasympathetic reactivation improving.",
        "",
    ]

    return "\n".join(lines)


def _format_hrr_section(short_samples, long_samples):
    out = []
    n_s = len(short_samples)
    n_l = len(long_samples)

    if n_l == 0:
        out.append("HRR_60: no confirmed push sessions on record yet.")
    elif n_l < TREND_MIN_N:
        # Sparse data: list individual readings rather than compute a median.
        readings = ", ".join(f"{v} bpm on {d.isoformat()}" for d, v in long_samples)
        plural = "reading" if n_l == 1 else "readings"
        out.append(f"HRR_60: {n_l} confirmed-push {plural} on record ({readings}). " f"Insufficient data for a trend; need N>={TREND_MIN_N}.")
    else:
        s_med = median_or_none([v for _, v in short_samples], MIN_SAMPLES_SHORT)
        l_med = median_or_none([v for _, v in long_samples], MIN_SAMPLES_LONG)
        out.append(f"HRR_60 {SHORT_WINDOW}d median: {_fmt(s_med)} bpm (N={n_s}). " f"{LONG_WINDOW}d: {_fmt(l_med)} bpm (N={n_l}).")
    out.append("")
    return out


def _fmt(v):
    return "-" if v is None else str(v)


def main():
    recs = load_all_sessions()
    if not recs:
        print(f"No session files found in {CONTEXT_DIR}/")
        return
    md = generate_summary(recs)
    SUMMARY_PATH.write_text(md + "\n")
    print(f"Wrote {SUMMARY_PATH} ({len(recs)} sessions parsed).")


if __name__ == "__main__":
    main()
