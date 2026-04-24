#!/usr/bin/env python3

import os
import csv
import yaml
import numpy as np
from datetime import datetime
from pathlib import Path

from process_morning_hrv import compute_rhr_hrv_from_rr_data

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


INPUT_DIR = "./hrv_input"
OUTPUT_DIR = "./hrv_context"

SESSION_TYPES = [
    "morning_hrv",
    "easy_aerobic",
    "rest_day",
    "zone2_continuous",
    "zone2_drift_monitor",
]

ZONE_BANDS = [
    ("below_target", 0, 118, "below target"),
    ("true_z2", 118, 126, "true Z2 (target)"),
    ("mild_overshoot", 126, 141, "mild overshoot"),
    ("significant_overshoot", 141, 153, "significant overshoot (VT2 territory)"),
    ("above_vt2", 153, 999, "above VT2"),
]

# --- Push detection constants ---

# Minimum peak HR to consider a session as having an intentional end push.
PUSH_DETECTION_MIN_PEAK = 140

# Peak must occur within this many seconds of recording end. Protocol is
# "push in last 30-60s, walk 2-7 min with recording running". Anchoring to
# last 4 min rejects mid-run drift peaks from old-protocol sessions.
PUSH_PEAK_MUST_BE_WITHIN_END_SEC = 4 * 60

# Look-back window for finding peak.
PUSH_SEARCH_WINDOW_SEC = 8 * 60

# Rise check: peak - min(HR in pre-push window) must be >= PUSH_MIN_RISE_BPM.
# Window is [peak - LO, peak - HI]. The HI exclusion keeps the window outside
# the push itself; the LO extent is long enough to find the pre-push baseline
# even when the ramp from Z2 to push HR takes 90-150s.
PUSH_MIN_RISE_BPM = 15
PUSH_RISE_WINDOW_LO_SEC = 180
PUSH_RISE_WINDOW_HI_SEC = 30

# HRR sample points (seconds after peak). Ordered; output respects this order.
HRR_OFFSETS = [30, 60, 90, 120]


# --- Utilities ---


def _ts_to_sec(ts):
    if not ts:
        return 0
    ts = ts.strip()
    if ":" in ts:
        try:
            parts = ts.split(":")
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(float(parts[2]))
        except Exception:
            return 0
    try:
        return int(float(ts))
    except Exception:
        return 0


# --- File I/O ---


def find_sessions(input_dir):
    sessions = []
    for yaml_path in sorted(Path(input_dir).glob("*.yaml")):
        prefix = yaml_path.stem.replace("_session_", "")
        sessions.append((prefix, yaml_path))
    return sessions


def find_csv(input_dir, prefix, suffix):
    p = Path(input_dir) / f"{prefix}_{suffix}_.csv"
    return p if p.exists() else None


def parse_csv(path):
    if path is None or not Path(path).exists():
        return []
    rows = []
    with open(path, newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        header_keys = {k.strip().lower() for k in reader.fieldnames if k}
        for row in reader:
            clean = {k.strip(): v.strip() for k, v in row.items() if k}
            # HRV Logger re-exports sometimes inline a header row mid-file
            if {v.lower() for v in clean.values()} & header_keys:
                continue
            rows.append(clean)
    return rows


# --- Multi-session file handling ---


def split_sessions(rows, gap_minutes=30):
    if not rows:
        return []

    def parse_wall(s):
        try:
            return datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S %z")
        except Exception:
            return None

    sessions, current = [], [rows[0]]
    prev_wall = parse_wall(rows[0].get("date", ""))
    for row in rows[1:]:
        wall = parse_wall(row.get("date", ""))
        if wall and prev_wall and (wall - prev_wall).total_seconds() > gap_minutes * 60:
            sessions.append(current)
            current = [row]
        else:
            current.append(row)
        if wall:
            prev_wall = wall
    sessions.append(current)
    return sessions


def select_session(sessions, strategy="last"):
    if not sessions:
        return []
    if isinstance(strategy, int):
        return sessions[min(strategy, len(sessions) - 1)]
    if strategy == "first":
        return sessions[0]
    if strategy == "last":
        return sessions[-1]

    def mean_hr(sess):
        vals = []
        for r in sess:
            hr_key = next(
                (k for k in r if "heart" in k.lower() and "rate" in k.lower()),
                None,
            )
            if hr_key:
                try:
                    vals.append(float(r[hr_key]))
                except (ValueError, TypeError):
                    pass
        return sum(vals) / len(vals) if vals else 0.0

    scored = [(i, mean_hr(s)) for i, s in enumerate(sessions)]
    best = max(scored, key=lambda x: x[1]) if strategy == "highest_hr" else min(scored, key=lambda x: x[1])
    return sessions[best[0]]


def _strategy_for_type(session_type):
    if session_type in ("easy_aerobic", "zone2_continuous", "zone2_drift_monitor"):
        return "highest_hr"
    return "last"


def load_session_rows(path, session_type):
    all_rows = parse_csv(path)
    sessions = split_sessions(all_rows)
    return select_session(sessions, strategy=_strategy_for_type(session_type))


# --- Data utilities ---


def deduplicate_features(rows):
    seen, out = set(), []
    for r in rows:
        key = (r.get("date", ""), r.get("timestamp", ""))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def deduplicate_hr(rows):
    seen, out = set(), []
    for r in rows:
        key = r.get("timestamp", "")
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def hr_array(hr_rows):
    return np.array([int(r["heart rate"]) for r in hr_rows])


def time_in_bands(hrs):
    counts = {name: 0 for name, _, _, _ in ZONE_BANDS}
    for h in hrs:
        for name, lo, hi, _ in ZONE_BANDS:
            if lo <= h < hi:
                counts[name] += 1
                break
    return counts


def cardiac_drift(hrs_array, start_pct=0.25, end_pct=0.75, window=120):
    n = len(hrs_array)
    early_start = int(n * start_pct)
    early_end = early_start + window
    late_end = int(n * end_pct)
    late_start = late_end - window
    if early_end > late_start or late_end > n:
        return None, None, None
    early = np.mean(hrs_array[early_start:early_end])
    late = np.mean(hrs_array[late_start:late_end])
    return round(float(early), 1), round(float(late), 1), round(float(late - early), 1)


def detect_push_and_hrr(hrs_array, ts_array):
    if len(hrs_array) == 0:
        return {"push_detected": False, "reason": "no data"}

    secs = np.array([_ts_to_sec(t) for t in ts_array])
    if secs[-1] == 0:
        return {"push_detected": False, "reason": "no timestamps"}

    t_end = secs[-1]
    search_mask = secs >= (t_end - PUSH_SEARCH_WINDOW_SEC)
    if not search_mask.any():
        return {"push_detected": False, "reason": "recording too short"}

    search_hrs = hrs_array[search_mask]
    search_secs = secs[search_mask]
    peak_local = int(np.argmax(search_hrs))
    peak_hr = int(search_hrs[peak_local])
    peak_sec = int(search_secs[peak_local])

    # Gate 1: peak amplitude
    if peak_hr < PUSH_DETECTION_MIN_PEAK:
        return {
            "push_detected": False,
            "reason": f"peak {peak_hr} bpm < {PUSH_DETECTION_MIN_PEAK} threshold",
            "peak_hr": peak_hr,
        }

    # Gate 2: peak position (must be near end of recording)
    time_from_end = t_end - peak_sec
    if time_from_end > PUSH_PEAK_MUST_BE_WITHIN_END_SEC:
        return {
            "push_detected": False,
            "reason": (f"peak {peak_hr} bpm occurred {time_from_end}s before end of " f"recording (> {PUSH_PEAK_MUST_BE_WITHIN_END_SEC}s cutoff); " f"likely mid-run drift, not a deliberate end push"),
            "peak_hr": peak_hr,
        }

    # Gate 3: rise from pre-push baseline
    # min(HR in [peak-180s, peak-30s]) captures the floor before the push
    # regardless of ramp duration.
    rise_mask = (secs >= peak_sec - PUSH_RISE_WINDOW_LO_SEC) & (secs < peak_sec - PUSH_RISE_WINDOW_HI_SEC)
    if rise_mask.any():
        baseline = int(np.min(hrs_array[rise_mask]))
        rise = peak_hr - baseline
        if rise < PUSH_MIN_RISE_BPM:
            return {
                "push_detected": False,
                "reason": (
                    f"peak {peak_hr} bpm but only {rise} bpm above pre-push "
                    f"baseline ({baseline} bpm in "
                    f"[peak-{PUSH_RISE_WINDOW_LO_SEC}s, "
                    f"peak-{PUSH_RISE_WINDOW_HI_SEC}s]); "
                    f"session drifted up rather than pushed"
                ),
                "peak_hr": peak_hr,
            }

    # Sample HRR curve
    def hr_at(target_sec):
        if target_sec > t_end:
            return None
        diffs = np.abs(secs - target_sec)
        if len(diffs) == 0:
            return None
        idx = int(np.argmin(diffs))
        if diffs[idx] > 3:
            return None
        return int(hrs_array[idx])

    result = {
        "push_detected": True,
        "peak_hr": peak_hr,
        "peak_elapsed_sec": peak_sec,
    }
    for off in HRR_OFFSETS:
        v = hr_at(peak_sec + off)
        result[f"hr_{off}"] = v
        result[f"hrr_{off}"] = (peak_hr - v) if v is not None else None
    return result


# --- Session processors ---


def process_morning_hrv(prefix, notes, rr_path):
    if rr_path is None or not rr_path.exists():
        return _format_morning_hrv_missing(prefix, notes)

    r = compute_rhr_hrv_from_rr_data(str(rr_path))

    rmssd = r["rmssd_ms"]
    rmssd_median = r["rmssd_ms_median"]

    br = r["breathing_rate_est"]
    br_line = f"| Breathing rate (est) | {br} /min |" if br is not None else ""

    lines = [f"\n## {prefix} - Morning HRV"]
    if notes and notes.strip() and notes.strip().lower() != "no notes":
        lines.append(f"**Notes:** {notes.strip()}")
    lines += [
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Resting HR | {r['rest_hr_bpm']} bpm |",
        f"| RMSSD | {rmssd} ms (normal rt-mean-ssd) |",
        f"| RMedSSD | {rmssd_median} ms (rt-median-ssd with scaling factors to approximate RMSSD) |",
        f"| stdRR (calm window) | {r['stdrr_ms']} ms |",
        br_line,
    ]
    return "\n".join(l for l in lines if l != "")


def _format_morning_hrv_missing(prefix, notes):
    return f"\n## {prefix} - Morning HRV\n" f"**Notes:** {notes or '-'}\n\n" f"No RR data file found."


def process_easy_aerobic(
    prefix,
    session_type,
    notes,
    feature_rows,
    hr_rows,
    events_rows,
    push_expected=True,
):
    features = deduplicate_features(feature_rows)
    hrs_raw = deduplicate_hr(hr_rows)

    if not hrs_raw:
        return f"\n## {prefix} - Easy Aerobic Run\n\n(no HR data found)"

    hrs_full = hr_array(hrs_raw)
    ts_full = [r["timestamp"] for r in hrs_raw]

    # Trim to "run portion" - from first 10-sample mean >110 bpm to end
    onset_idx = 0
    for i in range(len(hrs_full) - 10):
        if np.mean(hrs_full[i : i + 10]) > 110:
            onset_idx = i
            break
    hrs_run = hrs_full[onset_idx:]
    ts_run = ts_full[onset_idx:]

    hr_min = int(np.min(hrs_run))
    hr_max = int(np.max(hrs_run))
    duration_min = round(len(hrs_run) / 60, 1)

    # Aerobic portion excludes the last 3 min so bands reflect easy execution.
    aerobic_cutoff = max(0, len(hrs_run) - 180)
    hrs_aerobic = hrs_run[:aerobic_cutoff] if aerobic_cutoff > 60 else hrs_run
    band_counts = time_in_bands(hrs_aerobic)
    total = len(hrs_aerobic)
    aerobic_mean = round(float(np.mean(hrs_aerobic)), 1) if total else 0

    if aerobic_mean <= 125:
        exec_label = "on-target"
    elif aerobic_mean <= 130:
        exec_label = "slight overshoot"
    elif aerobic_mean <= 140:
        exec_label = "significant overshoot"
    else:
        exec_label = "above ceiling - not easy-aerobic"

    early_hr, late_hr, drift = cardiac_drift(hrs_aerobic)
    if drift is None:
        drift_label = "n/a"
    elif drift < 3:
        drift_label = "excellent"
    elif drift < 7:
        drift_label = "acceptable"
    else:
        drift_label = "elevated - ran too hard or under-recovered"

    # Push detection: skipped entirely when not expected, so a no-push report
    # cannot be confused with a detector failure.
    if push_expected:
        push = detect_push_and_hrr(hrs_run, ts_run)
    else:
        push = None

    def pct(name):
        return round(band_counts[name] / total * 100, 0) if total else 0

    lines = [f"\n## {prefix} - Easy Aerobic Run"]
    if notes and notes.strip() and notes.strip().lower() != "no notes":
        lines.append(f"**Notes:** {notes.strip()}")
    lines += [
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Duration | {duration_min} min |",
        f"| Aerobic-portion mean HR | {aerobic_mean} bpm ({exec_label}) |",
        f"| HR range | {hr_min}-{hr_max} bpm |",
    ]

    band_rows = []
    for name, lo, hi, desc in ZONE_BANDS:
        if band_counts[name] == 0:
            continue
        hr_range = f"{lo}-{hi - 1}" if hi < 999 else f">={lo}"
        band_rows.append(f"| {desc} ({hr_range}) | {band_counts[name]}s | {pct(name):.0f}% |")
    if band_rows:
        lines += [
            "",
            "**Time in band (aerobic portion, excl. last 3 min):**",
            "",
            "| Band | Seconds | % |",
            "|------|---------|---|",
        ]
        lines += band_rows

    # if drift is not None:
    #    lines += [
    #        "",
    #        f"**Cardiac drift:** {early_hr}->{late_hr} bpm, {drift:+.1f} bpm " f"({drift_label})",
    #    ]

    # Push / HRR section
    if push is None:
        lines += ["", "**No end push (intentional).**"]
    elif push["push_detected"]:
        hrr_parts = []
        for off in HRR_OFFSETS:
            v = push.get(f"hrr_{off}")
            hrr_parts.append(f"HRR_{off} {_fmt(v)}")
        lines += [
            "",
            f"**End push:** peak {push['peak_hr']} bpm, " + " / ".join(hrr_parts) + " bpm",
        ]
    else:
        reason = push.get("reason", "unknown")
        lines += ["", f"**No push detected:** {reason}"]

    return "\n".join(l for l in lines if l is not None)


def _fmt(v):
    return str(v) if v is not None else "-"


def process_rest_day(prefix, notes):
    return f"\n## {prefix} - Rest Day\n" f"**Notes:** {notes or '-'}\n\n" f"No session. Logged for load accounting."


def process_generic(prefix, session_type, notes, feature_rows, hr_rows):
    hrs_raw = deduplicate_hr(hr_rows)
    hrs_arr = hr_array(hrs_raw) if hrs_raw else np.array([])

    lines = [
        f"\n## {prefix} - {session_type.replace('_', ' ').title()}",
        f"**Session type:** {session_type}",
        f"**Notes:** {notes or '-'}",
        "",
    ]
    if len(hrs_arr):
        lines.append(f"HR mean {np.mean(hrs_arr):.0f} bpm, " f"range {np.min(hrs_arr):.0f}-{np.max(hrs_arr):.0f} bpm, " f"duration ~{len(hrs_arr) // 60} min")
    lines.append("")
    lines.append("> No dedicated processor for this session type. Basic stats only.")
    return "\n".join(lines)


# --- Dispatch ---


def process_session(prefix, yaml_path, input_dir):
    with open(yaml_path) as f:
        meta = yaml.safe_load(f) or {}

    session_type = meta.get("session_type", "unknown").strip()
    notes = meta.get("notes", "")
    push_expected = meta.get("push_expected", True)

    if session_type not in SESSION_TYPES:
        print(f"  warn: unknown session_type '{session_type}' in {yaml_path.name}")

    if session_type == "morning_hrv":
        rr_path = find_csv(input_dir, prefix, "RR")
        print(f"     RR file: {rr_path or '(not found)'}")
        return process_morning_hrv(prefix, notes, rr_path)

    if session_type == "rest_day":
        return process_rest_day(prefix, notes)

    features_path = find_csv(input_dir, prefix, "Features")
    hr_path = find_csv(input_dir, prefix, "HR")
    events_path = find_csv(input_dir, prefix, "Events")

    feature_rows = load_session_rows(features_path, session_type)
    hr_rows = load_session_rows(hr_path, session_type)
    events_rows = parse_csv(events_path) if events_path else []

    print(f"     Loaded: {len(feature_rows)} feature rows, " f"{len(hr_rows)} HR rows, {len(events_rows)} event rows")

    if session_type in ("easy_aerobic", "zone2_continuous", "zone2_drift_monitor"):
        return process_easy_aerobic(
            prefix,
            session_type,
            notes,
            feature_rows,
            hr_rows,
            events_rows,
            push_expected=push_expected,
        )

    return process_generic(prefix, session_type, notes, feature_rows, hr_rows)


def _output_filename(prefix, session_type):
    return f"20_session_{prefix}.md"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sessions = find_sessions(INPUT_DIR)

    if not sessions:
        print(f"No YAML sidecar files found in {INPUT_DIR}")
        return

    for prefix, yaml_path in sessions:
        with open(yaml_path) as f:
            meta = yaml.safe_load(f) or {}
        session_type = meta.get("session_type", "unknown").strip()

        out_path = Path(OUTPUT_DIR) / _output_filename(prefix, session_type)
        if out_path.exists():
            print(f"  skip {prefix} (exists)")
            continue

        print(f"  process {prefix} ...")
        try:
            md = process_session(prefix, yaml_path, input_dir=INPUT_DIR)
            with open(out_path, "w") as f:
                f.write(md + "\n")
            print(f"     -> {out_path}")
        except Exception as e:
            print(f"  error processing {prefix}: {e}")
            raise

    print(f"\nDone. Per-session files in {OUTPUT_DIR}/")
    print(f"Next: run weekly_summary.py to regenerate the rollup.")


if __name__ == "__main__":
    main()
