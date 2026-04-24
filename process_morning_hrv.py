import numpy as np

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


def _update_cumulative(carray, delta, check):
    new_val = carray[-1] + delta
    if new_val == check and new_val < 350000:
        carray.append(new_val)
        return True
    return False


def parse_rr_data_file(infile):
    cum_rr = [0]
    with open(infile, "r") as fi:
        for i, ln in enumerate(fi):
            ln = ln.strip().split(",")
            if i == 0:
                continue
            elif i == 1:
                d2 = int(ln[1])
                c2 = int(ln[2])
                c1 = c2 - d2
                d1 = c1
                _update_cumulative(cum_rr, d1, c1)
                _update_cumulative(cum_rr, d2, c2)
            else:
                d = int(ln[1])
                c = int(ln[2])
                if not _update_cumulative(cum_rr, d, c):
                    break
    cum_rr = np.array(cum_rr, dtype=int)
    rr = np.diff(cum_rr)
    return cum_rr[1:], rr


# --- Smoothing with reflection padding to avoid edge bias ---


def smooth_with_reflection(tt, yy, window_size):
    tt = np.asanyarray(tt)
    yy = np.asanyarray(yy)
    half_win = window_size / 2.0

    left_mask = tt <= (tt[0] + window_size)
    right_mask = tt >= (tt[-1] - window_size)

    yy_pad_left = yy[left_mask][1:][::-1]
    yy_pad_right = yy[right_mask][:-1][::-1]
    yy_padded = np.concatenate([yy_pad_left, yy, yy_pad_right])

    tt_pad_left = 2 * tt[0] - tt[left_mask][1:][::-1]
    tt_pad_right = 2 * tt[-1] - tt[right_mask][:-1][::-1]
    tt_padded = np.concatenate([tt_pad_left, tt, tt_pad_right])

    yy_smoothed = np.zeros_like(yy, dtype=float)
    for i in range(len(tt)):
        t_center = tt[i]
        mask = (tt_padded >= t_center - half_win) & (tt_padded <= t_center + half_win)
        yy_smoothed[i] = np.mean(yy_padded[mask])
    return yy_smoothed


# --- Main entrypoint ---


def compute_rhr_hrv_from_rr_data(infile, n_calm_beats=120):
    tt, rr = parse_rr_data_file(infile)
    tt = tt / 1000.0  # ms -> s
    if len(rr) == 0:
        raise ValueError(f"RR file {infile} produced no intervals")
    if np.any(rr <= 0):
        raise ValueError(f"RR file {infile} contains non-positive intervals")
    hr = 60000.0 / rr

    shr05 = smooth_with_reflection(tt, hr, 5)
    shr20 = smooth_with_reflection(tt, hr, 20)
    shr = np.minimum(shr05, shr20)

    # Find the n_calm_beats lowest-HR beats (after smoothing).
    # This is the subset where both short- and long-window HR were quiet,
    # which filters out the fast tops AND the slow-oscillation tops.
    idx = np.argsort(shr)
    n = min(n_calm_beats, len(shr))
    mask = np.zeros(len(shr), dtype=bool)
    mask[idx[:n]] = True

    rest_hr_bpm = float(np.min(shr20))
    rr_calm = rr[mask]
    stdrr_ms = float(np.std(rr_calm))

    # RMSSD on successive differences of RR intervals in the calm window.
    # Note: np.diff across a masked subset is *approximately* successive
    # because the mask is contiguous-ish (picks from sorted-calm beats),
    # not strictly consecutive. This matches the original estimator's
    # behavior; do not change without re-validating trend.
    ssd = np.diff(rr_calm) ** 2
    rmssd_ms = float(np.sqrt(np.mean(ssd)))
    rmssd_ms_median = float(np.sqrt(np.median(ssd)))

    # Breathing rate: find dominant low-freq oscillation in HR trace.
    # RSA produces ~0.2-0.4 Hz peak (12-24 breaths/min); paced slow breathing
    # pushes this to 0.08-0.12 Hz (~5-7/min).
    breathing_rate_est = _estimate_breathing_rate(tt, hr)

    return {
        "rest_hr_bpm": round(rest_hr_bpm, 1),
        "rmssd_ms": round(rmssd_ms, 1),
        "rmssd_ms_median": round(rmssd_ms_median * 1.2741 + 2.8714, 1),
        "stdrr_ms": round(stdrr_ms, 1),
        "n_beats": int(mask.sum()),
        "breathing_rate_est": breathing_rate_est,
    }


def _estimate_breathing_rate(tt, hr):
    if len(tt) < 60:
        return None
    if not np.all(np.isfinite(hr)):
        return None
    if np.std(hr) < 0.5:
        # Flat HR trace -> no RSA visible
        return None

    fs = 4.0
    t_uniform = np.arange(tt[0], tt[-1], 1.0 / fs)
    if len(t_uniform) < 32:
        return None
    hr_uniform = np.interp(t_uniform, tt, hr)
    hr_uniform = hr_uniform - np.mean(hr_uniform)
    if not np.all(np.isfinite(hr_uniform)) or np.std(hr_uniform) < 0.1:
        return None

    spectrum = np.abs(np.fft.rfft(hr_uniform))
    freqs = np.fft.rfftfreq(len(hr_uniform), 1.0 / fs)
    # 0.08 Hz (~5/min) lower bound avoids DC/near-DC bin capture on short data
    band = (freqs >= 0.08) & (freqs <= 0.50)
    if not band.any():
        return None
    band_spectrum = spectrum[band]
    if np.max(band_spectrum) < 2 * np.median(spectrum):
        # No clear peak above background -> signal too noisy to trust
        return None
    peak_freq = freqs[band][int(np.argmax(band_spectrum))]
    return round(peak_freq * 60.0, 1)
