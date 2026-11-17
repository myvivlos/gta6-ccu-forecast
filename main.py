#!/usr/bin/env python3
"""GTA VI launch-day Twitch CCU quantile forecaster."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
from typing import Any

DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
MIN_Q = 0.5
MAX_Q = 1_000_000_000.0
ISO_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?Z")
PC_WORD_RE = re.compile(r"\bpc\b")


def parse_iso_utc(ts: str) -> datetime:
    """Parse an ISO-8601 UTC timestamp into an aware datetime."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def norm_ppf(p: float) -> float:
    """Approximate standard normal inverse CDF (Acklam's rational approximation)."""
    if p <= 0.0:
        return -6.0
    if p >= 1.0:
        return 6.0
    if abs(p - 0.5) < 1e-12:
        return 0.0
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


def student_t_ppf(p: float, df: float) -> float:
    """Approximate Student-t inverse CDF via Cornish-Fisher expansion."""
    if abs(p - 0.5) < 1e-12:
        return 0.0
    z = norm_ppf(p)
    if df > 200.0:
        return z
    df = max(df, 2.0)
    z2 = z * z
    g1 = z * (z2 + 1.0) / 4.0
    g2 = z * (5.0 * z2 * z2 + 16.0 * z2 + 3.0) / 96.0
    g3 = z * (3.0 * z2 * z2 * z2 + 19.0 * z2 * z2 + 17.0 * z2 - 15.0) / 384.0
    g4 = z * (
        79.0 * z2**4 + 776.0 * z2**3 + 1482.0 * z2**2 - 1920.0 * z2 - 945.0
    ) / 92160.0
    return z + g1 / df + g2 / (df * df) + g3 / (df**3) + g4 / (df**4)


def weighted_quantile(
    val_weights: list[tuple[float, float]],
    tau: float,
    n_eff: float | None = None,
    weib_alpha: float = 0.0,
) -> float:
    """Linearly interpolated weighted empirical quantile using midpoint ranks."""
    vw = sorted((v, w) for v, w in val_weights if w > 0.0 and math.isfinite(v))
    if not vw:
        return 0.0
    if len(vw) == 1:
        return vw[0][0]
    tot = sum(w for _, w in vw)
    if tot <= 0.0:
        return vw[len(vw) // 2][0]

    mids: list[float] = []
    cum = 0.0
    for _, w in vw:
        mids.append((cum + 0.5 * w) / tot)
        cum += w

    if weib_alpha > 0.0 and n_eff is not None:
        adj = weib_alpha / (max(n_eff, 2.0) + 1.0)
        mids = [0.5 * adj + (1.0 - adj) * m for m in mids]

    if tau <= mids[0]:
        return vw[0][0]
    if tau >= mids[-1]:
        return vw[-1][0]
    for i in range(len(vw) - 1):
        if mids[i] <= tau <= mids[i + 1]:
            span = mids[i + 1] - mids[i]
            t = (tau - mids[i]) / span if span > 0.0 else 0.0
            return vw[i][0] * (1.0 - t) + vw[i + 1][0] * t
    return vw[-1][0]


def nearest_hour_value(
    hmap: dict[int, float], target_rh: int, offsets: tuple[int, ...]
) -> float | None:
    """Fetch hour value at target_rh or nearest offset."""
    if target_rh in hmap:
        return hmap[target_rh]
    for off in offsets:
        val = hmap.get(target_rh + off)
        if val is not None:
            return val
    return None


def build_hourly_map(
    rows: list[dict[str, Any]],
    day_start: datetime,
    time_key: str,
) -> dict[int, float]:
    """Aggregate viewer readings into integer hour offsets relative to day_start."""
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    ds_epoch = day_start.timestamp()
    for row in rows:
        ts = row.get(time_key)
        viewers = row.get("viewers")
        if ts is None or viewers is None:
            continue
        try:
            v = float(viewers)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v) or v < 0.0:
            continue
        dt_epoch = parse_iso_utc(ts).timestamp()
        rh = int(math.floor((dt_epoch - ds_epoch + 1e-6) / 3600.0))
        sums[rh] = sums.get(rh, 0.0) + v
        counts[rh] = counts.get(rh, 0) + 1
    return {rh: sums[rh] / counts[rh] for rh in sums}


def detect_target_is_console(notes: str) -> bool | None:
    """Infer from case notes whether the target release is console-only."""
    if not notes:
        return None
    head = notes.split("real unlock")[0].split("Source:")[0].lower()
    has_console = any(k in head for k in ("ps5", "ps4", "xbox", "switch", "console"))
    has_no_pc = ("console only" in head) or ("no pc" in head)
    has_pc = bool(PC_WORD_RE.search(head)) and not has_no_pc
    if has_console and not has_pc:
        return True
    if has_pc:
        return False
    return None


def parse_comparable(comp: dict[str, Any]) -> dict[str, Any] | None:
    """Extract release-day log-ratios and pre-release context for a comparable launch."""
    unlock = str(comp.get("unlock") or "")
    unlock_l = unlock.lower()
    if "not a launch" in unlock_l or "tier: target-history" in unlock_l:
        return None

    hourly = comp.get("hourly") or []
    ds_str = comp.get("day_start")
    if not ds_str or not hourly:
        return None
    ds = parse_iso_utc(ds_str)
    hmap = build_hourly_map(hourly, ds, "from")
    if not hmap:
        return None

    # Hour 1 is day_start (rh == 0); if missing/below 1, fall back to first whole hour >= unlock/day_start
    shift = 0
    v1 = hmap.get(0)
    if v1 is None or v1 < 1.0:
        cand_rhs: list[int] = []
        for match in ISO_TS_RE.finditer(unlock):
            u_dt = parse_iso_utc(match.group(0))
            u_rh = int(math.ceil((u_dt.timestamp() - ds.timestamp() - 1e-6) / 3600.0))
            if u_rh >= 0:
                cand_rhs.append(u_rh)
        cand_rhs.extend(sorted(rh for rh in hmap if rh >= 0 and hmap[rh] >= 1.0))
        chosen_rh = None
        for rh in cand_rhs:
            val = hmap.get(rh)
            if val is not None and val >= 1.0:
                chosen_rh = rh
                break
        if chosen_rh is None:
            return None
        shift = chosen_rh
        v1 = hmap[chosen_rh]

    if shift != 0:
        hmap = {rh - shift: val for rh, val in hmap.items()}

    v1 = max(float(v1), 1.0)
    early_vals = [val for rh, val in hmap.items() if rh < -24]
    early_max = max(early_vals) if early_vals else 0.0
    pre_vals = [hmap[rh] for rh in range(-13, 0) if rh in hmap]
    pre_max = max(pre_vals) if pre_vals else v1

    v_m1 = nearest_hour_value(hmap, -1, (-1, -2))
    v_m6 = nearest_hour_value(hmap, -6, (-1, 1, -2, 2))

    tier = "comp"
    if "early-access" in unlock_l:
        tier = "ea"
    elif "tier: weak" in unlock_l:
        tier = "weak"

    platforms = str(comp.get("platforms") or "").lower()
    is_console = not bool(PC_WORD_RE.search(platforms))

    ratios: dict[int, float] = {}
    for h in range(2, 25):
        vh = hmap.get(h - 1)
        if vh is not None and math.isfinite(vh):
            ratios[h] = math.log(max(vh, 1.0) / v1)

    if not ratios:
        return None

    return {
        "tier": tier,
        "is_console": is_console,
        "lv1": math.log(v1),
        "ea_l": math.log(max(early_max, 1.0) / v1),
        "pm_l": math.log(max(pre_max, 1.0) / v1),
        "m1": math.log(max(v_m1 if v_m1 is not None else v1, 1.0) / v1),
        "m6": math.log(max(v_m6 if v_m6 is not None else v1, 1.0) / v1),
        "ratios": ratios,
    }


def extract_target(case: dict[str, Any]) -> dict[str, Any]:
    """Extract target case hour-1 print and pre-release evidence."""
    ds = parse_iso_utc(case["day_start"])
    hourly_rows = case.get("history_hourly") or []
    if hourly_rows:
        hmap = build_hourly_map(hourly_rows, ds, "from")
    else:
        hmap = build_hourly_map(case.get("history") or [], ds, "at")

    known = case.get("known") or []
    if known and known[0].get("viewers") is not None:
        raw_p = float(known[0]["viewers"])
    else:
        raw_p = float(hmap.get(0, 1.0))
    if not math.isfinite(raw_p) or raw_p < 1.0:
        raw_p = 1.0
    p = raw_p

    early_vals = [val for rh, val in hmap.items() if rh < -24]
    has_early = len(early_vals) > 0
    early_max = max(early_vals) if has_early else 0.0

    pre_vals = [hmap[rh] for rh in range(-13, 0) if rh in hmap]
    has_pre = len(pre_vals) > 0
    pre_max = max(pre_vals) if has_pre else p

    v_m1 = nearest_hour_value(hmap, -1, (-1, -2))
    v_m6 = nearest_hour_value(hmap, -6, (-1, 1, -2, 2))

    return {
        "p": p,
        "lv1": math.log(p),
        "has_early": has_early,
        "ea_l": math.log(max(early_max, 1.0) / p),
        "has_pre": has_pre,
        "pm_l": math.log(max(pre_max, 1.0) / p),
        "has_m1": v_m1 is not None,
        "m1": math.log(max(v_m1 if v_m1 is not None else p, 1.0) / p),
        "has_m6": v_m6 is not None,
        "m6": math.log(max(v_m6 if v_m6 is not None else p, 1.0) / p),
        "is_console": detect_target_is_console(str(case.get("notes") or "")),
    }


def compute_comparable_weights(
    target: dict[str, Any], comps: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], float]]:
    """Compute similarity kernel weights for each comparable launch."""
    weighted: list[tuple[dict[str, Any], float]] = []
    for c in comps:
        d2 = ((c["lv1"] - target["lv1"]) / 2.5) ** 2
        if target["has_early"]:
            d2 += ((c["ea_l"] - target["ea_l"]) / 1.2) ** 2
        if target["has_pre"]:
            d2 += ((c["pm_l"] - target["pm_l"]) / 1.0) ** 2
        if target["has_m6"]:
            d2 += ((c["m6"] - target["m6"]) / 1.5) ** 2

        w = math.exp(-0.5 * d2)
        if c["tier"] == "weak":
            w *= 0.2
        if c["tier"] == "ea" and (not target["has_early"] or target["ea_l"] < -0.5):
            w *= 0.2
        if target["is_console"] is not None and c["is_console"] == target["is_console"]:
            w *= 2.5
        weighted.append((c, w))
    return weighted


def forecast_case(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Produce calibrated 23-hour quantile forecasts for one input case."""
    levels = [float(x) for x in (case.get("levels") or DEFAULT_LEVELS)]
    z_scores = [norm_ppf(tau) for tau in levels]
    n_q = len(levels)

    target = extract_target(case)
    p = target["p"]
    comps = [
        rec
        for raw_c in (case.get("comparables") or [])
        if (rec := parse_comparable(raw_c)) is not None
    ]

    if not comps:
        rows = []
        for h in range(2, 25):
            s = 0.15 * math.sqrt(h - 1)
            qs = [p * math.exp(z * s) for z in z_scores]
            rows.append({"hour": h, "q": finalize_quantiles(qs, p)})
        return rows

    weighted = compute_comparable_weights(target, comps)
    lam1 = 2.0
    lam2 = 1.0
    mix_gauss = 0.45
    smooth_h = 0.20
    tail_scale = 1.08
    weib_alpha = 0.75
    df_cap = 9.0
    thin_cnt = 8
    thin_neff = 4.5
    mid_idx = n_q // 2

    raw_q: dict[int, list[float]] = {}
    worst_r: dict[int, float] = {}
    count_map: dict[int, tuple[int, float]] = {}
    comp_med_r24: float | None = None

    for h in range(2, 25):
        all_y = [c["ratios"][h] for c in comps if h in c["ratios"]]
        pts = [
            (c, w, c["ratios"][h])
            for c, w in weighted
            if h in c["ratios"] and w > 1e-4
        ]
        if not pts:
            pts = [(c, 1.0, c["ratios"][h]) for c, _ in weighted if h in c["ratios"]]

        if not pts:
            count_map[h] = (0, 0.0)
            if h > 2 and (h - 1) in raw_q:
                raw_q[h] = list(raw_q[h - 1])
                worst_r[h] = worst_r[h - 1]
            else:
                s = 0.15 * math.sqrt(h - 1)
                raw_q[h] = [z * s for z in z_scores]
                worst_r[h] = -s * 2.0
            continue

        worst_r[h] = min(all_y)
        if h == 24 and all_y:
            s_all = sorted(all_y)
            comp_med_r24 = s_all[len(s_all) // 2]

        wtot = sum(w for _, w, _ in pts)
        mx1 = sum(w * c["lv1"] for c, w, _ in pts) / wtot
        my = sum(w * y for _, w, y in pts) / wtot

        if target["has_m1"] and len(pts) >= 4:
            mx2 = sum(w * c["m1"] for c, w, _ in pts) / wtot
            s11 = sum(w * (c["lv1"] - mx1) ** 2 for c, w, _ in pts) + lam1 * wtot
            s22 = sum(w * (c["m1"] - mx2) ** 2 for c, w, _ in pts) + lam2 * wtot
            s12 = sum(w * (c["lv1"] - mx1) * (c["m1"] - mx2) for c, w, _ in pts)
            sy1 = sum(w * (y - my) * (c["lv1"] - mx1) for c, w, y in pts)
            sy2 = sum(w * (y - my) * (c["m1"] - mx2) for c, w, y in pts)
            det = s11 * s22 - s12 * s12
            if det > 1e-12:
                b1 = (sy1 * s22 - sy2 * s12) / det
                b2 = (s11 * sy2 - s12 * sy1) / det
            else:
                b1 = b2 = 0.0
            dx1 = max(-1.5, min(1.5, target["lv1"] - mx1))
            dx2 = max(-0.8, min(0.8, target["m1"] - mx2))
            pred_mean = my + b1 * dx1 + b2 * dx2
            res_ws = [
                (y - (my + b1 * (c["lv1"] - mx1) + b2 * (c["m1"] - mx2)), w)
                for c, w, y in pts
            ]
        elif len(pts) >= 3:
            s11 = sum(w * (c["lv1"] - mx1) ** 2 for c, w, _ in pts) + lam1 * wtot
            sy1 = sum(w * (y - my) * (c["lv1"] - mx1) for c, w, y in pts)
            b1 = sy1 / s11 if s11 > 1e-12 else 0.0
            dx1 = max(-1.5, min(1.5, target["lv1"] - mx1))
            pred_mean = my + b1 * dx1
            res_ws = [(y - (my + b1 * (c["lv1"] - mx1)), w) for c, w, y in pts]
        else:
            pred_mean = my
            res_ws = [(y - my, w) for _, w, y in pts]

        rmean = sum(w * res for res, w in res_ws) / wtot
        rvar = sum(w * (res - rmean) ** 2 for res, w in res_ws) / wtot
        sum_w2 = sum(w * w for _, w in res_ws)
        n_eff = (wtot * wtot) / sum_w2 if sum_w2 > 0.0 else float(len(res_ws))
        count_map[h] = (len(pts), n_eff)
        rstd = math.sqrt(rvar * max(n_eff / max(n_eff - 1.5, 1.0), 1.0))
        if n_eff < 3.0:
            rstd = max(rstd, 0.15 * math.sqrt(h - 1))

        df = min(df_cap, max(n_eff - 1.5, 2.5))
        med_emp = weighted_quantile(res_ws, 0.5)
        row_r = [
            pred_mean
            + (1.0 - mix_gauss)
            * weighted_quantile(res_ws, tau, n_eff=n_eff, weib_alpha=weib_alpha)
            + mix_gauss * (med_emp + student_t_ppf(tau, df) * rstd)
            for tau in levels
        ]
        # Widen outer tails (5% / 95%) without widening inner 25% / 75% bands
        med_val = row_r[mid_idx]
        raw_q[h] = [
            row_r[k]
            if k == mid_idx or (0.10 < tau < 0.90)
            else med_val + tail_scale * (row_r[k] - med_val)
            for k, tau in enumerate(levels)
        ]

    # Temporal smoothing across adjacent hours + decay guard
    sm_map: dict[int, list[float]] = {}
    for h in range(2, 25):
        curr = raw_q[h]
        prev_q = raw_q[h - 1] if h > 2 else [0.0] * n_q
        next_q = raw_q[h + 1] if h < 24 else curr
        sm = [
            (1.0 - smooth_h) * curr[k] + 0.5 * smooth_h * (prev_q[k] + next_q[k])
            for k in range(n_q)
        ]
        floor_r = worst_r[h]
        sm = [max(r, floor_r) for r in sm]
        if h == 24 and comp_med_r24 is not None and comp_med_r24 < 0.0:
            if sm[mid_idx] < comp_med_r24:
                shift = comp_med_r24 - sm[mid_idx]
                sm = [r + shift for r in sm]
        sm_map[h] = sm

    # Monotonically widen every band with the horizon when comparable count is thin
    z_max = max(abs(z_scores[0]), abs(z_scores[-1]), 1.6448536)
    prev_lo = [0.0] * mid_idx
    prev_hi = [0.0] * (n_q - mid_idx - 1)
    for h in range(2, 25):
        sm = sm_map[h]
        med = sm[mid_idx]
        cnt, n_eff = count_map[h]
        if cnt < thin_cnt or n_eff < thin_neff:
            thin_gap = max(
                (thin_cnt - cnt) / max(thin_cnt, 1),
                (thin_neff - n_eff) / max(thin_neff, 1.0),
            )
            h_floor = 0.05 * thin_gap * math.sqrt(h - 1)
            for k in range(mid_idx):
                z_ratio = abs(z_scores[k]) / z_max
                spread = max(med - sm[k], prev_lo[k], h_floor * z_ratio)
                sm[k] = med - spread
                prev_lo[k] = spread
            for j, k in enumerate(range(mid_idx + 1, n_q)):
                z_ratio = abs(z_scores[k]) / z_max
                spread = max(sm[k] - med, prev_hi[j], h_floor * z_ratio)
                sm[k] = med + spread
                prev_hi[j] = spread
        else:
            for k in range(mid_idx):
                prev_lo[k] = max(prev_lo[k], med - sm[k])
            for j, k in enumerate(range(mid_idx + 1, n_q)):
                prev_hi[j] = max(prev_hi[j], sm[k] - med)

    # Exponentiation and defensive clamping
    out_rows: list[dict[str, Any]] = []
    for h in range(2, 25):
        qs = [p * math.exp(max(-20.0, min(20.0, r))) for r in sm_map[h]]
        out_rows.append({"hour": h, "q": finalize_quantiles(qs, p)})

    return out_rows


def finalize_quantiles(qs: list[float], p: float) -> list[float]:
    """Clamp to valid finite range, round, and enforce non-decreasing order."""
    fallback = max(p, 1.0)
    cleaned: list[float] = []
    for q in qs:
        if not math.isfinite(q) or q <= 0.0:
            q = fallback
        q = min(MAX_Q, max(MIN_Q, q))
        q = round(q, 2)
        if cleaned and q < cleaned[-1]:
            q = cleaned[-1]
        q = min(MAX_Q, max(MIN_Q, q))
        cleaned.append(q)
    return cleaned


def main() -> None:
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR") or "/input")
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    for in_path in sorted(input_dir.glob("*.in"), key=lambda path: path.name):
        with in_path.open("r", encoding="utf-8") as fh:
            case = json.load(fh)
        rows = forecast_case(case)
        payload = json.dumps({"hours": rows}, separators=(",", ":"))
        out_path = output_dir / f"{in_path.stem}.out"
        with out_path.open("w", encoding="utf-8") as fh:
            fh.write(payload)


if __name__ == "__main__":
    main()
