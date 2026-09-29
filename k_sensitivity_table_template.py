"""
k_sensitivity_table_template.py
====================================================================
mixbench_k_sensitivity_v2.py 가 이미 생성한 JSON 캐시
(MixBench_results/k_sensitivity_v2/k_sensitivity_v2_best.json)
를 읽어 **표 형식만 다양하게 출력**.

계산 로직 없음 — 캐시된 수치를 layout 만 바꾸어 출력.
숫자가 바뀌면 mixbench_k_sensitivity_v2.py 를 다시 실행하여 캐시
업데이트 후 본 스크립트를 재실행하면 됨.

출력 (NDCG@10 / R@1 두 metric 각각):

  [표 A] 옵션 A — Wide layout
         행 = dataset, 열 = (CLIP, GR, k=1..k=128, elbow 분석 4컬럼, Δ vs CLIP)
         → k=16 선택 정당화 + method 임팩트 한 표에 압축

  [표 B] Inverted layout — CLIP raw = k=0 baseline
         행 = k (k=0 baseline 포함), 열 = (dataset 4 + Avg)
         각 셀 = 절대값 (+Δ% vs CLIP raw)
         → k별 개선 추세를 dataset 간 비교

  [표 C] Elbow 요약 — Avg 기준
         행 = 핵심 k 값 (k=0, k=1, k=16, k=128) + Comment
         → 표 한 컷으로 elbow at k=16 정량 입증

실행:
  conda activate jiwoo
  python k_sensitivity_table_template.py                       # NDCG@10 + R@1
  python k_sensitivity_table_template.py --metric NDCG@10      # 단일 metric
  python k_sensitivity_table_template.py --json <경로>          # 다른 JSON 입력
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Windows cp949 인코딩 환경에서도 한글/유니코드 문자(─, ★, ↑) 출력 가능하도록 강제 UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # Python 3.7+
except Exception:
    pass

_HERE = Path(__file__).resolve().parent
DEFAULT_JSON = _HERE.parent / "MixBench_results" / "k_sensitivity_v2" / "k_sensitivity_v2_best.json"

# ─────────────────────────────────────────────
# 설정 (mixbench_k_sensitivity_v2 와 일관)
# ─────────────────────────────────────────────
DATASETS_ORDER = ["MSCOCO", "VisualNews", "Google_WIT", "OVEN"]
DEFAULT_KS = [1, 2, 4, 8, 16, 32, 64, 128]
K_FIXED = 16   # 우리가 선택한 k
METRICS = ["NDCG@10", "R@1"]


# ─────────────────────────────────────────────
# JSON 읽기 헬퍼
# ─────────────────────────────────────────────
def load_results(json_path: Path) -> dict:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def get_clip(results, subset, metric):
    return results[subset]["clip"].get(metric, 0.0)


def get_gr(results, subset, metric):
    return results[subset]["gr_clip"].get(metric, 0.0)


def get_subk(results, subset, k, metric):
    """k 가 정수 또는 문자열 키일 수 있음 (JSON 직렬화 차이)."""
    bpk = results[subset]["z_subK_best_per_k"]
    val = bpk.get(str(k)) or bpk.get(k)
    if val is None:
        return None
    return val.get(metric, 0.0)


def get_best_k(results, subset, metric, ks):
    best_k, best_v = None, -1.0
    for k in ks:
        v = get_subk(results, subset, k, metric)
        if v is not None and v > best_v:
            best_v, best_k = v, k
    return best_k, best_v


def avg(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def pct(num, den):
    return (num - den) / den * 100.0 if den > 0 else 0.0


# ─────────────────────────────────────────────
# [표 A] 옵션 A — Wide: dataset × k + 분석 컬럼
# ─────────────────────────────────────────────
def print_option_a(results, ks, metric, subsets):
    print(f"\n{'=' * 165}")
    print(f"  [표 A] {metric} — Wide layout (옵션 A): dataset × k + elbow 분석 + Δ vs CLIP")
    print(f"{'=' * 165}")

    label_w = 12
    cell_w = 6
    k_col_w = 6

    # 헤더
    header = " " * label_w
    header += f"{'CLIP':>{cell_w}}  {'GR':>{cell_w}}  "
    for k in ks:
        tag = f"k={k}" + ("★" if k == K_FIXED else "")
        header += f"{tag:>{k_col_w+1}}  "
    header += " │ "
    header += f"{'Δ(k=1→16)':>10}  {'Δ(16→128)':>10}  {'Best k':>7}  {'Gap k=16':>9}"
    header += " │ "
    header += f"{'Δ vs CLIP ★':>11}"
    print("  " + header)
    print("  " + "─" * (len(header)))

    # 집계용
    agg = {"clip": [], "gr": [], "ks": {k: [] for k in ks},
           "d_1_16": [], "d_16_128": [], "gap": [], "vs_clip": []}

    for subset in subsets:
        clip = get_clip(results, subset, metric)
        gr = get_gr(results, subset, metric)
        row = f"{subset:<{label_w}}"
        row += f"{clip:>{cell_w}.3f}  {gr:>{cell_w}.3f}  "

        v_at = {}
        for k in ks:
            v = get_subk(results, subset, k, metric)
            v_at[k] = v
            if v is None:
                row += f"{'—':>{k_col_w+1}}  "
            else:
                star = "★" if k == K_FIXED else " "
                row += f"{v:>{k_col_w}.3f}{star} "

        d_1_16 = pct(v_at[16], v_at[1]) if v_at.get(1) and v_at.get(16) else 0.0
        d_16_128 = pct(v_at[128], v_at[16]) if v_at.get(16) and v_at.get(128) else 0.0
        bk, bv = get_best_k(results, subset, metric, ks)
        gap = pct(bv, v_at[16]) if v_at.get(16) and bv else 0.0
        vs_clip = pct(v_at[16], clip) if v_at.get(16) and clip > 0 else 0.0

        row += " │ "
        row += f"{d_1_16:>+9.1f}%  {d_16_128:>+9.1f}%  k={bk:>4d}  {gap:>+7.2f}% "
        row += " │ "
        row += f"{vs_clip:>+10.1f}%"
        print("  " + row)

        agg["clip"].append(clip)
        agg["gr"].append(gr)
        for k in ks:
            agg["ks"][k].append(v_at[k])
        agg["d_1_16"].append(d_1_16)
        agg["d_16_128"].append(d_16_128)
        agg["gap"].append(gap)
        agg["vs_clip"].append(vs_clip)

    # Avg
    print("  " + "─" * (len(header)))
    row = f"{'Avg':<{label_w}}"
    row += f"{avg(agg['clip']):>{cell_w}.3f}  {avg(agg['gr']):>{cell_w}.3f}  "
    for k in ks:
        a = avg(agg["ks"][k])
        star = "★" if k == K_FIXED else " "
        row += f"{a:>{k_col_w}.3f}{star} "
    row += " │ "
    row += f"{avg(agg['d_1_16']):>+9.1f}% ★ {avg(agg['d_16_128']):>+8.1f}%  {'—':>6}  {avg(agg['gap']):>+7.2f}% "
    row += " │ "
    row += f"{avg(agg['vs_clip']):>+10.1f}% ★"
    print("  " + row)


# ─────────────────────────────────────────────
# [표 B] Inverted — k = row, dataset = col, 각 셀에 (Δ% vs CLIP)
# ─────────────────────────────────────────────
def print_inverted(results, ks, metric, subsets):
    cell_w = 16
    label_w = 12

    print(f"\n{'=' * 105}")
    print(f"  [표 B] {metric} 표 (CLIP raw = k=0 baseline)")
    print(f"{'=' * 105}")

    # 헤더
    header = " " * label_w
    for subset in subsets:
        header += f"  {subset:>{cell_w}}"
    header += f"  {'Avg':>{cell_w}}"
    print("  " + header)
    print("  " + "─" * len(header))

    # k=0 (CLIP raw) 행 — baseline
    clip_vals = [get_clip(results, s, metric) for s in subsets]
    avg_clip = avg(clip_vals)
    row = f"{'k=0 (CLIP)':<{label_w}}"
    for v in clip_vals:
        row += f"  {f'{v:.4f}':>{cell_w}}"
    row += f"  {f'{avg_clip:.4f}':>{cell_w}}"
    print("  " + row)
    print("  " + "─" * len(header))

    # k=1..k=128
    for k in ks:
        is_star = (k == K_FIXED)
        label = f"k={k}" + ("  ★" if is_star else "")
        row = f"{label:<{label_w}}"
        k_vals = []
        for i, subset in enumerate(subsets):
            v = get_subk(results, subset, k, metric)
            if v is None:
                row += f"  {'—':>{cell_w}}"
                continue
            d = pct(v, clip_vals[i])
            k_vals.append(v)
            cell = f"{v:.4f} ({d:+.1f}%)"
            row += f"  {cell:>{cell_w}}"
        if k_vals:
            a = avg(k_vals)
            d_avg = pct(a, avg_clip)
            star = " ★" if is_star else ""
            cell = f"{a:.4f} ({d_avg:+.1f}%){star}"
            row += f"  {cell:>{cell_w}}"
        print("  " + row)


# ─────────────────────────────────────────────
# [표 C] Elbow 요약 (Avg 기준)
# ─────────────────────────────────────────────
def print_elbow_summary(results, ks, metric, subsets):
    print(f"\n{'=' * 95}")
    print(f"  [표 C] Elbow 분석 요약 ({metric} Avg 기준)")
    print(f"{'=' * 95}")

    # k=0 (CLIP raw) 와 k별 평균
    clip_avg = avg([get_clip(results, s, metric) for s in subsets])
    k_avgs = {k: avg([get_subk(results, s, k, metric) for s in subsets]) for k in ks}

    header = f"{'':<28}  {f'Avg {metric}':>13}  {'Δ vs k=0':>13}  Comment"
    print("  " + header)
    print("  " + "─" * 93)

    print("  " + f"{'k=0  (CLIP raw)':<28}  {clip_avg:>13.4f}  {'—':>13}  baseline")

    pp_vs_clip = {}  # percentage points (relative to k=0)
    if 1 in k_avgs and k_avgs[1] is not None:
        d1 = pct(k_avgs[1], clip_avg)
        pp_vs_clip[1] = d1
        print("  " + f"{'k=1  (시작)':<28}  {k_avgs[1]:>13.4f}  {d1:>+12.1f}%  첫 SVD step")

    if K_FIXED in k_avgs and k_avgs[K_FIXED] is not None:
        d16 = pct(k_avgs[K_FIXED], clip_avg)
        pp_vs_clip[K_FIXED] = d16
        pp_gain_from_1 = d16 - pp_vs_clip.get(1, 0.0)
        print("  " + f"{'k=16 ★ (우리 선택)':<28}  {k_avgs[K_FIXED]:>13.4f}  {d16:>+12.1f}% ★  "
              f"elbow at {pp_gain_from_1:+.1f}%p gain from k=1")

    if 128 in k_avgs and k_avgs[128] is not None:
        d128 = pct(k_avgs[128], clip_avg)
        pp_vs_clip[128] = d128
        pp_drop_from_16 = d128 - pp_vs_clip.get(K_FIXED, 0.0)
        sign_text = "감소" if pp_drop_from_16 < 0 else "변동"
        print("  " + f"{'k=128 (far past elbow)':<28}  {k_avgs[128]:>13.4f}  {d128:>+12.1f}%  "
              f"k=16 대비 {pp_drop_from_16:+.1f}%p, {sign_text}")

    print("  " + "─" * 93)

    # 핵심 elbow 메시지 (percentage points)
    if 1 in pp_vs_clip and K_FIXED in pp_vs_clip:
        pp_gain = pp_vs_clip[K_FIXED] - pp_vs_clip[1]
        print(f"  Δ(k=1  → k=16):   {pp_gain:+.1f} percentage points  ← gain region")
    if K_FIXED in pp_vs_clip and 128 in pp_vs_clip:
        pp_drop = pp_vs_clip[128] - pp_vs_clip[K_FIXED]
        print(f"  Δ(k=16 → k=128):  {pp_drop:+.1f} percentage points  ← degradation region")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(DEFAULT_JSON),
                    help="k_sensitivity_v2_best.json 경로")
    ap.add_argument("--metric", default="all",
                    choices=["all", "NDCG@10", "R@1"],
                    help="단일 metric 만 출력 (default: all = NDCG@10 + R@1)")
    ap.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS),
                    help="comma-separated k 값")
    ap.add_argument("--subsets", default=",".join(DATASETS_ORDER),
                    help="comma-separated subset 이름")
    args = ap.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        print(f"[ERROR] JSON 파일이 없습니다: {json_path}")
        print(f"먼저 mixbench_k_sensitivity_v2.py 를 실행해 캐시를 만드세요.")
        sys.exit(1)

    results = load_results(json_path)
    ks = [int(k.strip()) for k in args.ks.split(",")]
    subsets = [s.strip() for s in args.subsets.split(",") if s.strip() in results]

    print(f"\nLoaded: {json_path}")
    print(f"Subsets: {subsets}")
    print(f"K values: {ks}")
    print(f"K fixed (우리 선택): k={K_FIXED}")

    metrics = METRICS if args.metric == "all" else [args.metric]

    for metric in metrics:
        print(f"\n\n{'#' * 90}")
        print(f"#  METRIC: {metric}")
        print(f"{'#' * 90}")
        print_option_a(results, ks, metric, subsets)
        print_inverted(results, ks, metric, subsets)
        print_elbow_summary(results, ks, metric, subsets)


if __name__ == "__main__":
    main()
