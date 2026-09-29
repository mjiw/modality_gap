"""
MixBench2026 — K Sensitivity Ablation v2 (Z-Score + SVD)
==================================================================
v1 (mixbench_k_sensitivity.py) 확장판:
  - k 값을 {1, 2, 4, 8, 16, 32, 64, 128} 로 확장
  - 각 데이터셋마다 별도 plot 생성 (한 plot에 1 데이터셋)
  - 각 plot에 baseline 수평선 추가: CLIP (raw), GR-CLIP
  - 방법별 색상 통일 (데이터셋 무관)

색상 (plot_alpha_sensitivity.py 와 일관):
  CLIP         : 회색 (#888888), 점선
  GR-CLIP              : 파랑 (#1976D2), 점선
  Z+Subspace Top-K(★)  : 빨강 (#D32F2F), 실선 + 다이아몬드 마커

출력:
  MixBench_results/k_sensitivity_v2/
    k_sensitivity_v2.json       — 각 (subset, k, α) raw 결과 + baseline
    k_sensitivity_v2_best.json  — 각 (subset, k) best α + baseline best
    fig_{subset}_k_sensitivity.png — 데이터셋별 figure (4개)
    fig_combined_grid.png       — 4개 데이터셋 통합 grid

실행:
  conda activate jiwoo
  python mixbench_k_sensitivity_v2.py
  python mixbench_k_sensitivity_v2.py --subsets MSCOCO,OVEN
  python mixbench_k_sensitivity_v2.py --ks 1,2,4,8,16,32,64,128
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_MIXBENCH_DIR = Path(__file__).resolve().parent
if str(_MIXBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_MIXBENCH_DIR))

from data_loader import (  # noqa: E402
    load_mixbench,
    evaluate_ranking,
    build_emb_corpus,
    process_emb_query,
    ALPHAS,
)
from mixbench_full_comparison_ext import (  # noqa: E402
    _build_from_precorrected,
    _l2norm_np,
    _compute_topk_directions,
    _project_out,
)


# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
KS_SWEEP_V2 = [1, 2, 4, 8, 16, 32, 64, 128]   # v1 [1,2,4,8,16] 에 32, 64, 128 추가

# 방법별 일관 색상 (plot_alpha_sensitivity.py 와 동일한 색/라벨)
METHOD_STYLES = {
    "clip":     {"label": "CLIP",               "color": "#888888",
                 "linestyle": "--", "marker": None},
    "gr_clip":  {"label": "GR-CLIP",                    "color": "#1976D2",
                 "linestyle": "--", "marker": None},
    "proposed": {"label": "Z + Subspace Top-K (Ours)",  "color": "#D32F2F",
                 "linestyle": "-",  "marker": "D"},
}


# ─────────────────────────────────────────────
# Baseline 계산 (Raw CLIP, GR-CLIP)
# ─────────────────────────────────────────────
def disp_name(name):
    """데이터셋 내부 식별자 → 표시용 이름.
    내부 키(캐시 폴더명 cache_Google_WIT_L14/, JSON 키)는 그대로 두고
    figure 라벨에서만 하이픈 표기를 쓴다."""
    return {"Google_WIT": "Google-WIT"}.get(name, name)


def compute_baseline_best(text_emb, image_emb, query_emb, modes, q_gt,
                           alphas, correction,
                           text_mean=None, image_mean=None,
                           text_std=None, image_std=None,
                           query_mean=None, query_std=None):
    """단일 baseline (raw 또는 gr_clip) 에 대해 α sweep → best 결과 반환."""
    best_mrr = -1
    best_alpha = None
    best_res = None
    for alpha in alphas:
        if correction == "raw":
            corpus = build_emb_corpus(text_emb, image_emb, modes, alpha)
            q = process_emb_query(query_emb)
        elif correction == "gr_clip":
            corpus = build_emb_corpus(text_emb, image_emb, modes, alpha,
                                      "gr_clip", text_mean, image_mean)
            q = process_emb_query(query_emb, "gr_clip", query_mean)
        else:
            raise ValueError(f"unknown correction: {correction}")

        scores = q @ corpus.T
        res = evaluate_ranking(scores, q_gt, modes=modes)
        if res["MRR"] > best_mrr:
            best_mrr = res["MRR"]
            best_alpha = alpha
            best_res = res

    return {
        "alpha": float(best_alpha),
        "MRR": best_res["MRR"],
        "R@1": best_res["R@1"],
        "R@5": best_res.get("R@5", 0.0),
        "R@10": best_res["R@10"],
        "R@20": best_res.get("R@20", 0.0),
        "NDCG@5": best_res.get("NDCG@5", 0.0),
        "NDCG@10": best_res["NDCG@10"],
        "NDCG@20": best_res.get("NDCG@20", 0.0),
    }


# ─────────────────────────────────────────────
# K Sensitivity (v2)
# ─────────────────────────────────────────────
def run_k_sensitivity_v2(subset, alphas=None, ks=KS_SWEEP_V2):
    """단일 데이터셋에서 baseline 2종 + Z-Score+SVD k sweep."""
    if alphas is None:
        alphas = ALPHAS

    text_emb, image_emb, query_emb, modes, q_gt = load_mixbench(subset)

    text_mask = np.array([m in ("text", "both") for m in modes])
    image_mask = np.array([m in ("image", "both") for m in modes])
    text_mean = text_emb[text_mask].mean(axis=0, keepdims=True)
    image_mean = image_emb[image_mask].mean(axis=0, keepdims=True)
    text_std = text_emb[text_mask].std(axis=0, keepdims=True) + 1e-8
    image_std = image_emb[image_mask].std(axis=0, keepdims=True) + 1e-8
    query_mean = query_emb.mean(axis=0, keepdims=True)
    query_std = query_emb.std(axis=0, keepdims=True) + 1e-8

    # ── Baselines ──
    print(f"  [baseline] CLIP (Raw) α sweep...")
    clip_best = compute_baseline_best(
        text_emb, image_emb, query_emb, modes, q_gt, alphas, "raw"
    )
    print(f"    best α={clip_best['alpha']:.1f}  MRR={clip_best['MRR']:.4f}  "
          f"NDCG@10={clip_best['NDCG@10']:.4f}")

    print(f"  [baseline] GR-CLIP α sweep...")
    gr_best = compute_baseline_best(
        text_emb, image_emb, query_emb, modes, q_gt, alphas, "gr_clip",
        text_mean=text_mean, image_mean=image_mean, query_mean=query_mean
    )
    print(f"    best α={gr_best['alpha']:.1f}  MRR={gr_best['MRR']:.4f}  "
          f"NDCG@10={gr_best['NDCG@10']:.4f}")

    # ── Z-score + L2 정규화 (proposed) ──
    t_z = _l2norm_np((text_emb - text_mean) / text_std).astype("float32")
    i_z = _l2norm_np((image_emb - image_mean) / image_std).astype("float32")
    q_z = _l2norm_np((query_emb - query_mean) / query_std).astype("float32")

    dirs_max = _compute_topk_directions(t_z, i_z, modes, max(ks))
    if dirs_max is None:
        print(f"  [{subset}] SVD 추정 불가 (both 쌍 부족) — proposed skip")
        return {"clip": clip_best, "gr_clip": gr_best,
                "best_per_k": {}, "raw": {}}

    # ── Z+Subspace Top-K (k sweep) ──
    raw_results = {}
    best_per_k = {}
    for kk in ks:
        if kk > dirs_max.shape[0]:
            print(f"  k={kk:>3d} skip (SVD가 {dirs_max.shape[0]}개 방향만 추출 가능)")
            continue
        dirs = dirs_max[:kk]
        t_p = _l2norm_np(_project_out(t_z, dirs)).astype("float32")
        i_p = _l2norm_np(_project_out(i_z, dirs)).astype("float32")
        q_p = _l2norm_np(_project_out(q_z, dirs)).astype("float32")

        k_res = {}
        for alpha in alphas:
            corpus = _build_from_precorrected(t_p, i_p, modes, alpha)
            scores = q_p @ corpus.T
            res = evaluate_ranking(scores, q_gt, modes=modes)
            k_res[alpha] = res
            raw_results[f"k{kk}_a{alpha:.1f}"] = res

        best_alpha = max(k_res, key=lambda a: k_res[a]["MRR"])
        best_res = k_res[best_alpha]
        best_per_k[kk] = {
            "alpha": float(best_alpha),
            "MRR": best_res["MRR"],
            "R@1": best_res["R@1"],
            "R@5": best_res.get("R@5", 0.0),
            "R@10": best_res["R@10"],
            "R@20": best_res.get("R@20", 0.0),
            "NDCG@5": best_res.get("NDCG@5", 0.0),
            "NDCG@10": best_res["NDCG@10"],
            "NDCG@20": best_res.get("NDCG@20", 0.0),
        }
        print(f"  k={kk:>3d}  best α={best_alpha:.1f}  "
              f"MRR={best_res['MRR']:.4f}  NDCG@10={best_res['NDCG@10']:.4f}")

    return {
        "clip": clip_best,
        "gr_clip": gr_best,
        "best_per_k": best_per_k,
        "raw": raw_results,
    }


# ─────────────────────────────────────────────
# 출력 표
# ─────────────────────────────────────────────
def print_k_sensitivity_table_v2(all_results, ks):
    """전체 데이터셋 × k 표 + baseline."""
    subsets = [s for s, r in all_results.items() if r]
    if not subsets:
        return

    for metric in ["NDCG@10", "R@1"]:
        print(f"\n{'=' * 110}")
        print(f"  K Sensitivity v2 — {metric} (best α per k)")
        print(f"{'=' * 110}")

        header = f"  {'Dataset':<14s}  {'CLIP':>8s}  {'GR-CLIP':>8s}"
        for kk in ks:
            header += f"  {'k=' + str(kk):>8s}"
        header += f"  {'best k':>8s}"
        print(header)
        print("  " + "-" * len(header))

        for subset in subsets:
            r = all_results[subset]
            row = f"  {subset:<14s}"
            row += f"  {r['clip'][metric]:>8.4f}"
            row += f"  {r['gr_clip'][metric]:>8.4f}"
            bpk = r["best_per_k"]
            vals_k = []
            for kk in ks:
                if kk in bpk:
                    row += f"  {bpk[kk][metric]:>8.4f}"
                    vals_k.append((kk, bpk[kk][metric]))
                else:
                    row += f"  {'—':>8s}"
            if vals_k:
                best_k, _ = max(vals_k, key=lambda x: x[1])
                row += f"  {'k=' + str(best_k):>8s}"
            else:
                row += f"  {'—':>8s}"
            print(row)
        print("  " + "-" * len(header))


# ─────────────────────────────────────────────
# Plot — 데이터셋별 별도
# ─────────────────────────────────────────────
def _plot_one_panel(ax, subset_result, ks, metric):
    """단일 subplot — 한 metric에 대해 baseline 수평선 + proposed 곡선."""
    bpk = subset_result["best_per_k"]
    ks_present = [k for k in ks if k in bpk]
    values = [bpk[k][metric] for k in ks_present]

    # CLIP 수평선
    clip_v = subset_result["clip"][metric]
    s = METHOD_STYLES["clip"]
    ax.axhline(clip_v, color=s["color"], linestyle=s["linestyle"],
               linewidth=2, label=s["label"])

    # GR-CLIP 수평선
    gr_v = subset_result["gr_clip"][metric]
    s = METHOD_STYLES["gr_clip"]
    ax.axhline(gr_v, color=s["color"], linestyle=s["linestyle"],
               linewidth=2, label=s["label"])

    # Proposed 곡선
    s = METHOD_STYLES["proposed"]
    if ks_present:
        ax.plot(ks_present, values, color=s["color"], linestyle=s["linestyle"],
                marker=s["marker"], markersize=8, linewidth=2, label=s["label"])
        # peak에 별표
        peak_idx = int(np.argmax(values))
        ax.scatter([ks_present[peak_idx]], [values[peak_idx]],
                   color=s["color"], s=160, marker="*",
                   edgecolors="black", linewidth=0.8, zorder=5)

    ax.set_ylabel(metric, fontsize=11)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10, framealpha=0.95)


def plot_per_dataset(all_results, out_dir, ks):
    """각 데이터셋마다 figure 1개 (좌: NDCG@10, 우: R@1)."""
    for subset, r in all_results.items():
        if not r or not r.get("best_per_k"):
            continue
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        _plot_one_panel(axes[0], r, ks, "NDCG@10")
        _plot_one_panel(axes[1], r, ks, "R@1")
        fig.suptitle(subset, fontsize=13)
        fig.tight_layout()
        out_path = out_dir / f"fig_{subset}_k_sensitivity.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  saved: {out_path.name}")


def plot_combined_grid(all_results, out_dir, ks):
    """2 metric × N 데이터셋 grid (2 rows × N cols).
    Row 1: 모든 데이터셋의 NDCG@10
    Row 2: 모든 데이터셋의 R@1
    """
    subsets = [s for s, r in all_results.items() if r and r.get("best_per_k")]
    if not subsets:
        return

    n_cols = len(subsets)
    fig, axes = plt.subplots(2, n_cols, figsize=(5.2 * n_cols, 9))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for j, subset in enumerate(subsets):
        r = all_results[subset]
        _plot_one_panel(axes[0, j], r, ks, "NDCG@10")
        _plot_one_panel(axes[1, j], r, ks, "R@1")
        axes[0, j].set_title(disp_name(subset), fontsize=12)  # 최상단 행에만 데이터셋명

    fig.tight_layout()
    out_path = out_dir / "fig_combined_grid.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_path.name}")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", default="MSCOCO,VisualNews,Google_WIT,OVEN",
                    help="comma-separated subset names")
    ap.add_argument("--ks", default=",".join(str(k) for k in KS_SWEEP_V2),
                    help=f"comma-separated k values (default: {KS_SWEEP_V2})")
    args = ap.parse_args()

    subsets = [s.strip() for s in args.subsets.split(",")]
    ks = [int(k.strip()) for k in args.ks.split(",")]

    out_dir = _MIXBENCH_DIR.parent / "MixBench_results" / "k_sensitivity_v2"
    out_dir.mkdir(exist_ok=True, parents=True)

    print(f"K Sensitivity v2 | subsets={subsets} | ks={ks}")
    print(f"  출력: {out_dir}\n")

    all_results = {}
    for subset in subsets:
        print(f"\n{'#' * 70}")
        print(f"  {subset}")
        print(f"{'#' * 70}")
        result = run_k_sensitivity_v2(subset, ks=ks)
        all_results[subset] = result

    # 표
    print_k_sensitivity_table_v2(all_results, ks)

    # Plot
    print(f"\n[plotting per dataset]")
    plot_per_dataset(all_results, out_dir, ks)

    print(f"\n[plotting combined grid]")
    plot_combined_grid(all_results, out_dir, ks)

    # 저장
    save_raw = {}
    save_best = {}
    for subset, r in all_results.items():
        if r is None:
            continue
        save_raw[subset] = r.get("raw", {})
        save_best[subset] = {
            "clip": r["clip"],
            "gr_clip": r["gr_clip"],
            "z_subK_best_per_k": r["best_per_k"],
        }
    with open(out_dir / "k_sensitivity_v2.json", "w") as f:
        json.dump(save_raw, f, indent=2)
    with open(out_dir / "k_sensitivity_v2_best.json", "w") as f:
        json.dump(save_best, f, indent=2)
    print(f"\n  저장: {out_dir / 'k_sensitivity_v2.json'}")
    print(f"  저장: {out_dir / 'k_sensitivity_v2_best.json'}")


if __name__ == "__main__":
    main()
