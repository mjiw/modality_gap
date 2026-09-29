"""
fig_pca_vs_svd_topk_removal_mixbench.py
==========================================================================
PCA top-k 부분공간 제거  vs  SVD-of-diff top-k 부분공간 제거 (정량 비교)
--------------------------------------------------------------------------
"PCA on stacked embeddings 의 top-k 방향" 과 "SVD on paired modality
difference 의 top-k 방향" 이 시각적으로 비슷해 보여도, **retrieval
성능 (best α 기준 NDCG@10) 으로는 어느 쪽이 더 우월한가** 를 측정한다.

비교 대상:
  - **No projection (k=0)**           : baseline (Z-score만 적용)
  - **PCA top-k removal**             : V_pca = SVD(stacked z-scored emb)
                                         의 top-k right singular vectors
                                         를 임베딩에서 제거
  - **SVD-of-diff top-k removal (Ours)** : V_svd = SVD(paired diff) 의
                                         top-k right singular vectors 제거

가설:
  PCA 와 SVD-of-diff 의 첫 번째 방향은 비슷할 수 있지만, k 가 커질수록
  부분공간 자체가 달라져 retrieval 성능에서 차이가 벌어진다 → 우리 방법이
  단순한 PCA 가 아님을 정량적으로 입증.

출력:
  MixBench_results/combined_4dataset/fig_pca_vs_svd_topk_removal_2x2.png
  MixBench_results/combined_4dataset/pca_vs_svd_topk_removal.json
  (per-subset 단일 패널은 MixBench_results/{subset}/figures_3way_mixbench/)

실행:
  conda activate jiwoo
  # 4 데이터셋 모두 + 통합 2×2 figure (기본)
  python fig_pca_vs_svd_topk_removal_mixbench.py
  # 특정 subset 만
  python fig_pca_vs_svd_topk_removal_mixbench.py --subsets MSCOCO,OVEN
  # k 값 변경
  python fig_pca_vs_svd_topk_removal_mixbench.py --ks 1,2,4,8,16
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data_loader import load_mixbench, ALPHAS  # noqa: E402
from mixbench_full_comparison_ext import (  # noqa: E402
    _l2norm_np, _project_out,
)
from k_sensitivity_svd_spectrum import (  # noqa: E402
    compute_zscored, compute_full_svd,
    evaluate_projection_best_alpha, K_FIXED,
)


# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
DATASETS_ORDER = ["MSCOCO", "VisualNews", "Google_WIT", "OVEN"]
KS_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128]

# 색상
COL_SVD  = "#D32F2F"   # red   : SVD-of-diff (Ours)
COL_PCA  = "#388E3C"   # green : PCA on stacked embeddings
COL_MARK = "#1976D2"   # blue  : k_fixed marker
COL_BASE = "#222222"   # black : baseline


# ─────────────────────────────────────────────
# PCA on stacked z-scored embeddings
# ─────────────────────────────────────────────
def disp_name(name):
    """데이터셋 내부 식별자 → 표시용 이름.
    내부 키(캐시 폴더명 cache_Google_WIT_L14/, JSON 키)는 그대로 두고
    figure 라벨에서만 하이픈 표기를 쓴다."""
    return {"Google_WIT": "Google-WIT"}.get(name, name)


def compute_pca_stacked(t_z, i_z):
    """X = [t_z; i_z] 의 centered SVD → top-rank right singular vectors.
    return: Vt_pca (rank × D)
    """
    X = np.vstack([t_z, i_z]).astype("float32")
    X_c = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    return Vt


# ─────────────────────────────────────────────
# 단일 subset 분석
# ─────────────────────────────────────────────
def analyze_one(subset, ks):
    """return: {'baseline': float, 'svd': {k: ndcg}, 'pca': {k: ndcg},
                'cos_pc1_v1': float}
    """
    print(f"\n{'#'*70}")
    print(f"  [{subset}]")
    print(f"{'#'*70}")

    text_emb, image_emb, query_emb, modes, q_gt = load_mixbench(subset, "L14")
    t_z, i_z, q_z = compute_zscored(text_emb, image_emb, query_emb, modes)

    # SVD on paired diff
    sigmas_svd, Vt_svd = compute_full_svd(t_z, i_z, modes)
    if sigmas_svd is None:
        print(f"  Skip {subset} (no 'both' pairs).")
        return None

    # PCA on stacked z-scored embeddings
    Vt_pca = compute_pca_stacked(t_z, i_z)

    D = Vt_svd.shape[0]
    rank_pca = Vt_pca.shape[0]
    print(f"  rank(SVD-of-diff)={D}, rank(PCA-stacked)={rank_pca}")

    # 첫 방향 cosine similarity (PCA PC1 vs SVD v1)
    cos_pc1_v1 = float(abs(Vt_pca[0] @ Vt_svd[0]))
    print(f"  |cos(PC1, v1)| = {cos_pc1_v1:.4f}  "
          f"(1.0 에 가까울수록 같은 방향)")

    # baseline (no projection)
    base_res = evaluate_projection_best_alpha(
        t_z, i_z, q_z, modes, q_gt,
        np.zeros((0, t_z.shape[1]), dtype="float32"), ALPHAS,
    )
    baseline = base_res["NDCG@10"]
    print(f"  baseline (no projection): NDCG@10={baseline:.4f}")

    svd_at, pca_at = {}, {}
    for k in ks:
        if k > min(D, rank_pca):
            print(f"  k={k:>3d} skip "
                  f"(rank limit: SVD={D}, PCA={rank_pca})")
            continue
        # SVD-of-diff top-k
        svd_res = evaluate_projection_best_alpha(
            t_z, i_z, q_z, modes, q_gt, Vt_svd[:k], ALPHAS,
        )
        # PCA top-k
        pca_res = evaluate_projection_best_alpha(
            t_z, i_z, q_z, modes, q_gt, Vt_pca[:k], ALPHAS,
        )
        svd_at[k] = svd_res["NDCG@10"]
        pca_at[k] = pca_res["NDCG@10"]
        print(f"  k={k:>3d}: SVD-of-diff={svd_res['NDCG@10']:.4f}  "
              f"PCA={pca_res['NDCG@10']:.4f}  "
              f"Δ={svd_res['NDCG@10']-pca_res['NDCG@10']:+.4f}")

    return {
        "baseline": baseline,
        "svd": svd_at,
        "pca": pca_at,
        "cos_pc1_v1": cos_pc1_v1,
    }


# ─────────────────────────────────────────────
# Per-subset 단일 패널 plot
# ─────────────────────────────────────────────
def _plot_one_panel(ax, result, subset, ks, k_fixed=K_FIXED,
                    show_xlabel=True, show_ylabel=True, show_legend=True):
    baseline = result["baseline"]
    ks_plot = [k for k in ks if k in result["svd"]]
    svd_vals = [result["svd"][k] for k in ks_plot]
    pca_vals = [result["pca"][k] for k in ks_plot]

    # baseline
    ax.axhline(baseline, color=COL_BASE, linestyle=":",
               linewidth=2.2, alpha=1.0,
               label="No projection (k=0)")

    # PCA top-k removal
    ax.plot(ks_plot, pca_vals, color=COL_PCA, linewidth=2,
            marker="s", markersize=7,
            label="PCA top-k removal", zorder=3)

    # SVD-of-diff top-k removal (Ours)
    ax.plot(ks_plot, svd_vals, color=COL_SVD, linewidth=2,
            marker="D", markersize=7,
            label="SVD-of-diff top-k removal (Ours)", zorder=3)

    # k_fixed vertical line
    ax.axvline(k_fixed, color=COL_MARK, linestyle="--",
               linewidth=1.4, alpha=0.7,
               label=f"k={k_fixed} (selected)")
    if k_fixed in result["svd"]:
        ax.scatter([k_fixed], [result["svd"][k_fixed]],
                   color=COL_MARK, s=120, marker="*",
                   edgecolors="black", linewidth=0.7, zorder=5)

    ax.set_xscale("log", base=2)
    if show_xlabel:
        ax.set_xlabel("k (# directions projected out)", fontsize=15)
    if show_ylabel:
        ax.set_ylabel("NDCG@10 (best α)", fontsize=15)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_title(f"{disp_name(subset)}   |cos(PC1, v1)|={result['cos_pc1_v1']:.3f}",
                 fontsize=12)
    ax.grid(True, alpha=0.3, which="both")
    if show_legend:
        ax.legend(fontsize=9, loc="best", framealpha=0.95)


def save_per_subset_panels(all_results, ks):
    for subset, result in all_results.items():
        if result is None:
            continue
        out_dir = (_HERE.parent / "MixBench_results" / subset
                   / "figures_3way_mixbench")
        out_dir.mkdir(exist_ok=True, parents=True)
        out_path = out_dir / "fig_pca_vs_svd_topk_removal.png"
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        _plot_one_panel(ax, result, subset, ks)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  saved: {subset}/{out_path.name}")


def save_combined_2x2(all_results, ks):
    subsets = [s for s, r in all_results.items() if r is not None]
    if len(subsets) < 2:
        return
    out_dir = _HERE.parent / "MixBench_results" / "combined_4dataset"
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / "fig_pca_vs_svd_topk_removal_2x2.png"

    n = len(subsets)
    n_cols = 2 if n >= 2 else 1
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6.5 * n_cols, 5.2 * n_rows))
    axes_flat = (np.array(axes).flatten().tolist()
                 if n > 1 else [axes])

    for idx, (ax, subset) in enumerate(zip(axes_flat, subsets)):
        # 가독성: 왼쪽 열에만 ylabel, 아래 행에만 xlabel, 첫 패널에만 legend
        col = idx % n_cols
        row = idx // n_cols
        show_xlabel = (row == n_rows - 1)
        show_ylabel = (col == 0)
        show_legend = (idx == 0)
        _plot_one_panel(ax, all_results[subset], subset, ks,
                        show_xlabel=show_xlabel,
                        show_ylabel=show_ylabel,
                        show_legend=show_legend)

    for ax in axes_flat[len(subsets):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved (combined): {out_path.name}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", default=",".join(DATASETS_ORDER),
                    help="comma-separated subset 이름")
    ap.add_argument("--ks", default=",".join(str(k) for k in KS_SWEEP),
                    help="comma-separated k 값")
    args = ap.parse_args()

    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    ks = sorted(set(int(k.strip()) for k in args.ks.split(",")))

    t0 = time.time()
    print(f"PCA top-k vs SVD-of-diff top-k removal")
    print(f"  subsets: {subsets}")
    print(f"  ks: {ks}")

    all_results = {}
    for subset in subsets:
        all_results[subset] = analyze_one(subset, ks)

    # 저장
    out_dir = _HERE.parent / "MixBench_results" / "combined_4dataset"
    out_dir.mkdir(exist_ok=True, parents=True)
    out_json = out_dir / "pca_vs_svd_topk_removal.json"

    save = {}
    for subset, r in all_results.items():
        if r is None:
            continue
        save[subset] = {
            "baseline_ndcg10": r["baseline"],
            "svd_ndcg10": r["svd"],
            "pca_ndcg10": r["pca"],
            "cos_pc1_v1": r["cos_pc1_v1"],
        }
    with open(out_json, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\n  raw 결과 저장: {out_json}")

    # Per-subset 단일 패널
    print(f"\n[per-subset panels]")
    save_per_subset_panels(all_results, ks)

    # Combined 2x2
    print(f"\n[combined 2x2]")
    save_combined_2x2(all_results, ks)

    print(f"\n총 시간: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
