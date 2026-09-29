"""
fig_pca_vs_svd_diff_mixbench.py
==========================================================================
Figure: PCA on embeddings  vs  SVD on paired modality difference  (MixBench)
--------------------------------------------------------------------------
원본 fig_pca_vs_svd_diff.py 의 MixBench 데이터셋 버전.
원본은 WIT paired 100K 를 사용하지만, 본 파일은 메인 실험과 동일한
MixBench 데이터셋 (MSCOCO / VisualNews / Google_WIT / OVEN) 을 사용한다.

같은 z-scored 임베딩에 대해 두 좌표계를 비교:

  (a) PCA on stacked embeddings
        X = [text; image] 의 centered SVD → PC1, PC2 축
        ⇒ PC1 이 modality 방향이라는 보장이 없음

  (b) SVD on paired modality difference (Ours)
        D = image - text 의 centered SVD → v1, v2 축
        ⇒ v1 은 정의상 paired modality 차이의 주축
        ⇒ text 와 image 가 v1 축에서 체계적으로 분리됨

메시지:
   "같은 알고리즘(centered SVD)이라도, 적용 대상 행렬이 다르면
    좌표축의 의미가 완전히 달라진다."

paired ('both') 아이템 수:
  MSCOCO=328, VisualNews=327, Google_WIT=1473, OVEN=334

실행:
  conda activate jiwoo
  # 기본 (Google_WIT)
  python fig_pca_vs_svd_diff_mixbench.py
  # 다른 데이터셋
  python fig_pca_vs_svd_diff_mixbench.py --subset MSCOCO
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
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

from data_loader import SEED, load_mixbench  # noqa: E402
from mixbench_full_comparison_ext import _l2norm_np  # noqa: E402


# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
DEFAULT_SUBSET = "Google_WIT"
SUBSETS = ["MSCOCO", "VisualNews", "Google_WIT", "OVEN"]

COL_TEXT  = "#1565C0"
COL_IMAGE = "#C62828"


def _out_path(subset: str) -> Path:
    d = _HERE.parent / "MixBench_results" / subset / "figures_3way_mixbench"
    d.mkdir(exist_ok=True, parents=True)
    return d / "fig_pca_vs_svd_diff.png"


def _combined_dir() -> Path:
    d = _HERE.parent / "MixBench_results" / "combined_4dataset"
    d.mkdir(exist_ok=True, parents=True)
    return d


# ─────────────────────────────────────────────
# (a) PCA on stacked embeddings
# ─────────────────────────────────────────────
def project_pca_stacked(t_z, i_z):
    """text + image 를 stack 후 centered SVD → PC1, PC2 에 투영."""
    X = np.vstack([t_z, i_z]).astype("float32")
    X_c = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    V2 = Vt[:2]
    Y = X_c @ V2.T
    n_text = len(t_z)
    return Y[:n_text], Y[n_text:]


# ─────────────────────────────────────────────
# (b) SVD on paired modality difference
# ─────────────────────────────────────────────
def project_svd_paired_diff(t_z, i_z):
    """paired diff = image - text 에 centered SVD → v1, v2.
    원본 t_z, i_z 를 v1, v2 에 투영."""
    diffs = i_z - t_z
    diffs_c = diffs - diffs.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(diffs_c, full_matrices=False)
    V2 = Vt[:2]
    return t_z @ V2.T, i_z @ V2.T


# ─────────────────────────────────────────────
# Plot helper
# ─────────────────────────────────────────────
def scatter_panel(ax, text_2d, image_2d, *, xlabel, ylabel, title):
    ax.scatter(text_2d[:, 0], text_2d[:, 1],
               c=COL_TEXT, s=10, alpha=0.6,
               label=f"Text  (n={len(text_2d)})",
               edgecolors="none")
    ax.scatter(image_2d[:, 0], image_2d[:, 1],
               c=COL_IMAGE, s=10, alpha=0.6,
               label=f"Image (n={len(image_2d)})",
               edgecolors="none")
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(loc="best", fontsize=12, markerscale=2.0,
              framealpha=0.95)
    ax.grid(True, alpha=0.25)


# ─────────────────────────────────────────────
# 단일 subset 처리
# ─────────────────────────────────────────────
def run_one(subset):
    t0 = time.time()
    out_path = _out_path(subset)
    print(f"\n{'#'*70}")
    print(f"  [PCA vs SVD-of-diff] subset={subset}")
    print(f"  output: {out_path}")
    print(f"{'#'*70}")

    # 1) 임베딩 로드 (캐시 hit)
    text_emb, image_emb, _, modes, _ = load_mixbench(subset, "L14")
    print(f"  loaded: text={text_emb.shape}, image={image_emb.shape}")

    # 2) Z-score (modality 별)
    text_mask  = np.array([m in ("text", "both")  for m in modes])
    image_mask = np.array([m in ("image", "both") for m in modes])
    t_mean = text_emb [text_mask ].mean(axis=0, keepdims=True)
    i_mean = image_emb[image_mask].mean(axis=0, keepdims=True)
    t_std  = text_emb [text_mask ].std (axis=0, keepdims=True) + 1e-8
    i_std  = image_emb[image_mask].std (axis=0, keepdims=True) + 1e-8

    t_z = _l2norm_np((text_emb  - t_mean) / t_std).astype("float32")
    i_z = _l2norm_np((image_emb - i_mean) / i_std).astype("float32")

    # 3) paired 'both' 전부 사용 (MixBench 는 작아서 sub-sampling 불필요)
    paired_idx = np.where(np.array([m == "both" for m in modes]))[0]
    t_sub = t_z[paired_idx]
    i_sub = i_z[paired_idx]
    print(f"  paired: {len(paired_idx)} (all used for visualization)")

    # 4) 두 좌표계 계산
    print(f"  (a) PCA on stacked embeddings ...")
    text_pca,  image_pca  = project_pca_stacked(t_sub, i_sub)
    print(f"  (b) SVD on paired diff ...")
    text_svd,  image_svd  = project_svd_paired_diff(t_sub, i_sub)

    # 5) 두 패널 plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    scatter_panel(
        axes[0], text_pca, image_pca,
        xlabel="PC1", ylabel="PC2",
        title="(a) PCA on embeddings",
    )
    scatter_panel(
        axes[1], text_svd, image_svd,
        xlabel=r"$v_1$  (paired-diff direction)",
        ylabel=r"$v_2$",
        title="(b) SVD on paired modality difference (Ours)",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_path.name}")
    print(f"  [{subset}] elapsed: {time.time() - t0:.1f}s")

    return {
        "text_pca": text_pca, "image_pca": image_pca,
        "text_svd": text_svd, "image_svd": image_svd,
    }


# ─────────────────────────────────────────────
# 4 데이터셋 통합 figure
# ─────────────────────────────────────────────
def render_combined(results):
    """results: {subset: {"text_pca","image_pca","text_svd","image_svd"}}
    두 종류 figure 생성:
      (1) 2×2 grid — SVD-of-diff 만 (메인 메시지 강조용, 논문 main 후보)
      (2) 2×2 grid — PCA only (대조용)
      (3) 2×4 grid — PCA + SVD-of-diff 비교 (rows=method, cols=datasets)
    """
    subsets = [s for s in SUBSETS if s in results]
    if len(subsets) < 2:
        print("  combined figure skip (need ≥ 2 datasets)")
        return

    out_dir = _combined_dir()

    # ───────── (1) 2×2 grid: SVD-of-diff 만 ─────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 11.5))
    axes_flat = axes.flatten()
    for ax, subset in zip(axes_flat, subsets):
        r = results[subset]
        scatter_panel(
            ax, r["text_svd"], r["image_svd"],
            xlabel=r"$v_1$  (paired-diff direction)",
            ylabel=r"$v_2$",
            title=subset,
        )
    for ax in axes_flat[len(subsets):]:
        ax.axis("off")
    fig.suptitle("SVD on paired modality difference (Ours) — 4 datasets",
                 fontsize=15)
    fig.tight_layout()
    out_svd = out_dir / "fig_svd_diff_2x2.png"
    fig.savefig(out_svd, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved (combined): {out_svd.name}")

    # ───────── (2) 2×2 grid: PCA only ─────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 11.5))
    axes_flat = axes.flatten()
    for ax, subset in zip(axes_flat, subsets):
        r = results[subset]
        scatter_panel(
            ax, r["text_pca"], r["image_pca"],
            xlabel="PC1", ylabel="PC2",
            title=subset,
        )
    for ax in axes_flat[len(subsets):]:
        ax.axis("off")
    fig.suptitle("PCA on embeddings — 4 datasets",
                 fontsize=15)
    fig.tight_layout()
    out_pca = out_dir / "fig_pca_emb_2x2.png"
    fig.savefig(out_pca, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved (combined): {out_pca.name}")

    # ───────── (3) 2×4 grid: PCA vs SVD-of-diff 비교 ─────────
    fig, axes = plt.subplots(2, len(subsets), figsize=(5.5 * len(subsets), 11))
    if len(subsets) == 1:
        axes = axes.reshape(2, 1)
    for j, subset in enumerate(subsets):
        r = results[subset]
        # Row 0: PCA on embeddings
        scatter_panel(
            axes[0, j], r["text_pca"], r["image_pca"],
            xlabel="PC1", ylabel="PC2",
            title=f"{subset}",
        )
        # Row 1: SVD on paired diff
        scatter_panel(
            axes[1, j], r["text_svd"], r["image_svd"],
            xlabel=r"$v_1$", ylabel=r"$v_2$",
            title="",
        )
        # 행별 라벨
        if j == 0:
            axes[0, j].set_ylabel("PCA on embeddings\nPC2", fontsize=14)
            axes[1, j].set_ylabel("SVD on paired diff (Ours)\n$v_2$",
                                  fontsize=14)
    fig.suptitle("(top) PCA on embeddings  vs  (bottom) SVD on paired modality difference",
                 fontsize=14)
    fig.tight_layout()
    out_compare = out_dir / "fig_pca_vs_svd_diff_2x4.png"
    fig.savefig(out_compare, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved (combined): {out_compare.name}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=DEFAULT_SUBSET, choices=SUBSETS,
                    help="단일 데이터셋 (--all 과 동시 사용 불가)")
    ap.add_argument("--all", action="store_true",
                    help="4 개 데이터셋 모두 처리 + 통합 2×2 figure 생성")
    args = ap.parse_args()

    t0 = time.time()
    targets = SUBSETS if args.all else [args.subset]
    print(f"Targets: {targets}")

    results = {}
    for subset in targets:
        results[subset] = run_one(subset)

    # --all 또는 다중 subset 처리 시 통합 figure 생성
    if len(results) >= 2:
        print(f"\n{'#'*70}")
        print(f"  Combined figures ({len(results)} datasets)")
        print(f"{'#'*70}")
        render_combined(results)
        print(f"  → {_combined_dir()}")

    print(f"\n{'='*70}")
    print(f"  전체 완료. 총 시간: {time.time()-t0:.0f}s | datasets: {len(targets)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
