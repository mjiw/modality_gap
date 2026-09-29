"""
Multimodal Subspace Projection — t-SNE / UMAP / PCA Visualization (MixBench)
==============================================================================
원본 multimodal_subspace_visualization.py 의 MixBench 4 데이터셋 버전.
원본은 WIT paired 100K 를 사용하지만, 본 파일은 메인 실험과 동일한
MixBench 데이터셋 (MSCOCO / VisualNews / Google_WIT / OVEN) 을 사용한다.

3가지 방법:
  1. CLIP (Raw)             — baseline (보정 없음)
  2. GR-CLIP                — mean shift only
  3. STADIRA (Ours)         — 제안 방법 (mean + variance + subspace)

paired ('both') 아이템 수:
  MSCOCO=328, VisualNews=327, Google_WIT=1473, OVEN=334
모든 paired 아이템을 sub-sampling 없이 시각화에 사용.

실행:
  conda activate jiwoo
  # 기본 (Google_WIT)
  python multimodal_subspace_visualization_mixbench.py
  # 다른 데이터셋
  python multimodal_subspace_visualization_mixbench.py --subset MSCOCO
  # 캐시된 2D 좌표만 읽어 다시 그림 (수 초)
  python multimodal_subspace_visualization_mixbench.py --replot
  # 캐시 무시하고 강제 재계산
  python multimodal_subspace_visualization_mixbench.py --force
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
from mixbench_full_comparison_ext import (  # noqa: E402
    _l2norm_np, _project_out,
)

from sklearn.manifold import TSNE  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

try:
    import umap  # noqa: E402
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
DEFAULT_SUBSET = "Google_WIT"
SUBSETS = ["MSCOCO", "VisualNews", "Google_WIT", "OVEN"]

TSNE_PERPLEXITY = 30        # paired 가 적으면 perplexity 도 낮춰야 안정적
TSNE_MAX_ITER   = 1500

SVD_K = 16

STAGES = [
    ("clip",        "1. CLIP"),
    ("gr_clip",     "2. GR-CLIP"),
    ("zscore_svd",  f"3. STADIRA (Ours, k={SVD_K})"),
]


# 캐시 schema 식별자 (gap metric 셋 변경 시 올림)
_CACHE_SCHEMA = 2
_GAP_KEYS = ("align_cos", "align_l2", "uniform_T", "uniform_I",
             "wasserstein", "centroid_l2")


def _fig_dir(subset: str) -> Path:
    d = _HERE.parent / "MixBench_results" / subset / "figures_3way_mixbench"
    d.mkdir(exist_ok=True, parents=True)
    return d


def _combined_dir() -> Path:
    d = _HERE.parent / "MixBench_results" / "combined_4dataset"
    d.mkdir(exist_ok=True, parents=True)
    return d


def _cache_dir(subset: str) -> Path:
    d = _fig_dir(subset) / "cache_2d"
    d.mkdir(exist_ok=True, parents=True)
    return d


def _cache_path(subset: str, method: str, sname: str) -> Path:
    if sname == "zscore_svd":
        return _cache_dir(subset) / f"{method}__{sname}_k{SVD_K}.npz"
    return _cache_dir(subset) / f"{method}__{sname}.npz"


def _save_2d_cache(subset, method, sname, X2d, n_text, gap):
    payload = {
        "X2d": X2d.astype("float32"),
        "n_text": np.int32(n_text),
        "schema": np.int32(_CACHE_SCHEMA),
    }
    for k in _GAP_KEYS:
        payload[f"gap_{k}"] = np.float32(gap[k])
    np.savez(_cache_path(subset, method, sname), **payload)


def _load_2d_cache(subset, method, sname):
    p = _cache_path(subset, method, sname)
    if not p.exists():
        return None
    z = np.load(p)
    schema_ok = ("schema" in z.files and int(z["schema"]) == _CACHE_SCHEMA)
    keys_ok = all(f"gap_{k}" in z.files for k in _GAP_KEYS)
    if not (schema_ok and keys_ok):
        return None
    gap = {k: float(z[f"gap_{k}"]) for k in _GAP_KEYS}
    return z["X2d"], int(z["n_text"]), gap


# ─────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────
def _compute_topk_directions(t, i, both_idx, k):
    """paired 'both' 쌍의 (image - text) 차이에 SVD → top-k 방향."""
    diffs = i[both_idx] - t[both_idx]
    diffs = diffs - diffs.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(diffs, full_matrices=False)
    return Vt[:k].astype("float32")


def _scatter_modality(ax, X2d, n_text, title, *, show_legend=True):
    ax.scatter(X2d[:n_text, 0], X2d[:n_text, 1],
               c="#1565C0", s=10, alpha=0.75,
               label=f"text (n={n_text})", edgecolors="none")
    ax.scatter(X2d[n_text:, 0], X2d[n_text:, 1],
               c="#C62828", s=10, alpha=0.75,
               label=f"image (n={X2d.shape[0]-n_text})",
               edgecolors="none")
    ax.set_title(title)
    if show_legend:
        ax.legend(loc="best", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def _uniformity(emb, t=2.0, n_sample=2000):
    if len(emb) > n_sample:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(emb), n_sample, replace=False)
        emb = emb[idx]
    n = len(emb)
    sq = np.sum(emb ** 2, axis=1, keepdims=True)
    sq_dist = sq + sq.T - 2 * (emb @ emb.T)
    iu = np.triu_indices(n, k=1)
    sqs = np.clip(sq_dist[iu], 0, None)
    return float(np.log(np.exp(-t * sqs).mean() + 1e-12))


def _gap_metric(X_text, X_image):
    """Modality gap metrics (paired)."""
    align_cos = float((1.0 - (X_text * X_image).sum(axis=1)).mean())
    align_l2 = float(((X_text - X_image) ** 2).sum(axis=1).mean())
    uniform_T = _uniformity(X_text)
    uniform_I = _uniformity(X_image)
    mt = X_text.mean(axis=0); mi = X_image.mean(axis=0)
    vt = X_text.var(axis=0);  vi = X_image.var(axis=0)
    mean_d2 = float(np.sum((mt - mi) ** 2))
    var_d2  = float(np.sum((np.sqrt(vt) - np.sqrt(vi)) ** 2))
    wasserstein = float(np.sqrt(mean_d2 + var_d2))
    centroid_l2 = float(np.linalg.norm(mt - mi))
    return dict(align_cos=align_cos, align_l2=align_l2,
                uniform_T=uniform_T, uniform_I=uniform_I,
                wasserstein=wasserstein, centroid_l2=centroid_l2)


# ─────────────────────────────────────────────
# Stage 별 임베딩 계산
# ─────────────────────────────────────────────
def _build_stage(stage_name, text_emb, image_emb, paired_idx,
                 t_mean, i_mean, t_std, i_std):
    if stage_name == "clip":
        return (_l2norm_np(text_emb).astype("float32"),
                _l2norm_np(image_emb).astype("float32"))
    if stage_name == "gr_clip":
        return (_l2norm_np(text_emb - t_mean).astype("float32"),
                _l2norm_np(image_emb - i_mean).astype("float32"))
    if stage_name == "zscore_svd":
        t_z = _l2norm_np((text_emb - t_mean) / t_std).astype("float32")
        i_z = _l2norm_np((image_emb - i_mean) / i_std).astype("float32")
        dirs = _compute_topk_directions(t_z, i_z, paired_idx, SVD_K)
        t_p = _l2norm_np(_project_out(t_z, dirs)).astype("float32")
        i_p = _l2norm_np(_project_out(i_z, dirs)).astype("float32")
        return t_p, i_p
    raise ValueError(f"unknown stage: {stage_name}")


def _get_2d(subset, method, sname, *, force, replot,
            text_emb, image_emb, paired_idx, vis_idx,
            t_mean, i_mean, t_std, i_std):
    if not force:
        cached = _load_2d_cache(subset, method, sname)
        if cached is not None:
            return cached
        if replot:
            raise FileNotFoundError(
                f"--replot 모드인데 캐시 없음: {_cache_path(subset, method, sname)}"
            )
    t_p, i_p = _build_stage(sname, text_emb, image_emb, paired_idx,
                              t_mean, i_mean, t_std, i_std)
    X_text, X_image = t_p[vis_idx], i_p[vis_idx]
    gap = _gap_metric(X_text, X_image)
    Xall = np.vstack([X_text, X_image])

    n_paired = len(vis_idx)
    if method == "pca":
        X2d = PCA(n_components=2, random_state=SEED).fit_transform(Xall)
    elif method == "tsne":
        # paired 적을 때 perplexity 조정
        perp = min(TSNE_PERPLEXITY, max(5, (n_paired * 2 - 1) // 4))
        tsne = TSNE(n_components=2, perplexity=perp,
                    max_iter=TSNE_MAX_ITER, random_state=SEED,
                    init="pca", learning_rate="auto")
        X2d = tsne.fit_transform(Xall)
    elif method == "umap":
        if not HAS_UMAP:
            return None
        reducer = umap.UMAP(n_components=2, random_state=SEED,
                            n_neighbors=min(30, max(5, n_paired // 5)),
                            min_dist=0.1)
        X2d = reducer.fit_transform(Xall)
    else:
        raise ValueError(f"unknown method: {method}")

    n_text = len(X_text)
    _save_2d_cache(subset, method, sname, X2d, n_text, gap)
    return X2d, n_text, gap


def _render_grid(subset, method, suptitle, *, force, replot,
                 text_emb, image_emb, paired_idx, vis_idx,
                 t_mean, i_mean, t_std, i_std,
                 save_singles=False):
    n_stages = len(STAGES)
    fig, axes = plt.subplots(1, n_stages, figsize=(5 * n_stages, 5))
    if n_stages == 1:
        axes = [axes]

    gaps = []
    for idx, (ax, (sname, slabel)) in enumerate(zip(axes, STAGES)):
        ts = time.time()
        result = _get_2d(subset, method, sname, force=force, replot=replot,
                         text_emb=text_emb, image_emb=image_emb,
                         paired_idx=paired_idx, vis_idx=vis_idx,
                         t_mean=t_mean, i_mean=i_mean,
                         t_std=t_std, i_std=i_std)
        if result is None:
            return None
        X2d, n_text, gap = result
        gaps.append((sname, gap))
        _scatter_modality(ax, X2d, n_text, slabel)

        if save_singles:
            fig_s, ax_s = plt.subplots(1, 1, figsize=(7, 6))
            _scatter_modality(ax_s, X2d, n_text, slabel)
            fig_s.tight_layout()
            single_name = f"fig{idx+1}_{method}_{sname}.png"
            fig_s.savefig(_fig_dir(subset) / single_name, dpi=200,
                          bbox_inches="tight", facecolor="white")
            plt.close(fig_s)

        elapsed = time.time() - ts
        flag = "(cache)" if elapsed < 0.5 else f"({elapsed:.0f}s)"
        print(f"    {method} {sname} {flag}")

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    grid_name = f"fig_{method}_grid.png"
    fig.savefig(_fig_dir(subset) / grid_name, dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    saved {grid_name}")
    return gaps


METHOD_LABEL = {"pca": "PCA", "tsne": "t-SNE", "umap": "UMAP"}


def _render_3x3_grid(subset, *, force, replot,
                     text_emb, image_emb, paired_idx, vis_idx,
                     t_mean, i_mean, t_std, i_std):
    """3 methods (PCA, t-SNE, UMAP) × 3 stages (CLIP, GR-CLIP, STADIRA) 단일 figure.
    행 = 차원축소 방법, 열 = 보정 단계.
    출력: fig_3way_3method_combined_3x3.png
    """
    methods = ["pca", "tsne"] + (["umap"] if HAS_UMAP else [])
    n_rows = len(methods)
    n_cols = len(STAGES)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.6 * n_cols, 4.6 * n_rows))
    # 1×N 또는 N×1 일 때 axes shape 정규화
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for r, method in enumerate(methods):
        for c, (sname, slabel) in enumerate(STAGES):
            ax = axes[r, c]
            result = _get_2d(subset, method, sname, force=force, replot=replot,
                             text_emb=text_emb, image_emb=image_emb,
                             paired_idx=paired_idx, vis_idx=vis_idx,
                             t_mean=t_mean, i_mean=i_mean,
                             t_std=t_std, i_std=i_std)
            if result is None:
                ax.axis("off")
                continue
            X2d, n_text, gap = result

            # 한 패널 (좌상단) 에만 legend 표시
            show_legend = (r == 0 and c == 0)
            # 첫 행에만 stage 제목, 나머지 행은 제목 없음
            title = slabel if r == 0 else ""
            _scatter_modality(ax, X2d, n_text, title,
                              show_legend=show_legend)
            # 첫 열에만 method 이름을 y축 라벨로
            if c == 0:
                ax.set_ylabel(METHOD_LABEL.get(method, method.upper()),
                              fontsize=14, fontweight="bold",
                              rotation=90, labelpad=12)

    fig.suptitle(subset, fontsize=15)
    fig.tight_layout()
    out_path = _fig_dir(subset) / "fig_3way_3method_combined_3x3.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    saved {out_path.name}")


def _print_gap_table(gaps, subset, source=""):
    print(f"\n{'='*112}")
    suffix = f" (baseline = CLIP, source = {source})" if source else ""
    print(f"  [{subset}] Modality Gap Metrics by Stage{suffix}")
    print(f"{'='*112}")
    header = (f"  {'stage':<14s}  "
              f"{'align_cos':>10s}  {'align_l2':>10s}  "
              f"{'uniform_T':>10s}  {'uniform_I':>10s}  "
              f"{'wasserstein':>11s}  {'centroid_l2':>11s}")
    print(header)
    print("  " + "-" * 100)
    base = gaps[0][1]
    for sname, gap in gaps:
        print(f"  {sname:<14s}  "
              f"{gap['align_cos']:>10.4f}  {gap['align_l2']:>10.4f}  "
              f"{gap['uniform_T']:>10.4f}  {gap['uniform_I']:>10.4f}  "
              f"{gap['wasserstein']:>11.4f}  {gap['centroid_l2']:>11.4f}")
    print("  " + "-" * 100)
    print(f"  {'Δ vs CLIP':<14s}")
    for sname, gap in gaps[1:]:
        d = {k: gap[k] - base[k] for k in _GAP_KEYS}
        print(f"  {sname:<14s}  "
              f"{d['align_cos']:>+10.4f}  {d['align_l2']:>+10.4f}  "
              f"{d['uniform_T']:>+10.4f}  {d['uniform_I']:>+10.4f}  "
              f"{d['wasserstein']:>+11.4f}  {d['centroid_l2']:>+11.4f}")
    print("=" * 112)


# ─────────────────────────────────────────────
# 단일 subset 처리
# ─────────────────────────────────────────────
def run_one(subset, args):
    np.random.seed(SEED)
    t0 = time.time()

    mode_str = "REPLOT" if args.replot else ("FORCE" if args.force else "AUTO")
    print(f"\n{'#'*70}")
    print(f"  3-way Visualization (MixBench) | mode={mode_str} | subset={subset}")
    print(f"  SVD k = {SVD_K}")
    print(f"{'#'*70}")

    # ── 데이터 로드 ──
    methods = ["pca", "tsne"] + (["umap"] if HAS_UMAP else [])
    all_cached = all(
        _cache_path(subset, m, s).exists() for m in methods for (s, _) in STAGES
    )
    if args.replot and all_cached and not args.metrics_only:
        print(f"  ✓ 모든 캐시 존재 — 임베딩 로드 건너뜀")
        text_emb = image_emb = None
        paired_idx = vis_idx = None
        t_mean = i_mean = t_std = i_std = None
    else:
        text_emb, image_emb, _, modes, _ = load_mixbench(subset, "L14")
        text_mask = np.array([m in ("text", "both") for m in modes])
        image_mask = np.array([m in ("image", "both") for m in modes])
        t_mean = text_emb[text_mask].mean(axis=0, keepdims=True)
        i_mean = image_emb[image_mask].mean(axis=0, keepdims=True)
        t_std  = text_emb[text_mask].std (axis=0, keepdims=True) + 1e-8
        i_std  = image_emb[image_mask].std(axis=0, keepdims=True) + 1e-8
        paired_idx = np.where(np.array([m == "both" for m in modes]))[0]
        # MixBench paired 는 충분히 작아서 전부 사용
        vis_idx = paired_idx.copy()
        print(f"  paired anchors: {len(paired_idx)}, vis used: {len(vis_idx)}")

    if args.metrics_only:
        gaps = []
        for sname, _ in STAGES:
            t_p, i_p = _build_stage(sname, text_emb, image_emb, paired_idx,
                                     t_mean, i_mean, t_std, i_std)
            X_text, X_image = t_p[vis_idx], i_p[vis_idx]
            gaps.append((sname, _gap_metric(X_text, X_image)))
        _print_gap_table(gaps, subset,
                         source=f"768-D embedding, N_paired={len(vis_idx)}")
        print(f"\n총 시간: {time.time()-t0:.0f}s")
        return

    common_kwargs = dict(
        force=args.force, replot=args.replot,
        text_emb=text_emb, image_emb=image_emb,
        paired_idx=paired_idx, vis_idx=vis_idx,
        t_mean=t_mean, i_mean=i_mean, t_std=t_std, i_std=i_std,
    )

    # ── (A) PCA ──
    print(f"\n  [PCA 2D]")
    _render_grid(subset, "pca",
                 suptitle=f"{subset} | PCA 2D: CLIP → GR-CLIP → STADIRA",
                 save_singles=False, **common_kwargs)

    # ── (B) t-SNE ──
    print(f"\n  [t-SNE]")
    tsne_gaps = _render_grid(subset, "tsne",
                             suptitle=f"{subset} | t-SNE: CLIP → GR-CLIP → STADIRA",
                             save_singles=True, **common_kwargs)

    # ── (C) UMAP ──
    if HAS_UMAP:
        print(f"\n  [UMAP]")
        _render_grid(subset, "umap",
                     suptitle=f"{subset} | UMAP",
                     save_singles=False, **common_kwargs)
    else:
        print(f"\n  [UMAP] skipped — `pip install umap-learn`")

    if tsne_gaps:
        _print_gap_table(tsne_gaps, subset, source="t-SNE input embedding")

    # ── (D) 3×3 combined (3 methods × 3 stages) ──
    if HAS_UMAP:
        print(f"\n  [3×3 combined]")
        _render_3x3_grid(subset, **common_kwargs)
    else:
        print(f"\n  [3×3 combined] skipped — UMAP unavailable")

    print(f"\n  [{subset}] 시간: {time.time()-t0:.0f}s")
    print(f"  저장 위치: {_fig_dir(subset)}")


# ─────────────────────────────────────────────
# 4 데이터셋 통합 2×2 grid figures
# ─────────────────────────────────────────────
def render_combined_2x2(subsets):
    """캐시된 2D 좌표를 읽어 4 데이터셋 통합 2×2 grid 생성.
    각 (method, stage) 조합마다 별도 figure:
      fig_{method}_{stage}_combined_2x2.png
    예시:
      fig_tsne_clip_combined_2x2.png       ← Intro 후보 (raw t-SNE 4 datasets)
      fig_tsne_zscore_svd_combined_2x2.png ← 본 방법 적용 후 4 datasets
    """
    out_dir = _combined_dir()
    methods = ["pca", "tsne"] + (["umap"] if HAS_UMAP else [])

    for method in methods:
        for (sname, slabel) in STAGES:
            # 4 데이터셋 모두 캐시 존재해야 통합 figure 생성
            cached_all = []
            for subset in subsets:
                c = _load_2d_cache(subset, method, sname)
                if c is None:
                    cached_all = None
                    break
                cached_all.append((subset, c))
            if cached_all is None:
                print(f"  combined skip — 캐시 부족: {method} / {sname}")
                continue

            fig, axes = plt.subplots(2, 2, figsize=(11.5, 11))
            axes_flat = axes.flatten()
            for ax, (subset, (X2d, n_text, gap)) in zip(axes_flat, cached_all):
                _scatter_modality(ax, X2d, n_text, subset)
            for ax in axes_flat[len(cached_all):]:
                ax.axis("off")
            fig.suptitle(f"{method.upper()}  |  {slabel}",
                         fontsize=14)
            fig.tight_layout()
            out_path = out_dir / f"fig_{method}_{sname}_combined_2x2.png"
            fig.savefig(out_path, dpi=200,
                        bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"  saved: {out_path.name}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=DEFAULT_SUBSET, choices=SUBSETS,
                    help="단일 데이터셋 (--all 과 동시 사용 불가)")
    ap.add_argument("--all", action="store_true",
                    help="4 개 데이터셋 모두 처리 + 통합 2×2 figure 생성")
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--force",  action="store_true")
    ap.add_argument("--metrics-only", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    targets = SUBSETS if args.all else [args.subset]
    print(f"Targets: {targets}")

    for subset in targets:
        run_one(subset, args)

    # 다중 subset 처리 시 통합 2x2 figure 생성
    if len(targets) >= 2 and not args.metrics_only:
        print(f"\n{'#'*70}")
        print(f"  Combined 2×2 figures ({len(targets)} datasets)")
        print(f"{'#'*70}")
        render_combined_2x2(targets)
        print(f"  → {_combined_dir()}")

    print(f"\n{'='*70}")
    print(f"  전체 완료. 총 시간: {time.time()-t0:.0f}s | datasets: {len(targets)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
