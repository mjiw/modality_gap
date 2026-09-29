"""
fig_spectrum_backbone_robustness.py
==========================================================================
SVD-of-difference 의 elbow 구조가 **backbone 에 무관함** 을 입증하는
supplementary figure.

목적: reviewer 가 "spectrum 의 elbow 가 ViT-L/14 에만 나타나는 데이터셋
specific 한 현상 아니냐" 라고 지적할 수 있는 가능성을 차단.

분석:
  각 (backbone, dataset) 마다 paired diff 행렬의 σ_k 를 계산.
  각 backbone 마다 4 데이터셋의 σ_k / σ_1 (normalized) 곡선을 overlay.
  → 모든 (backbone, dataset) 조합에서 같은 elbow 모양이 나타나면,
     이 구조는 CLIP-style contrastive learning 의 본질적 성질임을
     경험적으로 입증.

비교 backbone (4 개 — open_clip 계열):
  - CLIP-L/14         (본 연구 메인)
  - CLIP-B/16         (작은 사이즈, OpenAI)
  - OpenCLIP-L/14     (다른 학습 데이터)
  - SigLIP-400m       (다른 objective)

normalized 표시 이유:
  - backbone 마다 임베딩 절대 크기 다름 → 정규화로 shape 비교에 집중

출력:
  MixBench_results/combined_4dataset/fig_spectrum_backbone_robustness.png
  MixBench_results/combined_4dataset/backbone_spectrum_sigmas.json

실행:
  conda activate jiwoo
  python fig_spectrum_backbone_robustness.py
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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

from data_loader import load_mixbench  # noqa: E402
from mixbench_full_comparison_ext import _l2norm_np  # noqa: E402


# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
BACKBONES = ["L14", "B16", "OC_L14", "SigLIP400M"]
BACKBONE_LABELS = {
    "L14":        "CLIP-L/14",
    "B16":        "CLIP-B/16",
    "OC_L14":     "OpenCLIP-L/14",
    "SigLIP400M": "SigLIP-400m",
}

DATASETS = ["MSCOCO", "VisualNews", "Google_WIT", "OVEN"]
DATASET_COLORS = {
    "MSCOCO":     "#D32F2F",   # red
    "VisualNews": "#1976D2",   # blue
    "Google_WIT": "#388E3C",   # green
    "OVEN":       "#7B1FA2",   # purple
}

OUT_DIR = _HERE.parent / "MixBench_results" / "combined_4dataset"
OUT_DIR.mkdir(exist_ok=True, parents=True)
OUT_FIG  = OUT_DIR / "fig_spectrum_backbone_robustness.png"
OUT_JSON = OUT_DIR / "backbone_spectrum_sigmas.json"


# ─────────────────────────────────────────────
# 단일 (backbone, dataset) 의 sigmas 계산
# ─────────────────────────────────────────────
def disp_name(name):
    """데이터셋 내부 식별자 → 표시용 이름.
    내부 키(캐시 폴더명 cache_Google_WIT_L14/, JSON 키)는 그대로 두고
    figure 라벨에서만 하이픈 표기를 쓴다."""
    return {"Google_WIT": "Google-WIT"}.get(name, name)


def compute_sigmas(text_emb, image_emb, modes):
    """Z-score per modality → L2 → paired diff → centered SVD → sigmas."""
    text_mask  = np.array([m in ("text", "both")  for m in modes])
    image_mask = np.array([m in ("image", "both") for m in modes])
    both_mask  = np.array([m == "both" for m in modes])

    if both_mask.sum() < 2:
        return None

    t_mean = text_emb [text_mask ].mean(axis=0, keepdims=True)
    i_mean = image_emb[image_mask].mean(axis=0, keepdims=True)
    t_std  = text_emb [text_mask ].std (axis=0, keepdims=True) + 1e-8
    i_std  = image_emb[image_mask].std(axis=0, keepdims=True) + 1e-8

    t_z = _l2norm_np((text_emb  - t_mean) / t_std).astype("float32")
    i_z = _l2norm_np((image_emb - i_mean) / i_std).astype("float32")

    diffs = i_z[both_mask] - t_z[both_mask]
    diffs = diffs - diffs.mean(axis=0, keepdims=True)
    _, sigmas, _ = np.linalg.svd(diffs, full_matrices=False)
    return sigmas


# ─────────────────────────────────────────────
# 단일 패널 plot (1 backbone, 4 dataset overlay)
# ─────────────────────────────────────────────
def plot_one_panel(ax, sigmas_per_dataset, backbone_label):
    for ds in DATASETS:
        if ds not in sigmas_per_dataset:
            continue
        s = sigmas_per_dataset[ds]
        s_norm = s / s[0]                # σ_1 = 1 로 정규화
        ks = np.arange(1, len(s) + 1)
        ax.plot(ks, s_norm,
                color=DATASET_COLORS[ds],
                linewidth=1.8, label=disp_name(ds), zorder=3)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("k", fontsize=14)
    ax.set_ylabel(r"$\sigma_k / \sigma_1$ (normalized)", fontsize=14)
    ax.set_title(backbone_label, fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=10, framealpha=0.95)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    t0 = time.time()
    print(f"[Backbone robustness] backbones={BACKBONES}, datasets={DATASETS}")

    all_sigmas = {}   # bb → ds → list of sigmas
    for bb in BACKBONES:
        all_sigmas[bb] = {}
        print(f"\n  Backbone: {bb} ({BACKBONE_LABELS[bb]})")
        for ds in DATASETS:
            try:
                ts = time.time()
                text_emb, image_emb, _, modes, _ = load_mixbench(ds, bb)
                sigmas = compute_sigmas(text_emb, image_emb, modes)
                if sigmas is None:
                    print(f"    {ds:<12s}  SKIP (no 'both' pairs)")
                    continue
                all_sigmas[bb][ds] = sigmas
                el = time.time() - ts
                print(f"    {ds:<12s}  rank={len(sigmas):>3d}  "
                      f"σ_1={sigmas[0]:.3f}  σ_16={sigmas[15]:.3f}  "
                      f"({el:.1f}s)")
            except Exception as e:
                print(f"    {ds:<12s}  SKIP: {type(e).__name__}: {e}")

    # ── Plot 2×2 grid (4 backbones) ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.flatten()

    for ax, bb in zip(axes_flat, BACKBONES):
        if not all_sigmas[bb]:
            ax.axis("off")
            ax.set_title(f"{BACKBONE_LABELS[bb]} (cache missing)",
                         fontsize=12)
            continue
        plot_one_panel(ax, all_sigmas[bb], BACKBONE_LABELS[bb])

    fig.suptitle(
        "Singular value spectrum across CLIP-style backbones — "
        "elbow structure is backbone-independent",
        fontsize=14, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  saved figure: {OUT_FIG.name}")

    # ── JSON 저장 (raw sigmas) ──
    save = {bb: {ds: s.tolist() for ds, s in sd.items()}
            for bb, sd in all_sigmas.items()}
    with open(OUT_JSON, "w") as f:
        json.dump(save, f, indent=2)
    print(f"  saved json:   {OUT_JSON.name}")

    print(f"\n총 시간: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
