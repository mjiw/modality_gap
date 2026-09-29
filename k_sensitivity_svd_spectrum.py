"""
k_sensitivity_svd_spectrum.py
=====================================================================
K=16 elbow 선택의 **이론적 근거 보조 figure** 3종 생성.

목적:
  메인 실험(mixbench_k_sensitivity_v2.py)이 보여주는 "k=16 empirical
  elbow" 가 단순 경험적 sweet spot 이 아닌, paired modality difference
  의 SVD 구조(modality direction 분포)에서 자연스럽게 도출됨을 입증.

세 가지 보조 figure:

  [Figure A] Singular value spectrum (σ_k vs k)
    paired modality difference 행렬 D 의 SVD 의 singular value 분포.
    σ가 k≈16 부근에서 급격히 작아지며 elbow 형성을 시각화.
    각 데이터셋별 1 panel.

  [Figure B] Cumulative variance explained
    누적 분산 비율 Σ_(i≤k) σ_i² / Σ σ_i² (log-x).
    "k=16 만으로 분산의 XX% 가 흡수된다" 라는 정량적 해석.

  [Figure C] Top-k vs Bottom-k projection — empirical validation
    Top-k 제거 (우리 방법, 큰 σ 부터) vs Bottom-k 제거 (작은 σ 부터)
    의 NDCG@10 비교. 가설:
      Top-k 제거 → modality 보정 → 성능 증가 후 elbow
      Bottom-k 제거 → semantic 정보 손상 → 즉시 감소
    → "top-k = modality, bottom-k = semantic" 의 empirical 검증.

출력 (MixBench_results/k_sensitivity_v2/):
    fig_svd_spectrum.png         (Figure A)
    fig_svd_cumulative.png       (Figure B)
    fig_topk_vs_bottomk.png      (Figure C, optional)
    svd_spectrum.json            (raw singular values + Top/Bottom-k 결과)

실행:
  conda activate tablerag   # 또는 jiwoo
  python k_sensitivity_svd_spectrum.py
  python k_sensitivity_svd_spectrum.py --subsets MSCOCO,OVEN
  python k_sensitivity_svd_spectrum.py --skip-fig-c   # Figure C 생략 (빠른 실행)
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data_loader import (  # noqa: E402
    load_mixbench, evaluate_ranking, ALPHAS,
)
from mixbench_full_comparison_ext import (  # noqa: E402
    _build_from_precorrected, _l2norm_np, _project_out,
)

# UTF-8 stdout (Windows cp949 호환)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
DATASETS_ORDER = ["MSCOCO", "VisualNews", "Google_WIT", "OVEN"]
K_FIXED = 16
KS_PROJ_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128]

# 색상 (다른 plot 스크립트와 일관)
# 채도·명도를 ablation_bar_chart.py / ratio figure (tab10 대역) 와 통일.
# Ours 의 빨강은 막대그래프의 STADIRA 와 정확히 같은 값을 쓴다.
COL_MOD     = "#2D8E32"   # green:  spectrum 곡선 (Figure A)
COL_MARKER  = "#1C70C3"   # blue:   k=16 marker
# Figure C (Top-k vs Bottom-k)
COL_TOP     = "#D62728"   # red:    Top-k (Ours) — STADIRA 와 동일 (통일 위해 고정)
COL_BOT     = "#E77F13"   # orange: Bottom-k
COL_SHUF    = "#7E57C2"   # muted violet: Shuffled pairs (tab:purple 대역, 유지)


# ─────────────────────────────────────────────
# 임베딩 전처리 (Z-Score + L2 normalize)
# ─────────────────────────────────────────────
def disp_name(name):
    """데이터셋 내부 식별자 → 표시용 이름.
    내부 키(캐시 폴더명 cache_Google_WIT_L14/, JSON 키)는 그대로 두고
    figure 라벨에서만 하이픈 표기를 쓴다."""
    return {"Google_WIT": "Google-WIT"}.get(name, name)


def compute_zscored(text_emb, image_emb, query_emb, modes):
    text_mask = np.array([m in ("text", "both") for m in modes])
    image_mask = np.array([m in ("image", "both") for m in modes])
    text_mean = text_emb[text_mask].mean(axis=0, keepdims=True)
    image_mean = image_emb[image_mask].mean(axis=0, keepdims=True)
    text_std = text_emb[text_mask].std(axis=0, keepdims=True) + 1e-8
    image_std = image_emb[image_mask].std(axis=0, keepdims=True) + 1e-8
    query_mean = query_emb.mean(axis=0, keepdims=True)
    query_std = query_emb.std(axis=0, keepdims=True) + 1e-8

    t_z = _l2norm_np((text_emb - text_mean) / text_std).astype("float32")
    i_z = _l2norm_np((image_emb - image_mean) / image_std).astype("float32")
    q_z = _l2norm_np((query_emb - query_mean) / query_std).astype("float32")
    return t_z, i_z, q_z


def compute_full_svd(t_z, i_z, modes):
    """paired diff 행렬 (centered) 의 full SVD.
    반환: sigmas (rank,), Vt_full (rank, D)
    """
    both_mask = np.array([m == "both" for m in modes])
    if both_mask.sum() < 2:
        return None, None
    diffs = i_z[both_mask] - t_z[both_mask]
    diffs_centered = diffs - diffs.mean(axis=0, keepdims=True)
    _, sigmas, Vt = np.linalg.svd(diffs_centered, full_matrices=False)
    return sigmas, Vt


# ─────────────────────────────────────────────
# [Figure A] Singular value spectrum
# ─────────────────────────────────────────────
def plot_spectrum(all_sigmas, out_path, k_fixed=K_FIXED):
    subsets = list(all_sigmas.keys())
    n = len(subsets)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, subset in zip(axes, subsets):
        sigmas = all_sigmas[subset]
        ks = np.arange(1, len(sigmas) + 1)

        ax.plot(ks, sigmas, color=COL_MOD, linewidth=1.8, zorder=3)

        # k=16 elbow 강조
        if k_fixed <= len(sigmas):
            sigma_at_k = sigmas[k_fixed - 1]
            ax.axvline(k_fixed, color=COL_MARKER, linestyle="--",
                       linewidth=1.4, alpha=0.7,
                       label=f"$k$={k_fixed}")
            ax.scatter([k_fixed], [sigma_at_k], color=COL_MARKER,
                       s=110, marker="*", edgecolors="black",
                       linewidth=0.7, zorder=5)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("$k$", fontsize=11)
        ax.set_ylabel(r"$\sigma_k$ (log scale)", fontsize=11)
        ax.set_title(disp_name(subset), fontsize=12)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=9, loc="best", framealpha=0.95)

    fig.suptitle("Singular value spectrum of paired modality difference matrix",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_path.name}")


# ─────────────────────────────────────────────
# [Figure B] Cumulative variance explained
# ─────────────────────────────────────────────
def plot_cumulative_variance(all_sigmas, out_path, k_fixed=K_FIXED):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # 데이터셋별 cumulative variance 곡선
    colors = ["#D32F2F", "#1976D2", "#388E3C", "#7B1FA2"]
    for (subset, sigmas), color in zip(all_sigmas.items(), colors):
        sigmas2 = sigmas ** 2
        cumvar = np.cumsum(sigmas2) / sigmas2.sum()
        ks = np.arange(1, len(sigmas) + 1)
        ax.plot(ks, cumvar, color=color, linewidth=1.8, label=disp_name(subset))

        # k=16 위치 annotation
        if k_fixed <= len(sigmas):
            cv_at_k = cumvar[k_fixed - 1]
            ax.scatter([k_fixed], [cv_at_k], color=color, s=80,
                       marker="o", edgecolors="black", linewidth=0.6,
                       zorder=5)
            ax.annotate(f"{cv_at_k * 100:.1f}%",
                        xy=(k_fixed, cv_at_k),
                        xytext=(k_fixed * 1.4, cv_at_k - 0.04),
                        fontsize=9, color=color, fontweight="bold")

    ax.axvline(k_fixed, color="black", linestyle="--", linewidth=1.2,
               alpha=0.5, label=f"$k$={k_fixed}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("$k$", fontsize=11)
    ax.set_ylabel("Cumulative variance explained", fontsize=11)
    ax.set_title("Cumulative variance explained by top-k SVD directions\n"
                 "(paired modality difference matrix)", fontsize=12)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=10, loc="lower right", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_path.name}")


# ─────────────────────────────────────────────
# [Figure C] Top-k vs Bottom-k projection — empirical validation
# ─────────────────────────────────────────────
def evaluate_projection_best_alpha(t_z, i_z, q_z, modes, q_gt, V, alphas):
    """V (k × D) 방향 제거 후 best α 선택."""
    t_p = _l2norm_np(_project_out(t_z, V)).astype("float32")
    i_p = _l2norm_np(_project_out(i_z, V)).astype("float32")
    q_p = _l2norm_np(_project_out(q_z, V)).astype("float32")

    best_mrr = -1.0
    best_res = None
    for alpha in alphas:
        corpus = _build_from_precorrected(t_p, i_p, modes, alpha)
        scores = q_p @ corpus.T
        res = evaluate_ranking(scores, q_gt, modes=modes)
        if res["MRR"] > best_mrr:
            best_mrr = res["MRR"]
            best_res = res
    return best_res


def compute_shuffled_svd(t_z, i_z, modes, seed):
    """pairing 을 무작위로 깨뜨린 paired-diff 의 SVD.

    'both' 쌍에서 image 쪽만 permutation 해서 (i_z[perm] - t_z) 를 만든다.
    → 행렬의 주변 분포(marginal)와 rank 는 그대로 두고 '올바른 쌍' 이라는
      정보만 제거한 negative control.
    반환: Vt_full (rank, D)  — 실패 시 None
    """
    both_mask = np.array([m == "both" for m in modes])
    if both_mask.sum() < 2:
        return None
    t_b = t_z[both_mask]
    i_b = i_z[both_mask]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(t_b))
    diffs = i_b[perm] - t_b
    diffs_centered = diffs - diffs.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(diffs_centered, full_matrices=False)
    return Vt


def run_shuffled_control(cache, ks, n_seeds, out_path, alphas=None):
    """Shuffled paired-diff control 을 Top-k/Bottom-k 와 **같은 run** 에서 계산.

    baseline 도 같은 파이프라인에서 나오므로 곡선 3종이 하나의 기준선을
    공유한다 (별도 run 으로 돌렸을 때 생기던 baseline 불일치가 사라짐).
    결과는 shuffled_control.json 형식으로 저장.
    """
    alphas = ALPHAS if alphas is None else alphas
    baseline, true_at, mean_at, std_at = {}, {}, {}, {}
    for subset, d in cache.items():
        print(f"  [{subset}] shuffled control: {n_seeds} seeds × {len(ks)} k ...")
        base_res = evaluate_projection_best_alpha(
            d["t_z"], d["i_z"], d["q_z"], d["modes"], d["q_gt"],
            np.zeros((0, d["t_z"].shape[1]), dtype="float32"), alphas,
        )
        baseline[subset] = base_res["NDCG@10"]

        Vt_true = d["Vt_full"]
        D = Vt_true.shape[0]
        Vt_shuf = [compute_shuffled_svd(d["t_z"], d["i_z"], d["modes"], s)
                   for s in range(n_seeds)]
        Vt_shuf = [v for v in Vt_shuf if v is not None]

        t_map, m_map, s_map = {}, {}, {}
        for k in ks:
            if k > D:
                continue
            t_map[k] = evaluate_projection_best_alpha(
                d["t_z"], d["i_z"], d["q_z"], d["modes"], d["q_gt"],
                Vt_true[:k], alphas)["NDCG@10"]
            vals = [evaluate_projection_best_alpha(
                        d["t_z"], d["i_z"], d["q_z"], d["modes"], d["q_gt"],
                        V[:k], alphas)["NDCG@10"]
                    for V in Vt_shuf]
            m_map[k] = float(np.mean(vals))
            s_map[k] = float(np.std(vals))
            print(f"    k={k:>3d}: True={t_map[k]:.4f}  "
                  f"Shuf={m_map[k]:.4f}±{s_map[k]:.4f}  "
                  f"gap={t_map[k]-m_map[k]:+.4f}")
        true_at[subset], mean_at[subset], std_at[subset] = t_map, m_map, s_map

    save = {
        "_comment": "Top-k/Bottom-k 와 동일 run 에서 계산됨 (baseline 공유).",
        "n_seeds": n_seeds,
        "alphas": list(alphas),
        "ks": list(ks),
        "baseline_ndcg10": baseline,
        "true_ndcg10": {s: {str(k): v for k, v in m.items()}
                        for s, m in true_at.items()},
        "shuffled_mean_ndcg10": {s: {str(k): v for k, v in m.items()}
                                 for s, m in mean_at.items()},
        "shuffled_std_ndcg10": {s: {str(k): v for k, v in m.items()}
                                for s, m in std_at.items()},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save, f, indent=2, ensure_ascii=False)
    print(f"  saved: {Path(out_path).name}")


def load_shuffled_control(path):
    """Shuffled paired-diff control (5 seeds) 결과 로드. 없으면 None.

    형식: shuffled_control.json
      baseline_ndcg10 / true_ndcg10 / shuffled_mean_ndcg10 / shuffled_std_ndcg10
    주의: 이 control 은 별도 run 이라 자체 baseline 을 가진다 (Top-k/Bottom-k
    run 의 baseline 과 소수점 3~4자리에서 다를 수 있음).
    """
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    for subset in d["shuffled_mean_ndcg10"]:
        out[subset] = {
            "baseline": d["baseline_ndcg10"][subset],
            "mean": {int(k): v for k, v in d["shuffled_mean_ndcg10"][subset].items()},
            "std": {int(k): v for k, v in d["shuffled_std_ndcg10"][subset].items()},
        }
    return {"per_subset": out, "n_seeds": d.get("n_seeds")}


def plot_topk_vs_bottomk(all_topk, all_bottomk, all_baseline, ks,
                         out_path, k_fixed=K_FIXED, delta=False,
                         shuffled=None, alpha_fixed=None):
    """delta=False: 절대 NDCG@10 (패널별 y축 autoscale)
    delta=True : baseline 대비 ΔNDCG@10 (4개 패널 y축 공유)

    Δ 버전은 y 단위가 모든 패널에서 동일하고 0선이 곧 no-projection
    baseline 이므로, "패널마다 y축 범위가 달라 효과가 과장돼 보인다"
    는 지적에 대한 방어용으로 함께 저장한다.

    shuffled: load_shuffled_control() 결과. 주면 세 번째 곡선
      (Shuffled paired-diff, mean ± std 밴드) 를 함께 그린다.
      delta 모드에서는 shuffled 자신의 run baseline 을 빼서
      서로 다른 run 의 offset 차이를 제거한다.
    """
    subsets = list(all_topk.keys())
    n = len(subsets)
    n_cols = 2 if n >= 2 else 1
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.6 * n_cols, 4.8 * n_rows))
    if n == 1:
        axes_list = [axes]
    else:
        axes_list = np.array(axes).flatten().tolist()

    # Δ 모드: 전 패널 공통 y-limit 사전 계산
    shared_ylim = None
    if delta:
        allv = []
        for s in subsets:
            b = all_baseline.get(s)
            if b is None:
                continue
            allv += [all_topk[s][k] - b for k in all_topk[s]]
            allv += [all_bottomk[s][k] - b for k in all_bottomk[s]]
            sh = (shuffled or {}).get("per_subset", {}).get(s)
            if sh is not None:
                sb = sh["baseline"]
                allv += [sh["mean"][k] - sh["std"][k] - sb for k in sh["mean"]]
                allv += [sh["mean"][k] + sh["std"][k] - sb for k in sh["mean"]]
        if allv:
            lo, hi = min(allv), max(allv)
            pad = 0.10 * (hi - lo)
            shared_ylim = (lo - pad, hi + pad)

    for ax, subset in zip(axes_list, subsets):
        ks_plot = [k for k in ks if k in all_topk[subset]]
        baseline = all_baseline.get(subset)
        off = baseline if (delta and baseline is not None) else 0.0
        top_vals = [all_topk[subset][k] - off for k in ks_plot]
        bot_vals = [all_bottomk[subset][k] - off for k in ks_plot]

        # baseline (k=0, no projection)
        if baseline is not None:
            ax.axhline(baseline - off, color="#222222", linestyle=":",
                       linewidth=2.2, alpha=1.0, label="Z-score only ($k$=0)")

        # Top-k removal (Ours)
        ax.plot(ks_plot, top_vals, color=COL_TOP, linewidth=2,
                marker="D", markersize=7,
                label="Top-$k$ (Ours)", zorder=3)

        # Bottom-k removal (control)
        ax.plot(ks_plot, bot_vals, color=COL_BOT, linewidth=2,
                marker="s", markersize=7,
                label="Bottom-$k$", zorder=3)

        # Shuffled paired-diff removal (control, 5 seeds: mean ± std)
        sh = (shuffled or {}).get("per_subset", {}).get(subset)
        if sh is not None:
            # 별도 run 이므로 delta 모드에서는 자신의 baseline 을 뺀다
            sh_off = sh["baseline"] if delta else 0.0
            ks_sh = [k for k in ks_plot if k in sh["mean"]]
            mu = np.array([sh["mean"][k] for k in ks_sh]) - sh_off
            sd = np.array([sh["std"][k] for k in ks_sh])
            n_seeds = (shuffled or {}).get("n_seeds")
            seed_txt = f", {n_seeds} seeds" if n_seeds else ""
            ax.fill_between(ks_sh, mu - sd, mu + sd,
                            color=COL_SHUF, alpha=0.13, linewidth=0, zorder=2)
            ax.plot(ks_sh, mu, color=COL_SHUF, linewidth=2,
                    marker="o", markersize=6, linestyle="-",
                    label=f"Shuffled pairs{seed_txt}", zorder=3)

        # k=16 vertical line (selected k)
        ax.axvline(k_fixed, color=COL_MARKER, linestyle="--",
                   linewidth=1.4, alpha=0.7,
                   label=f"$k$={k_fixed}")

        # k=16 peak marker for Top-k
        if k_fixed in all_topk[subset]:
            ax.scatter([k_fixed], [all_topk[subset][k_fixed] - off],
                       color=COL_MARKER, s=120, marker="*",
                       edgecolors="black", linewidth=0.7, zorder=5)

        ax.set_xscale("log", base=2)
        ax.set_xlabel("$k$ (number of removed directions)", fontsize=15)
        # 기준점(Z-score only, k=0)·α 설정은 캡션/범례로 넘기고
        # 축에는 측정량만 남긴다.
        ax.set_ylabel(r"$\Delta$ NDCG@10" if delta
                      else "NDCG@10", fontsize=15)
        ax.tick_params(axis="both", labelsize=12)
        ax.set_title(disp_name(subset), fontsize=12)
        ax.grid(True, alpha=0.3, which="both")
        if shared_ylim is not None:
            ax.set_ylim(*shared_ylim)

        # 범례는 패널마다 좌하단에 (곡선이 좌하단을 거의 쓰지 않음)
        h, l = ax.get_legend_handles_labels()
        ax.legend(h, l, loc="lower left", fontsize=9, ncol=1,
                  frameon=True, framealpha=0.95, edgecolor="#BDBDBD",
                  borderpad=0.5, labelspacing=0.35, handlelength=1.8)

    # 남는 빈 패널 숨기기
    for ax in axes_list[len(subsets):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_path.name}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", default=",".join(DATASETS_ORDER),
                    help="comma-separated subset 이름")
    ap.add_argument("--ks", default=",".join(str(k) for k in KS_PROJ_SWEEP),
                    help="Top-k/Bottom-k 비교용 k 값")
    ap.add_argument("--skip-fig-c", action="store_true",
                    help="Figure C (Top-k vs Bottom-k) 생략 — 빠른 실행")
    ap.add_argument("--shuffled-json", default=None,
                    help="Shuffled paired-diff control JSON 경로 "
                         "(default: {out_dir}/shuffled_control.json)")
    ap.add_argument("--replot", action="store_true",
                    help="재계산 없이 svd_spectrum.json 만 읽어 Figure C 재생성")
    ap.add_argument("--alpha", type=float, default=None,
                    help="α 를 이 값으로 고정 (예: 0.5). 생략하면 "
                         "0.0~1.0 sweep 후 MRR 최대 α 선택(기존 동작). "
                         "고정 시 출력 파일명에 _a0.5 접미사가 붙어 "
                         "sweep 결과를 덮어쓰지 않음.")
    ap.add_argument("--shuffled-seeds", type=int, default=0,
                    help="Shuffled paired-diff control 을 이 seed 수만큼 "
                         "같은 run 에서 새로 계산 (0 = 계산 안 함, "
                         "기존 shuffled_control.json 재사용). 예: 10")
    args = ap.parse_args()

    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    ks = sorted(set(int(k.strip()) for k in args.ks.split(",")))

    out_dir = _HERE.parent / "MixBench_results" / "k_sensitivity_v2"
    out_dir.mkdir(exist_ok=True, parents=True)

    # α: 고정(--alpha) 또는 기존 sweep. 고정 시 파일명에 접미사를 붙여
    # sweep 결과(svd_spectrum.json 등)를 덮어쓰지 않는다.
    alphas = ALPHAS if args.alpha is None else [args.alpha]
    sfx = "" if args.alpha is None else f"_a{args.alpha:g}"
    SPEC_JSON = out_dir / f"svd_spectrum{sfx}.json"
    SHUF_JSON = out_dir / f"shuffled_control{sfx}.json"
    FIG_ABS   = out_dir / f"fig_topk_vs_bottomk{sfx}.png"
    FIG_DELTA = out_dir / f"fig_topk_vs_bottomk_delta{sfx}.png"

    # ── replot 모드: 저장된 결과만으로 Figure C 재생성 (α sweep 재계산 없음) ──
    if args.replot:
        src = SPEC_JSON
        with open(src, encoding="utf-8") as f:
            saved = json.load(f)
        if "topk_ndcg10" not in saved:
            raise SystemExit(
                f"{src.name} 에 Figure C 결과가 없습니다 "
                f"(--skip-fig-c 로 생성된 파일). 전체 실행이 필요합니다.")
        to_int = lambda d: {s: {int(k): v for k, v in vals.items()}
                            for s, vals in d.items()}
        all_topk = to_int(saved["topk_ndcg10"])
        all_bottomk = to_int(saved["bottomk_ndcg10"])
        all_baseline = saved["baseline_ndcg10"]
        all_topk = {s: all_topk[s] for s in subsets if s in all_topk}
        all_bottomk = {s: all_bottomk[s] for s in subsets if s in all_bottomk}
        shuf = load_shuffled_control(args.shuffled_json or SHUF_JSON)
        print(f"[replot] {src.name} 로드 — {len(all_topk)} subsets"
              + ("" if shuf is None else
                 f", shuffled control {shuf['n_seeds']} seeds"))
        plot_topk_vs_bottomk(all_topk, all_bottomk, all_baseline, ks,
                             FIG_ABS, shuffled=shuf, alpha_fixed=args.alpha)
        plot_topk_vs_bottomk(all_topk, all_bottomk, all_baseline, ks,
                             FIG_DELTA, delta=True, shuffled=shuf,
                             alpha_fixed=args.alpha)
        return

    print(f"Output dir: {out_dir}")
    print(f"Subsets: {subsets}")

    # ── 1) 데이터셋별 임베딩 로드 + SVD ──
    all_sigmas = {}
    cache = {}   # Figure C 용 임베딩 캐시
    for subset in subsets:
        print(f"\n[{subset}] loading + SVD ...")
        text_emb, image_emb, query_emb, modes, q_gt = load_mixbench(subset)
        t_z, i_z, q_z = compute_zscored(text_emb, image_emb, query_emb, modes)
        sigmas, Vt_full = compute_full_svd(t_z, i_z, modes)
        if sigmas is None:
            print(f"  Skip {subset} (no 'both' pairs).")
            continue
        all_sigmas[subset] = sigmas
        cache[subset] = {
            "t_z": t_z, "i_z": i_z, "q_z": q_z,
            "modes": modes, "q_gt": q_gt,
            "Vt_full": Vt_full,
        }
        print(f"  shape: sigmas={sigmas.shape}, "
              f"σ_1={sigmas[0]:.4f}, σ_16={sigmas[15]:.4f}, "
              f"σ_min={sigmas[-1]:.4f}")

    if not all_sigmas:
        print("No subsets loaded successfully. Exit.")
        sys.exit(1)

    # ── 2) Figure A: spectrum ──
    print(f"\n[Figure A] Singular value spectrum")
    plot_spectrum(all_sigmas, out_dir / "fig_svd_spectrum.png")

    # ── 3) Figure B: cumulative variance ──
    print(f"\n[Figure B] Cumulative variance")
    plot_cumulative_variance(all_sigmas, out_dir / "fig_svd_cumulative.png")

    # ── 4) Figure C: Top-k vs Bottom-k ──
    all_topk, all_bottomk, all_baseline = {}, {}, {}
    if not args.skip_fig_c:
        print(f"\n[Figure C] Top-k vs Bottom-k projection (k sweep)")
        for subset, d in cache.items():
            print(f"  [{subset}] α sweep × {len(ks)}개 k × 2 (top/bottom) ...")
            Vt_full = d["Vt_full"]
            D = Vt_full.shape[0]

            # baseline: 아무것도 projection 안 함 (k=0 effectively, after Z-Score)
            base_res = evaluate_projection_best_alpha(
                d["t_z"], d["i_z"], d["q_z"], d["modes"], d["q_gt"],
                np.zeros((0, d["t_z"].shape[1]), dtype="float32"), alphas,
            )
            all_baseline[subset] = base_res["NDCG@10"]
            print(f"    baseline (no projection): NDCG@10={base_res['NDCG@10']:.4f}")

            top_at, bot_at = {}, {}
            for k in ks:
                if k > D:
                    print(f"    k={k} skip (D={D})")
                    continue
                top_res = evaluate_projection_best_alpha(
                    d["t_z"], d["i_z"], d["q_z"], d["modes"], d["q_gt"],
                    Vt_full[:k], alphas,
                )
                bot_res = evaluate_projection_best_alpha(
                    d["t_z"], d["i_z"], d["q_z"], d["modes"], d["q_gt"],
                    Vt_full[-k:], alphas,
                )
                top_at[k] = top_res["NDCG@10"]
                bot_at[k] = bot_res["NDCG@10"]
                print(f"    k={k:>3d}: Top={top_res['NDCG@10']:.4f}  "
                      f"Bottom={bot_res['NDCG@10']:.4f}  "
                      f"Δ={top_res['NDCG@10']-bot_res['NDCG@10']:+.4f}")
            all_topk[subset] = top_at
            all_bottomk[subset] = bot_at

        # Shuffled paired-diff control
        shuf_path = args.shuffled_json or SHUF_JSON
        if args.shuffled_seeds > 0:
            print(f"\n[Shuffled control] {args.shuffled_seeds} seeds "
                  f"(Top-k/Bottom-k 와 동일 run)")
            run_shuffled_control(cache, ks, args.shuffled_seeds, shuf_path,
                                 alphas=alphas)
        shuf = load_shuffled_control(shuf_path)
        if shuf is None:
            print("  (shuffled_control.json 없음 — 2-curve 로 그림)")
        else:
            print(f"  shuffled control 로드: {len(shuf['per_subset'])} subsets, "
                  f"{shuf['n_seeds']} seeds")

        # (1) 절대 NDCG@10 — 패널별 y축 (main paper)
        plot_topk_vs_bottomk(all_topk, all_bottomk, all_baseline, ks,
                             FIG_ABS, shuffled=shuf, alpha_fixed=args.alpha)
        # (2) baseline 대비 Δ — y축 공유 (appendix / rebuttal 용)
        plot_topk_vs_bottomk(all_topk, all_bottomk, all_baseline, ks,
                             FIG_DELTA, delta=True, shuffled=shuf,
                             alpha_fixed=args.alpha)

    # ── 5) Raw 데이터 저장 ──
    save = {
        "k_fixed": K_FIXED,
        "sigmas": {s: sigs.tolist() for s, sigs in all_sigmas.items()},
    }
    if not args.skip_fig_c:
        save["baseline_ndcg10"] = all_baseline
        save["topk_ndcg10"] = all_topk
        save["bottomk_ndcg10"] = all_bottomk

    out_json = SPEC_JSON
    with open(out_json, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\n  raw 데이터 저장: {out_json}")


if __name__ == "__main__":
    main()
