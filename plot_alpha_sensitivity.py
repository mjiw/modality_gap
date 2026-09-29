"""
α-Sensitivity Plot — GR-CLIP Figure 3b 형식

기존 grid search 결과(JSON)를 재활용해서 method별 α-metric 곡선 plot.
새 실험 안 함 — 저장된 결과만 읽어 시각화.

비교 대상 (default):
  - CLIP (Raw)
  - GR-CLIP
  - Global Z-Score
  - Z + Subspace Top-K (제안 — 최적 k 자동 선택 또는 --k 지정)

실행 예시:
  conda activate jiwoo

  # MSCOCO mixbench (작은 코퍼스)
  python plot_alpha_sensitivity.py --subset MSCOCO

  # 3개 subset 한꺼번에 (1개 figure × 3 패널)
  python plot_alpha_sensitivity.py --all

  # WIT 100K 대규모 (BF 결과)
  python plot_alpha_sensitivity.py \\
    --results-file ../MixBench_results/WIT_large_100000/bf_6methods.json \\
    --title "WIT 100K (BF)"

  # MRR로 그리기 (기본은 NDCG@10)
  python plot_alpha_sensitivity.py --subset MSCOCO --metric MRR

  # Subspace k 고정
  python plot_alpha_sensitivity.py --subset MSCOCO --k 8
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# 방법별 스타일
METHOD_STYLES = {
    "raw":      {"label": "CLIP",                  "color": "#888888",
                 "marker": "o", "linestyle": "--"},
    "gr_clip":  {"label": "GR-CLIP",                "color": "#1976D2",
                 "marker": "s", "linestyle": "-"},
    "zscore":   {"label": "Global Z-Score",         "color": "#388E3C",
                 "marker": "^", "linestyle": "-"},
    "z_subK":   {"label": "Z + Subspace Top-K (Ours)", "color": "#D32F2F",
                 "marker": "D", "linestyle": "-"},
}

DEFAULT_METHODS = ["raw", "gr_clip", "z_subK"]


# ─────────────────────────────────────────────
# Result 파싱
# ─────────────────────────────────────────────
def extract_series(results, method, metric, fixed_k=None):
    """
    한 method의 (α, metric) 시리즈 반환. None이면 데이터 없음.

    method:
      - 'raw', 'gr_clip', 'zscore' → key 패턴: f"{method}_a{α}"
      - 'z_subK' → key 패턴: f"z_subK_a{α}_k{k}". fixed_k이면 그 k만,
                   None이면 평균 metric 가장 높은 k 자동 선택.
    """
    if method == "z_subK":
        # 모든 (α, k)에서 metric 모으기
        per_k = {}  # k -> {alpha: metric}
        for key, vals in results.items():
            m = re.match(r"z_subK_a([\d.]+)_k(\d+)$", key)
            if m and metric in vals:
                alpha = float(m.group(1))
                kk = int(m.group(2))
                per_k.setdefault(kk, {})[alpha] = float(vals[metric])

        if not per_k:
            return None, None

        # k 선택
        if fixed_k is not None and fixed_k in per_k:
            chosen_k = fixed_k
        else:
            # 자동: α 평균 metric 가장 높은 k
            chosen_k = max(per_k.keys(),
                           key=lambda k: np.mean(list(per_k[k].values())))

        sorted_pairs = sorted(per_k[chosen_k].items())
        return sorted_pairs, chosen_k

    # 일반 method
    entries = {}
    pattern = re.compile(rf"^{re.escape(method)}_a([\d.]+)$")
    for key, vals in results.items():
        m = pattern.match(key)
        if m and metric in vals:
            alpha = float(m.group(1))
            entries[alpha] = float(vals[metric])

    if not entries:
        return None, None
    return sorted(entries.items()), None


# ─────────────────────────────────────────────
# Plot 1개 axis
# ─────────────────────────────────────────────
def plot_one_panel(ax, results, metric, methods, fixed_k, title):
    for method in methods:
        if method not in METHOD_STYLES:
            print(f"  ⚠ unknown method {method}, skip")
            continue
        series, used_k = extract_series(results, method, metric,
                                         fixed_k=fixed_k)
        if series is None:
            print(f"  ⚠ method '{method}': '{metric}' 데이터 없음, skip")
            continue

        alphas = [a for a, _ in series]
        values = [v for _, v in series]
        style = METHOD_STYLES[method]
        label = style["label"]
        if method == "z_subK" and used_k is not None:
            label = label.replace("(Ours)", f"(k={used_k}, Ours)")

        ax.plot(alphas, values,
                label=label,
                color=style["color"], marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2, markersize=7)

        # peak 점에 mini-marker
        peak_idx = int(np.argmax(values))
        ax.scatter([alphas[peak_idx]], [values[peak_idx]],
                   color=style["color"], s=120, marker="*",
                   edgecolors="black", linewidth=0.8, zorder=5)

    ax.set_ylabel(metric, fontsize=12)
    if title:
        ax.set_title(title, fontsize=11)
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([round(0.1 * i, 1) for i in range(11)])


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def load_subset_results(subset):
    base_dir = Path(__file__).parent.parent / "MixBench_results" / subset
    # ext 결과 우선 (사용자 기준 파일이 mixbench_full_comparison_ext.py)
    candidates = [
        base_dir / "full_comparison_ext.json",
        base_dir / "full_comparison.json",
        base_dir / "full_comparison_hnsw.json",
    ]
    for fp in candidates:
        if fp.exists():
            with open(fp) as f:
                return json.load(f), fp
    raise FileNotFoundError(
        f"{subset}의 결과 JSON 못 찾음: {base_dir}/*.json"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["MSCOCO", "VisualNews", "Google_WIT", "OVEN"],
                    help="단일 데이터셋")
    ap.add_argument("--all", action="store_true",
                    help="4개 subset을 한 figure에 패널로 (MSCOCO/VisualNews/Google_WIT/OVEN)")
    ap.add_argument("--results-file", type=Path, default=None,
                    help="직접 JSON 경로 지정 (e.g. WIT_large_100000/bf_6methods.json)")
    ap.add_argument("--metric", default="NDCG@10",
                    help="측정 metric (NDCG@10, MRR, R@10 등). 기본 NDCG@10")
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                    help=f"plot할 method 키들 (기본 {DEFAULT_METHODS})")
    ap.add_argument("--k", type=int, default=None,
                    help="z_subK의 k 값 고정 (생략 시 자동 선택)")
    ap.add_argument("--title", default=None, help="figure 제목 override")
    ap.add_argument("--output", type=Path, default=None,
                    help="출력 PNG 경로 (생략 시 자동 명명)")
    ap.add_argument("--split-metrics", action="store_true",
                    help="--all 과 함께: metric 마다 1x4 (4 subset) figure 따로 저장")
    ap.add_argument("--metrics", nargs="+",
                    default=["MRR", "R@1", "R@5", "R@10", "NDCG@10", "NDCG@100"],
                    help="--split-metrics 시 그릴 metric 리스트")
    args = ap.parse_args()

    # 사용자가 'R1' / 'NDCG10' 같이 @ 없이 넘긴 metric 을 JSON 표준 키 'R@1' / 'NDCG@10' 로 정규화.
    # MRR 처럼 숫자 없는 metric 은 그대로.
    import re as _re
    def _norm_metric(m):
        if "@" in m:
            return m
        match = _re.match(r"^(R|NDCG|MAP)(\d+)$", m)
        if match:
            return f"{match.group(1)}@{match.group(2)}"
        return m
    args.metric = _norm_metric(args.metric)
    args.metrics = [_norm_metric(m) for m in args.metrics]

    if not (args.all or args.subset or args.results_file):
        ap.error("--all / --subset / --results-file 중 하나 필요")

    out_dir = Path(__file__).parent / "figures"
    out_dir.mkdir(exist_ok=True)

    # 단일 source
    if args.results_file or args.subset:
        if args.results_file:
            with open(args.results_file) as f:
                results = json.load(f)
            title = args.title or args.results_file.stem
            out_name = f"alpha_sensitivity_{args.results_file.stem}.png"
        else:
            results, fp = load_subset_results(args.subset)
            title = args.title or f"{args.subset}  ({args.metric} vs α)"
            out_name = f"alpha_sensitivity_{args.subset}.png"

        out_path = args.output or (out_dir / out_name)

        fig, ax = plt.subplots(figsize=(8, 5))
        plot_one_panel(ax, results, args.metric, args.methods, args.k, title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"\n저장: {out_path}")

    elif args.all:
        subsets = ["MSCOCO", "VisualNews", "Google_WIT", "OVEN"]

        # subset 별 결과 한 번만 로드 (split-metrics 모드에서 N번 재사용)
        subset_results = {}
        for s in subsets:
            try:
                subset_results[s], _ = load_subset_results(s)
            except FileNotFoundError as e:
                subset_results[s] = None
                print(f"  ⚠ {s}: {e}")

        if args.split_metrics:
            # metric 마다 1x4 figure (4 subset 가로) 따로 저장
            for metric in args.metrics:
                fig, axes = plt.subplots(1, 4, figsize=(26, 5))
                for j, subset in enumerate(subsets):
                    ax = axes[j]
                    results = subset_results[subset]
                    if results is None:
                        ax.set_title(f"{subset} (missing)", fontsize=11)
                        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                                transform=ax.transAxes, fontsize=10)
                        continue
                    plot_one_panel(ax, results, metric, args.methods,
                                   args.k, subset)
                # 파일명에 '@' 같은 특수문자 안전화
                safe = metric.replace("@", "")
                out_path = out_dir / f"alpha_sensitivity_{safe}.png"
                plt.tight_layout()
                plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
                plt.close()
                print(f"저장: {out_path}")
        else:
            # 기존 2x4 grid (NDCG@10 / R@1 두 metric 묶음 1장)
            metrics = ["NDCG@10", "R@1"]
            fig, axes = plt.subplots(2, 4, figsize=(26, 10))
            for j, subset in enumerate(subsets):
                results = subset_results[subset]
                if results is None:
                    for i in range(len(metrics)):
                        ax = axes[i, j]
                        if i == 0:
                            ax.set_title(f"{subset} (missing)", fontsize=11)
                        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                                transform=ax.transAxes, fontsize=10)
                    continue
                for i, metric in enumerate(metrics):
                    title = subset if i == 0 else None
                    plot_one_panel(axes[i, j], results, metric, args.methods,
                                   args.k, title)

            out_path = args.output or (out_dir / "alpha_sensitivity_all_2x4.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
            plt.close()
            print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
