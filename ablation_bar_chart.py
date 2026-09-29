"""
ablation_bar_chart.py
================================================================
5-way component ablation 결과를 SMORE 스타일 막대그래프로 시각화.

입력: MixBench_results/ablation_5way_k16_cross_dataset.json
      (mixbench_ablation_5way_k16.py --all 로 생성)

출력: MixBench_results/fig_ablation_5way_bar.png
      좌측 = NDCG@10, 우측 = R@1
      각 panel: dataset 그룹 × 5개 method bar

스타일 (SMORE Figure 3 와 유사):
  - 2 horizontal panel (metric 2종)
  - pastel 색상 (Ours = pink highlight)
  - x축: dataset 이름 (Bold)
  - y축: metric 값

실행:
  conda activate tablerag
  python ablation_bar_chart.py
  python ablation_bar_chart.py --metrics NDCG@10        # 단일 metric panel
  python ablation_bar_chart.py --metrics NDCG@10,R@1,MRR  # 3 panel
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

# UTF-8 stdout (Windows cp949 호환)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


_HERE = Path(__file__).resolve().parent

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
# v3 = 최신 재실행 결과 (NDCG@10 / R@1). 이전 run 들은
#   ablation_5way_k16_cross_dataset.json    (원본, MRR·R@5·R@10 포함)
#   ablation_5way_k16_cross_dataset_v2.json
# 에 그대로 보존 — --json 으로 언제든 지정 가능.
DEFAULT_JSON = _HERE.parent / "MixBench_results" / "ablation_5way_k16_cross_dataset_v3.json"
DATASETS_ORDER = ["MSCOCO", "VisualNews", "Google_WIT", "OVEN"]
METRICS = ["NDCG@10", "R@1"]

# 5-way 방법 — (표시 라벨, JSON 의 "method" 필드 매칭 키워드)
# 표시 라벨은 figure legend / 출력 표 에 쓰임. 매칭 키워드는 JSON "method" 필드 substring 검색용.
METHOD_DEFS = [
    ("CLIP",                                 "CLIP (Raw)"),   # display, JSON match
    ("GR-CLIP",                              "GR-CLIP"),
    ("z-score only",                         "Z-Score only"),
    ("Directional Component Removal only",   "SVD only"),
    ("STADIRA (Ours)",                       "Z-Score + SVD"),
]

# 색상(hue)은 유지하고 채도·명도만 ViCLIP/CLAP ratio figure(tab10) 수준으로 맞춤.
# teal / gold / red 는 이미 tab10 대역 안이라 그대로 두고, 혼자 바래 보이던
# 파랑(S=37.7%)과 살짝 밝던 회색만 보정했다.
COLORS = [
    "#BDBDBD",  # CLIP
    "#8CB9E8",  # GR-CLIP
    "#81DA81",  # Z-score only — 청록(H 168.6) → 초록(H 120), 파랑과 90.7° 간격
    "#F1D17A",  # Directional Component Removal only
    "#E56161",  # STADIRA (Ours) — H 354→0(정빨강), L 75.3→64
]


# ─────────────────────────────────────────────
# JSON 읽기 + 메트릭 추출
# ─────────────────────────────────────────────
def disp_name(name):
    """데이터셋 내부 식별자 → 표시용 이름.
    내부 키(캐시 폴더명 cache_Google_WIT_L14/, JSON 키)는 그대로 두고
    figure 라벨에서만 하이픈 표기를 쓴다."""
    return {"Google_WIT": "Google-WIT"}.get(name, name)


def load_results(json_path: Path) -> dict:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def get_metric(tables: dict, dataset: str, method_keyword: str, metric: str) -> float | None:
    """tables[dataset] 에서 method 이름에 method_keyword 포함된 row 의 metric 반환."""
    rows = tables.get(dataset, [])
    for r in rows:
        if method_keyword.lower() in r.get("method", "").lower():
            return r.get(metric, None)
    return None


# ─────────────────────────────────────────────
# 막대그래프 (SMORE 스타일)
# ─────────────────────────────────────────────
def plot_bar(results: dict, metrics: list, datasets: list, out_path: Path):
    tables = results.get("tables", {})
    n_metrics = len(metrics)
    n_methods = len(METHOD_DEFS)
    n_datasets = len(datasets)

    # 한 dataset 그룹 안의 5개 bar 폭 (group 전체 폭 = 0.75)
    bar_width = 0.75 / n_methods

    fig, axes = plt.subplots(1, n_metrics, figsize=(6.5 * n_metrics, 5.5))
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(n_datasets)

    for ax, metric in zip(axes, metrics):
        # 각 method 별 그룹된 bar
        all_vals_panel = []
        for i, (label, keyword) in enumerate(METHOD_DEFS):
            values = []
            for ds in datasets:
                v = get_metric(tables, ds, keyword, metric)
                values.append(v if v is not None else 0.0)
                if v is not None:
                    all_vals_panel.append(v)
            offset = (i - (n_methods - 1) / 2) * bar_width
            ax.bar(x + offset, values, bar_width,
                   label=label, color=COLORS[i],
                   edgecolor="white", linewidth=0.6)

        # 축 / 라벨 (범례는 figure 하단 중앙에 한 번만 — 아래 fig.legend)
        ax.set_xticks(x)
        ax.set_xticklabels([disp_name(d) for d in datasets], fontsize=15)
        ax.set_ylabel(metric, fontsize=15)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)

        # y 축 범위 (SMORE 처럼 살짝 아래 잘라 차이 강조)
        if all_vals_panel:
            y_min = max(0.0, min(all_vals_panel) * 0.88)
            y_max = max(all_vals_panel) * 1.06
            ax.set_ylim(y_min, y_max)

    # ── Figure 하단 중앙에 범례 1회 (SMORE / CVPR 관행) ──
    # 첫 axis 의 handles 를 재사용해 5개 method 를 가로로 배치.
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_,
               loc="lower center",
               bbox_to_anchor=(0.5, -0.02),
               ncol=len(METHOD_DEFS),
               fontsize=13,
               frameon=True, framealpha=0.95)

    # suptitle 제거 — paper LaTeX caption 으로 별도 작성하므로 figure 내부엔 불필요
    # 범례 공간 확보를 위해 bottom margin 을 조정 (tight_layout 대신 subplots_adjust)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out_path}")


# ─────────────────────────────────────────────
# Console 출력 (sanity check)
# ─────────────────────────────────────────────
def print_summary(results: dict, metrics: list, datasets: list):
    tables = results.get("tables", {})
    print(f"\nLoaded k={results.get('k')} ablation results")
    print(f"Datasets: {datasets}")
    print(f"Metrics: {metrics}\n")
    for metric in metrics:
        print(f"=== {metric} ===")
        print(f"  {'Method':<22s}  " + "  ".join(f"{d:>12s}" for d in datasets) +
              f"  {'Avg':>10s}")
        print("  " + "─" * (22 + 14 * len(datasets) + 12))
        for label, keyword in METHOD_DEFS:
            vals = [get_metric(tables, ds, keyword, metric) for ds in datasets]
            row = f"  {label:<22s}"
            for v in vals:
                row += f"  {v:>12.4f}" if v is not None else f"  {'—':>12s}"
            vals_clean = [v for v in vals if v is not None]
            avg = sum(vals_clean) / len(vals_clean) if vals_clean else 0.0
            row += f"  {avg:>10.4f}"
            print(row)
        print()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(DEFAULT_JSON),
                    help="ablation_5way_k16_cross_dataset.json 경로")
    ap.add_argument("--metrics", default=",".join(METRICS),
                    help="comma-separated metrics (예: NDCG@10,R@1)")
    ap.add_argument("--subsets", default=",".join(DATASETS_ORDER),
                    help="comma-separated subset 이름")
    ap.add_argument("--out", default=None,
                    help="출력 PNG 경로 (default: 같은 폴더 fig_ablation_5way_bar.png)")
    args = ap.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        print(f"[ERROR] JSON 없음: {json_path}")
        print(f"먼저 'python mixbench_ablation_5way_k16.py --all' 실행 필요")
        sys.exit(1)

    results = load_results(json_path)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    datasets = [d.strip() for d in args.subsets.split(",") if d.strip()]

    out_path = Path(args.out) if args.out else json_path.parent / "fig_ablation_5way_bar.png"

    # sanity check console 출력
    print_summary(results, metrics, datasets)

    # plot
    plot_bar(results, metrics, datasets, out_path)


if __name__ == "__main__":
    main()
