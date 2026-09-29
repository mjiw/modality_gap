"""
make_spectrum_clean_and_table.py
==========================================================================
기존 svd_spectrum.json 을 활용해서:
  (1) k=16 마커 없는 깔끔한 spectrum figure 를 2×2 layout 으로 재생성
       → 출력: fig_svd_spectrum_clean_2x2.png
  (2) σ_k 값 표를 markdown / CSV / LaTeX 세 형식으로 출력
       → 출력: svd_spectrum_table.{md,csv,tex}

목적:
  - Subspace Analysis 섹션에서 k=16 정당화 figure 와 분리
  - 본문에 표로 σ 값을 함께 제시 (figure 보조)

실행:
  conda activate jiwoo
  python make_spectrum_clean_and_table.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent


# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
JSON_PATH = (_HERE.parent / "MixBench_results" / "k_sensitivity_v2"
             / "svd_spectrum.json")
OUT_DIR = _HERE.parent / "MixBench_results" / "k_sensitivity_v2"

DATASETS = ["MSCOCO", "VisualNews", "Google_WIT", "OVEN"]
# 표에 표시할 k 값들 (log₂ 간격)
KS_TABLE = [1, 2, 4, 8, 16, 32, 64, 128]
COL_SVD = "#D32F2F"


# ─────────────────────────────────────────────
# (1) Clean spectrum figure (k=16 마커 없음, 2×2)
# ─────────────────────────────────────────────
def disp_name(name):
    """데이터셋 내부 식별자 → 표시용 이름.
    내부 키(캐시 폴더명 cache_Google_WIT_L14/, JSON 키)는 그대로 두고
    figure 라벨에서만 하이픈 표기를 쓴다."""
    return {"Google_WIT": "Google-WIT"}.get(name, name)


def make_clean_spectrum(sigmas):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes_flat = axes.flatten()

    for ax, subset in zip(axes_flat, DATASETS):
        if subset not in sigmas:
            ax.axis("off")
            continue
        s = np.array(sigmas[subset])
        ks = np.arange(1, len(s) + 1)
        ax.plot(ks, s, color=COL_SVD, linewidth=1.8, zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("k", fontsize=15)
        ax.set_ylabel(r"$\sigma_k$ (log scale)", fontsize=15)
        ax.tick_params(axis="both", labelsize=12)
        ax.set_title(disp_name(subset), fontsize=13)
        ax.grid(True, alpha=0.3, which="both")
        # k=16 마커 / 별표 / legend 모두 제거됨

    fig.tight_layout()
    out_path = OUT_DIR / "fig_svd_spectrum_clean_2x2.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved figure: {out_path}")


# ─────────────────────────────────────────────
# (2) σ_k 표 — markdown / CSV / LaTeX
# ─────────────────────────────────────────────
def make_tables(sigmas):
    # 데이터 수집: subset → [σ_1, σ_2, σ_4, σ_8, σ_16, σ_32, σ_64, σ_128]
    rows = []
    for subset in DATASETS:
        if subset not in sigmas:
            continue
        s = sigmas[subset]
        row = {"subset": subset}
        for k in KS_TABLE:
            if k - 1 < len(s):
                row[f"σ_{k}"] = s[k - 1]
            else:
                row[f"σ_{k}"] = None
        rows.append(row)

    # ───── markdown ─────
    md_lines = []
    header = "| Dataset | " + " | ".join(f"$\\sigma_{{{k}}}$" for k in KS_TABLE) + " |"
    sep = "|---|" + "|".join("---:" for _ in KS_TABLE) + "|"
    md_lines.append(header)
    md_lines.append(sep)
    for row in rows:
        cells = [row["subset"]]
        for k in KS_TABLE:
            v = row[f"σ_{k}"]
            cells.append("—" if v is None else f"{v:.3f}")
        md_lines.append("| " + " | ".join(cells) + " |")
    md_table = "\n".join(md_lines)

    md_path = OUT_DIR / "svd_spectrum_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Singular value table (paired-difference SVD)\n\n")
        f.write(md_table + "\n")
    print(f"  saved table:  {md_path}")

    # ───── CSV ─────
    csv_lines = ["Dataset," + ",".join(f"sigma_{k}" for k in KS_TABLE)]
    for row in rows:
        cells = [row["subset"]]
        for k in KS_TABLE:
            v = row[f"σ_{k}"]
            cells.append("" if v is None else f"{v:.4f}")
        csv_lines.append(",".join(cells))
    csv_path = OUT_DIR / "svd_spectrum_table.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines) + "\n")
    print(f"  saved CSV:    {csv_path}")

    # ───── LaTeX ─────
    tex_lines = []
    tex_lines.append("\\begin{table}[ht]")
    tex_lines.append("\\centering")
    tex_lines.append("\\caption{Singular values $\\sigma_k$ of the paired modality "
                     "difference matrix across MixBench datasets.}")
    tex_lines.append("\\label{tab:svd_spectrum}")
    tex_lines.append("\\begin{tabular}{l" + "c" * len(KS_TABLE) + "}")
    tex_lines.append("\\toprule")
    tex_lines.append("Dataset & " +
                     " & ".join(f"$\\sigma_{{{k}}}$" for k in KS_TABLE) +
                     " \\\\")
    tex_lines.append("\\midrule")
    for row in rows:
        cells = [row["subset"].replace("_", "\\_")]
        for k in KS_TABLE:
            v = row[f"σ_{k}"]
            cells.append("—" if v is None else f"{v:.3f}")
        tex_lines.append(" & ".join(cells) + " \\\\")
    tex_lines.append("\\bottomrule")
    tex_lines.append("\\end{tabular}")
    tex_lines.append("\\end{table}")
    tex_table = "\n".join(tex_lines)

    tex_path = OUT_DIR / "svd_spectrum_table.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_table + "\n")
    print(f"  saved LaTeX:  {tex_path}")

    # ───── 콘솔 출력 ─────
    print("\n[markdown preview]")
    print(md_table)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    if not JSON_PATH.exists():
        print(f"ERROR: JSON not found: {JSON_PATH}")
        sys.exit(1)

    with open(JSON_PATH) as f:
        data = json.load(f)

    sigmas = data.get("sigmas", {})
    if not sigmas:
        print("ERROR: 'sigmas' key not found in JSON")
        sys.exit(1)

    print(f"Loaded sigmas for: {list(sigmas.keys())}")
    print(f"Output dir: {OUT_DIR}\n")

    print("[1] Clean spectrum figure (k=16 markers removed, 2x2)")
    make_clean_spectrum(sigmas)

    print("\n[2] σ_k tables (markdown / CSV / LaTeX)")
    make_tables(sigmas)

    print("\nDone.")


if __name__ == "__main__":
    main()
