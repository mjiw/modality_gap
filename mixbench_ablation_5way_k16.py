"""
MixBench2026 — 5-way Ablation with **k=16 fixed** (universal k)

mixbench_ablation_5way.py 와 동일하지만:
  • SVD only (방법 4) 와 Z-Score + SVD (방법 5) 의 k 를 **16 으로 고정**
  • k sweep 없음 → "per-dataset best k" 가 아닌 universal-k 결과 보고
  • Oracle k selection 비판 무력화용

논문 비교 가이드:
  • mixbench_ablation_5way.py  →  per-dataset best k (oracle)
  • mixbench_ablation_5way_k16.py  →  universal k=16 (deployment-ready)
  • 두 결과를 함께 보고 → "k tuning 없이도 대부분의 이득" 메시지

5가지 방법 (k=16 고정):
  1. CLIP             — Raw                     (Mean ✗  Variance ✗  Subspace ✗)
  2. GR-CLIP          — Mean shift only         (Mean ✓  Variance ✗  Subspace ✗)
  3. Z-Score only     — Mean + Variance         (Mean ✓  Variance ✓  Subspace ✗)
  4. SVD only         — Subspace k=16           (Mean ✗  Variance ✗  Subspace ✓)
  5. Z-Score + SVD    — All combined (Ours, k=16) (Mean ✓  Variance ✓  Subspace ✓)

실행:
  conda activate jiwoo
  python mixbench_ablation_5way_k16.py --subset MSCOCO
  python mixbench_ablation_5way_k16.py --all
  python mixbench_ablation_5way_k16.py --all --k 8   # k 값 변경 가능
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
# 고정된 k (universal)
# ─────────────────────────────────────────────
DEFAULT_K = 16


# ─────────────────────────────────────────────
# 새 ablation: SVD only (k 고정, z-score 없이 raw 임베딩 + subspace 제거)
# ─────────────────────────────────────────────
def run_svd_only_fixed_k(text_emb, image_emb, query_emb, modes, q_gt, alphas, k):
    """raw 임베딩(L2만)에 paired-diff SVD top-k 방향 제거 + α sweep.
    k 값은 고정 — 데이터셋별 튜닝 없음."""
    t_raw = _l2norm_np(text_emb).astype("float32")
    i_raw = _l2norm_np(image_emb).astype("float32")
    q_raw = _l2norm_np(query_emb).astype("float32")

    dirs_max = _compute_topk_directions(t_raw, i_raw, modes, k)
    if dirs_max is None:
        print(f"  Skip: 'both' pairs 부족 → SVD 추정 불가")
        return {}

    t_p = _l2norm_np(_project_out(t_raw, dirs_max)).astype("float32")
    i_p = _l2norm_np(_project_out(i_raw, dirs_max)).astype("float32")
    q_p = _l2norm_np(_project_out(q_raw, dirs_max)).astype("float32")

    out = {}
    for alpha in alphas:
        corpus = _build_from_precorrected(t_p, i_p, modes, alpha)
        scores = q_p @ corpus.T
        res = evaluate_ranking(scores, q_gt, modes=modes)
        out[f"svd_only_a{alpha:.1f}"] = res
    return out


# ─────────────────────────────────────────────
# 5-way 실행 (k 고정)
# ─────────────────────────────────────────────
def run_5way_ablation_fixed_k(subset, k, alphas=None):
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

    results = {}

    # ── 1. CLIP (Raw) ──
    print(f"\n[1/5] CLIP (Raw)...")
    for alpha in alphas:
        corpus = build_emb_corpus(text_emb, image_emb, modes, alpha)
        q = process_emb_query(query_emb)
        scores = q @ corpus.T
        results[f"raw_a{alpha:.1f}"] = evaluate_ranking(scores, q_gt, modes=modes)
    best = max([k_ for k_ in results if k_.startswith("raw_")],
               key=lambda k_: results[k_]["MRR"])
    print(f"  Best: {best}  MRR={results[best]['MRR']:.4f}  NDCG@10={results[best]['NDCG@10']:.4f}")

    # ── 2. GR-CLIP (Mean shift only) ──
    print(f"\n[2/5] GR-CLIP (Mean shift only)...")
    for alpha in alphas:
        corpus = build_emb_corpus(text_emb, image_emb, modes, alpha,
                                  "gr_clip", text_mean, image_mean)
        q = process_emb_query(query_emb, "gr_clip", query_mean)
        scores = q @ corpus.T
        results[f"gr_clip_a{alpha:.1f}"] = evaluate_ranking(scores, q_gt, modes=modes)
    best = max([k_ for k_ in results if k_.startswith("gr_clip_")],
               key=lambda k_: results[k_]["MRR"])
    print(f"  Best: {best}  MRR={results[best]['MRR']:.4f}  NDCG@10={results[best]['NDCG@10']:.4f}")

    # ── 3. Z-Score only (Mean + Variance) ──
    print(f"\n[3/5] Z-Score only...")
    for alpha in alphas:
        corpus = build_emb_corpus(text_emb, image_emb, modes, alpha,
                                  "global_zscore", text_mean, image_mean,
                                  text_std, image_std)
        q = process_emb_query(query_emb, "global_zscore", query_mean, query_std)
        scores = q @ corpus.T
        results[f"zscore_a{alpha:.1f}"] = evaluate_ranking(scores, q_gt, modes=modes)
    best = max([k_ for k_ in results if k_.startswith("zscore_")],
               key=lambda k_: results[k_]["MRR"])
    print(f"  Best: {best}  MRR={results[best]['MRR']:.4f}  NDCG@10={results[best]['NDCG@10']:.4f}")

    # ── 4. SVD only (Subspace only, k 고정) ──
    print(f"\n[4/5] SVD only (Subspace only, k={k} fixed)...")
    svd_only_res = run_svd_only_fixed_k(text_emb, image_emb, query_emb,
                                          modes, q_gt, alphas, k)
    results.update(svd_only_res)
    if svd_only_res:
        best = max([k_ for k_ in results if k_.startswith("svd_only_")],
                   key=lambda k_: results[k_]["MRR"])
        print(f"  Best: {best}  MRR={results[best]['MRR']:.4f}  NDCG@10={results[best]['NDCG@10']:.4f}")

    # ── 5. Z-Score + SVD (Ours, k 고정) ──
    print(f"\n[5/5] Z-Score + SVD (Ours, k={k} fixed)...")
    t_z = _l2norm_np((text_emb - text_mean) / text_std).astype("float32")
    i_z = _l2norm_np((image_emb - image_mean) / image_std).astype("float32")
    q_z = _l2norm_np((query_emb - query_mean) / query_std).astype("float32")
    dirs = _compute_topk_directions(t_z, i_z, modes, k)
    if dirs is None:
        print(f"  Skip: 'both' pairs 부족")
    else:
        t_p = _l2norm_np(_project_out(t_z, dirs)).astype("float32")
        i_p = _l2norm_np(_project_out(i_z, dirs)).astype("float32")
        q_p = _l2norm_np(_project_out(q_z, dirs)).astype("float32")
        for alpha in alphas:
            corpus = _build_from_precorrected(t_p, i_p, modes, alpha)
            scores = q_p @ corpus.T
            results[f"z_svd_a{alpha:.1f}"] = evaluate_ranking(scores, q_gt, modes=modes)
        best = max([k_ for k_ in results if k_.startswith("z_svd_")],
                   key=lambda k_: results[k_]["MRR"])
        print(f"  Best: {best}  MRR={results[best]['MRR']:.4f}  NDCG@10={results[best]['NDCG@10']:.4f}")

    return results


# ─────────────────────────────────────────────
# 5-way ablation 표 출력 (k 고정)
# ─────────────────────────────────────────────
def make_ablation_defs(k):
    return [
        ("1. CLIP (Raw)",                       "raw_",      "✗", "✗", "✗"),
        ("2. GR-CLIP",                          "gr_clip_",  "✓", "✗", "✗"),
        ("3. Z-Score only",                     "zscore_",   "✓", "✓", "✗"),
        (f"4. SVD only (k={k})",                "svd_only_", "✗", "✗", "✓"),
        (f"5. Z-Score + SVD (Ours, k={k})",     "z_svd_",    "✓", "✓", "✓"),
    ]


def _best_for_prefix(results, prefix):
    keys = [k for k in results if k.startswith(prefix)]
    if not keys:
        return None, None
    best_key = max(keys, key=lambda k: results[k]["MRR"])
    hp_str = best_key[len(prefix):]
    return results[best_key], hp_str


def print_ablation_table(subset, results, k):
    ablation_defs = make_ablation_defs(k)
    print(f"\n{'=' * 110}")
    print(f"  [{subset}] 5-way Ablation Table  (k={k} fixed, universal)")
    print(f"{'=' * 110}")
    print(f"  {'Method':<35s}  {'Mean':>5s}  {'Var':>5s}  {'Sub':>5s}  "
          f"{'best α':>8s}  {'R@1':>8s}  {'R@10':>8s}  {'NDCG@10':>9s}")
    print("  " + "-" * 101)

    rows = []
    for label, prefix, mean_c, var_c, sub_c in ablation_defs:
        res, hp = _best_for_prefix(results, prefix)
        if res is None:
            print(f"  {label:<35s}  {mean_c:>5s}  {var_c:>5s}  {sub_c:>5s}  "
                  f"{'(skipped)':>8s}  {'—':>8s}  {'—':>8s}  {'—':>9s}")
            continue
        print(f"  {label:<35s}  {mean_c:>5s}  {var_c:>5s}  {sub_c:>5s}  "
              f"{hp:>8s}  {res['R@1']:>8.4f}  "
              f"{res['R@10']:>8.4f}  {res['NDCG@10']:>9.4f}")
        rows.append({
            "method": label, "mean": mean_c, "var": var_c, "subspace": sub_c,
            "k": k if sub_c == "✓" else None,
            "hp": hp, "MRR": res["MRR"], "R@1": res["R@1"],
            "R@5": res.get("R@5", 0.0), "R@10": res["R@10"],
            "R@20": res.get("R@20", 0.0),
            "NDCG@5": res.get("NDCG@5", 0.0),
            "NDCG@10": res["NDCG@10"],
            "NDCG@20": res.get("NDCG@20", 0.0),
        })
    print("  " + "-" * 101)

    if rows:
        baseline = next((r for r in rows if r["method"].startswith("1.")), None)
        if baseline and baseline["NDCG@10"] > 0:
            print(f"\n  NDCG@10 개선율 (vs CLIP baseline)")
            print("  " + "-" * 60)
            for r in rows[1:]:
                rel = (r["NDCG@10"] / baseline["NDCG@10"] - 1) * 100
                abs_diff = r["NDCG@10"] - baseline["NDCG@10"]
                print(f"  {r['method']:<35s}  Δ={abs_diff:>+7.4f}  ({rel:>+6.1f}%)")

    return rows


def print_cross_dataset_table(all_results, k):
    subsets = list(all_results.keys())
    ablation_defs = make_ablation_defs(k)
    print(f"\n{'=' * 120}")
    print(f"  Cross-Dataset 5-way Ablation — NDCG@10  (k={k} fixed)")
    print(f"{'=' * 120}")

    header = f"  {'Method':<35s}  {'Mean':>5s}  {'Var':>5s}  {'Sub':>5s}"
    for s in subsets:
        header += f"  {s:>12s}"
    header += f"  {'Avg':>10s}"
    print(header)
    print("  " + "-" * (35 + 6 + 6 + 6 + 14 * len(subsets) + 12))

    for label, prefix, mean_c, var_c, sub_c in ablation_defs:
        row = f"  {label:<35s}  {mean_c:>5s}  {var_c:>5s}  {sub_c:>5s}"
        vals = []
        for s in subsets:
            res, _ = _best_for_prefix(all_results[s], prefix)
            if res is None:
                row += f"  {'—':>12s}"
                continue
            v = res["NDCG@10"]
            vals.append(v)
            row += f"  {v:>12.4f}"
        if vals:
            row += f"  {np.mean(vals):>10.4f}"
        else:
            row += f"  {'—':>10s}"
        print(row)
    print("  " + "-" * (35 + 6 + 6 + 6 + 14 * len(subsets) + 12))

    # R@1
    print(f"\n  Cross-Dataset 5-way Ablation — R@1  (k={k} fixed)")
    print("  " + "-" * (35 + 6 + 6 + 6 + 14 * len(subsets) + 12))
    print(header)
    print("  " + "-" * (35 + 6 + 6 + 6 + 14 * len(subsets) + 12))
    for label, prefix, mean_c, var_c, sub_c in ablation_defs:
        row = f"  {label:<35s}  {mean_c:>5s}  {var_c:>5s}  {sub_c:>5s}"
        vals = []
        for s in subsets:
            res, _ = _best_for_prefix(all_results[s], prefix)
            if res is None:
                row += f"  {'—':>12s}"
                continue
            v = res["R@1"]
            vals.append(v)
            row += f"  {v:>12.4f}"
        if vals:
            row += f"  {np.mean(vals):>10.4f}"
        else:
            row += f"  {'—':>10s}"
        print(row)
    print("  " + "-" * (35 + 6 + 6 + 6 + 14 * len(subsets) + 12))


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["MSCOCO", "VisualNews", "Google_WIT", "OVEN"])
    ap.add_argument("--all", action="store_true", help="4개 데이터셋 전부 실행 (MSCOCO/VisualNews/Google_WIT/OVEN)")
    ap.add_argument("--k", type=int, default=DEFAULT_K,
                    help=f"고정 k 값 (default: {DEFAULT_K})")
    ap.add_argument("--alpha", type=float, default=None,
                    help="α 를 이 값으로 고정 (예: 0.3). 생략하면 0.0~1.0 "
                         "sweep 후 MRR 최대 α 선택(기존 동작). 고정 시 "
                         "출력 파일명에 _a0.3 접미사가 붙어 sweep 결과와 분리됨")
    args = ap.parse_args()

    if not args.all and not args.subset:
        ap.error("--subset 또는 --all 중 하나를 지정하세요")

    k = args.k
    if args.alpha is not None and not (0.0 <= args.alpha <= 1.0):
        ap.error("--alpha 는 0.0 ~ 1.0 범위여야 합니다")
    alphas = [args.alpha] if args.alpha is not None else ALPHAS
    sfx = "" if args.alpha is None else f"_a{args.alpha:g}"
    RESULT_DIR = _MIXBENCH_DIR.parent / "MixBench_results"
    RESULT_DIR.mkdir(exist_ok=True, parents=True)

    subsets = ["MSCOCO", "VisualNews", "Google_WIT", "OVEN"] if args.all else [args.subset]
    all_dataset_results = {}
    all_rows = {}

    _amsg = "α sweep (best by MRR)" if args.alpha is None else f"α={args.alpha:g} fixed"
    print(f"=== 5-way Ablation — k={k} fixed (universal), {_amsg} ===")

    for subset in subsets:
        print(f"\n{'#' * 80}")
        print(f"  데이터셋: {subset}  (k={k} fixed)")
        print(f"{'#' * 80}")

        results = run_5way_ablation_fixed_k(subset, k=k, alphas=alphas)
        all_dataset_results[subset] = results
        rows = print_ablation_table(subset, results, k)
        all_rows[subset] = rows

        out_dir = RESULT_DIR / subset
        out_dir.mkdir(exist_ok=True)
        save_results = {kk: {kkk: vv for kkk, vv in v.items() if isinstance(vv, (int, float))}
                        for kk, v in results.items()}
        out_path = out_dir / f"ablation_5way_k{k}{sfx}.json"
        with open(out_path, "w") as f:
            json.dump({"k": k, "alphas": alphas,
                       "raw": save_results, "table": rows}, f, indent=2)
        print(f"\n  저장: {out_path}")

    if len(all_dataset_results) > 1:
        print_cross_dataset_table(all_dataset_results, k)
        out_path = RESULT_DIR / f"ablation_5way_k{k}_cross_dataset{sfx}.json"
        with open(out_path, "w") as f:
            summary = {s: {kk: {kkk: vv for kkk, vv in v.items() if isinstance(vv, (int, float))}
                           for kk, v in r.items()}
                       for s, r in all_dataset_results.items()}
            json.dump({"k": k, "alphas": alphas,
                       "raw": summary, "tables": all_rows}, f, indent=2)
        print(f"\n  Cross-dataset 저장: {out_path}")


if __name__ == "__main__":
    main()
