"""
MixBench2026 — 논문용 핵심 비교 (총 5가지)
==================================================================
사용자 제안 방법(★ 9번)과 비교 대상 baselines.

비교 대상 5가지:
  1. Raw                          — baseline
  2. GR-CLIP                      — mean shift only
  3. Global Z-Score               — embedding-level z-score (분산 정규화 포함)
  9. Z + Subspace Top-K  ★ 제안   — paired diff SVD 상위 k개 방향 제거 (k ∈ {1,2,4,8,16})
  11. Z-Score + Adaptive          — embedding(3) + score-level adaptive 결합

데이터셋 (MixBench2026 subsets):
  - MSCOCO, VisualNews, Google_WIT, OVEN

실행:
  conda activate jiwoo
  python mixbench_full_comparison_ext.py --subset MSCOCO
  python mixbench_full_comparison_ext.py --subset OVEN
  python mixbench_full_comparison_ext.py --all
  python mixbench_full_comparison_ext.py --all --alpha 0.5   # 1~3, 9번만, α=0.5 고정
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────
# 기존 mixbench_full_comparison.py 재사용 (같은 디렉토리)
# ─────────────────────────────────────────────
_MIXBENCH_DIR = Path(__file__).resolve().parent
if str(_MIXBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_MIXBENCH_DIR))

from data_loader import (  # noqa: E402
    load_mixbench,
    evaluate_ranking,
    inspect_top5_modes,
    run_adaptive,
    build_emb_corpus,
    process_emb_query,
    ALPHAS, SEED,
)

# ─────────────────────────────────────────────
# Helper: 보정된 임베딩 + α → 코퍼스 (방법 9 Subspace Top-K 용)
# ─────────────────────────────────────────────
def _build_from_precorrected(t_corr, im_corr, modes, alpha, mixed_corr=None):
    """옵션 B: mixed_corr (already-projected mixed_emb) 가 주어지면
    'both' 항목 corpus 에 alpha-blend 대신 mixed_corr 사용."""
    text_mask = np.array([m == "text" for m in modes])
    image_mask = np.array([m == "image" for m in modes])
    both_mask = np.array([m == "both" for m in modes])
    corpus = np.zeros_like(t_corr)
    corpus[text_mask] = t_corr[text_mask]
    corpus[image_mask] = im_corr[image_mask]
    if mixed_corr is not None:
        corpus[both_mask] = mixed_corr[both_mask]
    else:
        corpus[both_mask] = alpha * t_corr[both_mask] + (1 - alpha) * im_corr[both_mask]
    corpus = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8)
    return corpus.astype("float32")


def _l2norm_np(x, eps=1e-8):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


# ─────────────────────────────────────────────
# Subspace projection helpers (방법 9: 제안 방법 핵심)
#   Liang et al. "Mind the Gap" — text/image 평균 차이 벡터 방향을 "modality axis"
#   로 간주하고, 임베딩에서 해당 성분을 투영으로 제거 → 순수 semantic 정보만 남김.
# ─────────────────────────────────────────────
def _compute_topk_directions(t, i, modes, k):
    """Top-k principal directions of paired modality differences (SVD).
    diff_i = image_i - text_i  (for 'both' items) → stack → SVD → top-k right-sv.
    반환: [k, D] orthonormal rows. 'both' 쌍이 부족하면 None."""
    both_mask = np.array([m == "both" for m in modes])
    if both_mask.sum() < k + 1:
        return None
    diffs = i[both_mask] - t[both_mask]
    diffs = diffs - diffs.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(diffs, full_matrices=False)
    return Vt[:k].astype("float32")


def _project_out(x, directions):
    """x: [N, D], directions: [k, D] or [D] (orthonormal)
    returns: x - (x · V^T) · V   — 해당 부분공간 성분 제거."""
    if directions.ndim == 1:
        directions = directions[None, :]
    coef = x @ directions.T
    return x - coef @ directions


# ─────────────────────────────────────────────
# key 에서 하이퍼파라미터 추출 (출력용)
# ─────────────────────────────────────────────
def parse_hyperparams(key: str) -> dict:
    hp = {}
    for part in key.split("_"):
        if len(part) < 2:
            continue
        prefix, rest = part[0], part[1:]
        try:
            value = float(rest)
            if prefix == "a":
                hp["alpha"] = value
            elif prefix == "b":
                hp["beta"] = value
            elif prefix == "l":
                hp["lambda"] = value
            elif prefix == "k":
                hp["k"] = int(value)
        except ValueError:
            continue
    return hp


def format_hp(hp: dict) -> str:
    parts = []
    if "alpha" in hp:
        parts.append(f"α={hp['alpha']:.1f}")
    if "beta" in hp:
        parts.append(f"β={hp['beta']:.1f}")
    if "lambda" in hp:
        parts.append(f"λ={hp['lambda']:.1f}")
    if "k" in hp:
        parts.append(f"k={hp['k']}")
    return ",".join(parts) if parts else "-"


# ─────────────────────────────────────────────
# 메인 실험 루프
# ─────────────────────────────────────────────
def run_all_methods(subset, inspect_n=0, alphas=None, skip_adaptive=False):
    """편의 wrapper — subset 이름 → load_mixbench → run_all_methods_on_data."""
    text_emb, image_emb, query_emb, modes, q_gt = load_mixbench(subset)
    return run_all_methods_on_data(text_emb, image_emb, query_emb, modes, q_gt,
                                    inspect_n=inspect_n, alphas=alphas,
                                    skip_adaptive=skip_adaptive)


def run_all_methods_on_data(text_emb, image_emb, query_emb, modes, q_gt,
                             inspect_n=0, alphas=None, skip_adaptive=False):
    """5가지 방법 실행 → results dict 반환 (이미 load·필터된 데이터로).
    cross_modal/multimodal/mixed 등 외부에서 필터링된 corpus·query 를 받아 처리.
    - alphas : α 리스트 (None이면 ALPHAS 사용)
    - skip_adaptive : True면 1~3, 9번만, 11번(adaptive score-level) 생략
    - inspect_n > 0 이면 1,2,3번의 best-α 에서 top-5 샘플 검사
    """
    if alphas is None:
        alphas = ALPHAS

    if len(query_emb) == 0 or len(text_emb) == 0:
        print(f"  비어있음 (queries={len(query_emb)}, corpus={len(text_emb)}) — 스킵")
        return {}

    text_mask = np.array([m in ("text", "both") for m in modes])
    image_mask = np.array([m in ("image", "both") for m in modes])

    # 기존 (raw/gr_clip/global_zscore) 용 통계량
    text_mean = text_emb[text_mask].mean(axis=0, keepdims=True)
    image_mean = image_emb[image_mask].mean(axis=0, keepdims=True)
    text_std = text_emb[text_mask].std(axis=0, keepdims=True) + 1e-8
    image_std = image_emb[image_mask].std(axis=0, keepdims=True) + 1e-8
    query_mean = query_emb.mean(axis=0, keepdims=True)
    query_std = query_emb.std(axis=0, keepdims=True) + 1e-8

    results = {}

    # ─────────────────────────────────────────
    # 1/5 Raw
    # ─────────────────────────────────────────
    print(f"\n[1/5] Raw...")
    for alpha in alphas:
        corpus = build_emb_corpus(text_emb, image_emb, modes, alpha)
        q = process_emb_query(query_emb)
        scores = q @ corpus.T
        res = evaluate_ranking(scores, q_gt, modes=modes)
        results[f"raw_a{alpha:.1f}"] = res
    best_raw = max([k for k in results if k.startswith("raw_")],
                   key=lambda k: results[k]["MRR"])
    print(f"  Best: {best_raw}  MRR={results[best_raw]['MRR']:.4f}")
    if inspect_n > 0:
        ba = float(best_raw.split("_a")[1])
        corpus = build_emb_corpus(text_emb, image_emb, modes, ba)
        q = process_emb_query(query_emb)
        inspect_top5_modes(q @ corpus.T, q_gt, modes, f"1. Raw (α={ba})", inspect_n)

    # ─────────────────────────────────────────
    # 2/5 GR-CLIP
    # ─────────────────────────────────────────
    print(f"\n[2/5] GR-CLIP...")
    for alpha in alphas:
        corpus = build_emb_corpus(text_emb, image_emb, modes, alpha,
                                  "gr_clip", text_mean, image_mean)
        q = process_emb_query(query_emb, "gr_clip", query_mean)
        scores = q @ corpus.T
        res = evaluate_ranking(scores, q_gt, modes=modes)
        results[f"gr_clip_a{alpha:.1f}"] = res
    best_gr = max([k for k in results if k.startswith("gr_clip_")],
                  key=lambda k: results[k]["MRR"])
    print(f"  Best: {best_gr}  MRR={results[best_gr]['MRR']:.4f}")
    if inspect_n > 0:
        ba = float(best_gr.split("_a")[1])
        corpus = build_emb_corpus(text_emb, image_emb, modes, ba,
                                  "gr_clip", text_mean, image_mean)
        q = process_emb_query(query_emb, "gr_clip", query_mean)
        inspect_top5_modes(q @ corpus.T, q_gt, modes, f"2. GR-CLIP (α={ba})", inspect_n)

    # ─────────────────────────────────────────
    # 3/5 Global Z-Score
    # ─────────────────────────────────────────
    print(f"\n[3/5] Global Z-Score...")
    for alpha in alphas:
        corpus = build_emb_corpus(text_emb, image_emb, modes, alpha,
                                  "global_zscore", text_mean, image_mean,
                                  text_std, image_std)
        q = process_emb_query(query_emb, "global_zscore", query_mean, query_std)
        scores = q @ corpus.T
        res = evaluate_ranking(scores, q_gt, modes=modes)
        results[f"zscore_a{alpha:.1f}"] = res
    best_zs = max([k for k in results if k.startswith("zscore_")],
                  key=lambda k: results[k]["MRR"])
    print(f"  Best: {best_zs}  MRR={results[best_zs]['MRR']:.4f}")
    if inspect_n > 0:
        ba = float(best_zs.split("_a")[1])
        corpus = build_emb_corpus(text_emb, image_emb, modes, ba,
                                  "global_zscore", text_mean, image_mean,
                                  text_std, image_std)
        q = process_emb_query(query_emb, "global_zscore", query_mean, query_std)
        inspect_top5_modes(q @ corpus.T, q_gt, modes,
                           f"3. Global Z-Score (α={ba})", inspect_n)

    # Z-normalized + L2-normalized 임베딩 (방법 9, 11 공용)
    t_z = (text_emb - text_mean) / text_std
    i_z = (image_emb - image_mean) / image_std
    q_z = (query_emb - query_mean) / query_std
    t_z = _l2norm_np(t_z).astype("float32")
    i_z = _l2norm_np(i_z).astype("float32")
    q_z = _l2norm_np(q_z).astype("float32")

    # ─────────────────────────────────────────
    # 9/11 Z-Score + Subspace Projection (SVD, k=16 universal)  ★ 제안 방법
    #   'both' paired (image_i - text_i) 차이 벡터들을 SVD → 상위 k 주성분 제거.
    #   k=16 으로 고정 (universal — 데이터셋별 oracle k 선택 회피).
    #   k sensitivity 는 mixbench_k_sensitivity.py 에서 별도 측정.
    # ─────────────────────────────────────────
    K_FIXED = 16
    print(f"\n[4/5] Z + Subspace Projection (SVD, k={K_FIXED} fixed)... ★ 제안 방법 (9번)")
    dirs_max = _compute_topk_directions(t_z, i_z, modes, K_FIXED)
    if dirs_max is None:
        print(f"  Skip: 'both' pairs 부족 → top-k 추정 불가")
    else:
        t_k = _l2norm_np(_project_out(t_z, dirs_max)).astype("float32")
        i_k = _l2norm_np(_project_out(i_z, dirs_max)).astype("float32")
        q_k = _l2norm_np(_project_out(q_z, dirs_max)).astype("float32")
        for alpha in alphas:
            corpus = _build_from_precorrected(t_k, i_k, modes, alpha)
            scores = q_k @ corpus.T
            res = evaluate_ranking(scores, q_gt, modes=modes)
            results[f"z_subK_a{alpha:.1f}_k{K_FIXED}"] = res
        best_sk = max([k for k in results if k.startswith("z_subK_")],
                      key=lambda k: results[k]["MRR"])
        print(f"  Best: {best_sk}  MRR={results[best_sk]['MRR']:.4f}")

    if skip_adaptive:
        print(f"\n[--alpha 지정] 5번(Adaptive score-level) 생략")
        return results

    # ─────────────────────────────────────────
    # 5/5 Z-Score + Adaptive (결합, 11번)
    # ─────────────────────────────────────────
    print(f"\n[5/5] Z-Score + Adaptive (결합, 11번)...")
    res = run_adaptive(t_z, i_z, q_z, modes, q_gt,
                       label="11. Z-Score+Adaptive", inspect_n=inspect_n)
    results["zscore+adaptive"] = res
    print(f"  MRR={res['MRR']:.4f}  avg_α={res['avg_alpha']:.2f}")

    return results


# ─────────────────────────────────────────────
# 출력
# ─────────────────────────────────────────────
METHOD_CONFIGS = [
    ("raw_",        "1. Raw"),
    ("gr_clip_",    "2. GR-CLIP"),
    ("zscore_",     "3. Global Z-Score"),
    ("z_subK_",     "9. Z + Subspace Top-K (제안)"),
]


def _extract_method_bests(results):
    method_bests = {}
    for prefix, label in METHOD_CONFIGS:
        keys = [k for k in results if k.startswith(prefix)]
        if keys:
            best_key = max(keys, key=lambda k: results[k]["MRR"])
            method_bests[label] = (results[best_key], parse_hyperparams(best_key))
    if "zscore+adaptive" in results:
        method_bests["11. Z-Score+Adaptive"] = (results["zscore+adaptive"], None)
    return method_bests


def print_results(subset, results):
    print(f"\n{'='*125}")
    print(f"  [{subset}] 전체 방법론 비교 — 12가지 (ViT-L/14)")
    print(f"{'='*125}")

    method_bests = _extract_method_bests(results)

    print(f"  {'Method':<22s}  {'hp':>14s}  {'MRR':>7s}  {'R@1':>7s}  {'R@5':>7s}  "
          f"{'R@10':>7s}  {'R@20':>7s}  {'NDCG@5':>7s}  {'NDCG@10':>7s}  {'NDCG@20':>7s}  "
          f"{'NDCG@100':>8s}")
    print("  " + "-" * 128)
    for label, (res, hp) in method_bests.items():
        hp_str = "auto" if hp is None else format_hp(hp)
        print(f"  {label:<22s}  {hp_str:>14s}  {res['MRR']:>7.4f}  {res['R@1']:>7.4f}  "
              f"{res.get('R@5',0):>7.4f}  {res['R@10']:>7.4f}  {res['R@20']:>7.4f}  "
              f"{res.get('NDCG@5',0):>7.4f}  {res.get('NDCG@10',0):>7.4f}  "
              f"{res.get('NDCG@20',0):>7.4f}  {res.get('NDCG@100',0):>8.4f}")
    print("  " + "-" * 128)

    # Top-5 modality 분포
    print(f"\n  === Top-5 결과의 Modality 분포 (%) ===")
    print(f"  {'Method':<22s}  {'hp':>14s}  {'text':>14s}  {'image':>14s}  {'both':>14s}")
    print("  " + "-" * 90)
    for label, (res, hp) in method_bests.items():
        hp_str = "auto" if hp is None else format_hp(hp)
        if "top5_text_ratio" in res:
            print(f"  {label:<22s}  {hp_str:>14s}  "
                  f"{res['top5_text_ratio']*100:>5.1f}% ({int(res['top5_text_count']):>5d})  "
                  f"{res['top5_image_ratio']*100:>5.1f}% ({int(res['top5_image_count']):>5d})  "
                  f"{res['top5_both_ratio']*100:>5.1f}% ({int(res['top5_both_count']):>5d})")
        else:
            print(f"  {label:<22s}  {hp_str:>14s}  {'N/A':>14s}  {'N/A':>14s}  {'N/A':>14s}")
    print("  " + "-" * 90)

    # 개선폭 요약
    if "1. Raw" in method_bests:
        raw_mrr = method_bests["1. Raw"][0]["MRR"]
        best_label = max(method_bests, key=lambda k: method_bests[k][0]["MRR"])
        best_mrr = method_bests[best_label][0]["MRR"]
        delta = (best_mrr / raw_mrr - 1) * 100 if raw_mrr > 0 else 0
        print(f"\n  Best method: {best_label}  MRR={best_mrr:.4f}  "
              f"(vs Raw +{best_mrr - raw_mrr:.4f}, {delta:+.1f}%)")

    # 제안 방법(9, 11) vs baseline (1, 2, 3)
    if "3. Global Z-Score" in method_bests:
        z_mrr = method_bests["3. Global Z-Score"][0]["MRR"]
        print(f"\n  vs 3. Global Z-Score (MRR={z_mrr:.4f}):")
        for label in ["9. Z + Subspace Top-K (제안)", "11. Z-Score+Adaptive"]:
            if label in method_bests:
                m = method_bests[label][0]["MRR"]
                d = (m / z_mrr - 1) * 100 if z_mrr > 0 else 0
                arrow = "↑" if m > z_mrr else ("↓" if m < z_mrr else "=")
                print(f"    {arrow} {label:<28s}  MRR={m:.4f}  ({d:+.1f}%)")


def print_cross_dataset_table(all_dataset_results):
    subsets = list(all_dataset_results.keys())
    all_methods = [label for _, label in METHOD_CONFIGS]
    has_adaptive = any("zscore+adaptive" in r for r in all_dataset_results.values())
    if has_adaptive:
        all_methods = all_methods + ["11. Z-Score+Adaptive"]

    dataset_bests = {s: _extract_method_bests(r) for s, r in all_dataset_results.items()}

    for metric in ["MRR", "R@1", "R@5", "R@10", "NDCG@10", "NDCG@100"]:
        print(f"\n{'='*100}")
        print(f"  전체 데이터셋 × 방법론 {metric} 비교 (ViT-L/14)")
        print(f"{'='*100}")
        hdr = f"  {'Method':<22s}"
        for s in subsets:
            hdr += f"  {s:>15s}"
        print(hdr)
        print("  " + "-" * (22 + 17 * len(subsets)))
        for m in all_methods:
            row = f"  {m:<22s}"
            for s in subsets:
                v = dataset_bests[s].get(m, (None, None))
                val = v[0][metric] if v[0] and metric in v[0] else None
                row += f"  {val:>15.4f}" if val is not None else f"  {'N/A':>15s}"
            print(row)
        print("  " + "-" * (22 + 17 * len(subsets)))


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["MSCOCO", "VisualNews", "Google_WIT", "OVEN"])
    ap.add_argument("--all", action="store_true", help="3개 데이터셋 전부 실행")
    ap.add_argument("--inspect", type=int, default=0,
                    help="N개 샘플 쿼리의 top-5 modality 분포 검사 (1/2/3 적용, 0=비활성)")
    ap.add_argument("--alpha", type=float, default=None,
                    help="특정 α로 고정 (1~3, 9번만 실행, α sweep 및 11번 생략)")
    args = ap.parse_args()

    if not args.all and not args.subset:
        ap.error("--subset 또는 --all 중 하나를 지정하세요")
    if args.alpha is not None and not (0.0 <= args.alpha <= 1.0):
        ap.error("--alpha는 0.0~1.0 범위여야 합니다")

    alphas = [args.alpha] if args.alpha is not None else ALPHAS
    skip_adaptive = args.alpha is not None

    np.random.seed(SEED)

    RESULT_DIR = _MIXBENCH_DIR.parent / "MixBench_results"
    RESULT_DIR.mkdir(exist_ok=True, parents=True)

    subsets = ["MSCOCO", "VisualNews", "Google_WIT", "OVEN"] if args.all else [args.subset]
    all_dataset_results = {}

    for subset in subsets:
        print(f"\n{'#'*80}")
        print(f"  데이터셋: {subset}")
        print(f"{'#'*80}")

        results = run_all_methods(subset, inspect_n=args.inspect,
                                  alphas=alphas, skip_adaptive=skip_adaptive)
        all_dataset_results[subset] = results

        print_results(subset, results)

        # 저장 (기존 JSON 로직 그대로 유지, 파일명만 _ext)
        save_results = {}
        for k, v in results.items():
            save_results[k] = {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float))}
        out_dir = RESULT_DIR / subset
        out_dir.mkdir(exist_ok=True)
        out_name = (f"full_comparison_ext_a{args.alpha:.1f}.json"
                    if args.alpha is not None else "full_comparison_ext.json")
        with open(out_dir / out_name, "w") as f:
            json.dump(save_results, f, indent=2)
        print(f"\n저장: {out_dir / out_name}")

    if len(all_dataset_results) > 1:
        print_cross_dataset_table(all_dataset_results)

        with open(RESULT_DIR / "cross_dataset_comparison_ext.json", "w") as f:
            summary = {}
            for subset, results in all_dataset_results.items():
                summary[subset] = {}
                for k, v in results.items():
                    summary[subset][k] = {kk: vv for kk, vv in v.items()
                                          if isinstance(vv, (int, float))}
            json.dump(summary, f, indent=2)
        print(f"\n전체 저장: {RESULT_DIR / 'cross_dataset_comparison_ext.json'}")


if __name__ == "__main__":
    main()
