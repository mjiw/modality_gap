"""
MixBench2026 — CLIP Backbone 비교 실험 (GR-CLIP 논문 Table 1 재현)
==================================================================
여러 CLIP backbone (및 VLM2Vec) 에 대해:
  - Baseline (Raw)
  - GR-CLIP (mean shift, 기존 방법)
  - Z-Score (embedding-level 분산 정규화)
  - 제안 방법: Z + Subspace Top-K (k=16)

출력 형식 (GR-CLIP 논문 Table 1 그대로):
  각 셀 = CELL_METRICS 순서 (기본 "NDCG@10 / R@10 / R@1")
  행 = backbone × method
  열 = 4개 dataset (Google_WIT, MSCOCO, OVEN, VisualNews)

실행:
  conda activate jiwoo
  python mixbench_backbone_comparison.py --all                # 5개 CLIP backbone × 4 dataset
  python mixbench_backbone_comparison.py --backbones B16 L14  # 일부만
  python mixbench_backbone_comparison.py --include-vlm2vec    # VLM2Vec 추가
  python mixbench_backbone_comparison.py --subset MSCOCO --backbones L14  # 빠른 확인
"""

from __future__ import annotations

import os
# Windows + conda OpenMP 중복 충돌 우회 (numpy/torch/mkl 사이 libiomp5md.dll 충돌).
# CLIP 추론 워크로드에서는 실질적으로 안전. import 전에 설정해야 효력 있음.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_MIXBENCH_DIR = Path(__file__).resolve().parent
if str(_MIXBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_MIXBENCH_DIR))

from data_loader import (  # noqa: E402
    load_mixbench,
    evaluate_ranking,
    build_emb_corpus,
    process_emb_query,
    ALPHAS, SEED, BACKBONES,
)
from mixbench_full_comparison_ext import (  # noqa: E402
    _compute_topk_directions,
    _project_out,
    _build_from_precorrected,
    _l2norm_np,
)

# Subspace projection k는 메인 실험과 동일하게 고정
K_FIXED = 16

# 표 출력에 사용할 데이터셋 순서 (GR-CLIP paper Table 1 순서)
DATASETS = ["Google_WIT", "MSCOCO", "OVEN", "VisualNews"]

# 핵심 메트릭 — 셀당 "NDCG@10 / R@10 / R@1"
CELL_METRICS = ["NDCG@10", "R@10", "R@1"]

# 비교할 방법 (key prefix, 표시 라벨)
METHOD_DEFS = [
    ("raw",       "Raw"),
    ("gr_clip",   "GR-CLIP"),
    ("zscore",    "Z-Score"),
    ("z_subK",    "Ours (Z+Sub)"),
]


def run_methods_one(text_emb, image_emb, query_emb, modes, q_gt, alphas, mixed_emb=None):
    """한 (backbone, dataset) 조합에 대해 4가지 방법 실행 → results dict.

    옵션 B: mixed_emb 가 제공되면 (VLM2Vec) 'both' 항목 corpus 에 mixed_emb 사용.
    paired (text_emb, image_emb) 는 modality 방향 (V) 추출에만 사용.
    """
    if len(query_emb) == 0 or len(text_emb) == 0:
        return {}

    text_mask = np.array([m in ("text", "both") for m in modes])
    image_mask = np.array([m in ("image", "both") for m in modes])
    both_mask = np.array([m == "both" for m in modes])

    text_mean = text_emb[text_mask].mean(axis=0, keepdims=True)
    image_mean = image_emb[image_mask].mean(axis=0, keepdims=True)
    text_std = text_emb[text_mask].std(axis=0, keepdims=True) + 1e-8
    image_std = image_emb[image_mask].std(axis=0, keepdims=True) + 1e-8
    query_mean = query_emb.mean(axis=0, keepdims=True)
    query_std = query_emb.std(axis=0, keepdims=True) + 1e-8

    # 옵션 B: mixed_emb 통계 ('both' 항목에서만 산출)
    has_mixed = mixed_emb is not None and both_mask.sum() > 0
    if has_mixed:
        mixed_mean = mixed_emb[both_mask].mean(axis=0, keepdims=True)
        mixed_std = mixed_emb[both_mask].std(axis=0, keepdims=True) + 1e-8
    else:
        mixed_mean = None
        mixed_std = None

    results = {}

    # 1. Raw
    for alpha in alphas:
        corpus = build_emb_corpus(text_emb, image_emb, modes, alpha,
                                  mixed_emb=mixed_emb)
        q = process_emb_query(query_emb)
        results[f"raw_a{alpha:.1f}"] = evaluate_ranking(q @ corpus.T, q_gt, modes=modes)

    # 2. GR-CLIP
    for alpha in alphas:
        corpus = build_emb_corpus(text_emb, image_emb, modes, alpha,
                                  "gr_clip", text_mean, image_mean,
                                  mixed_emb=mixed_emb, mixed_mean=mixed_mean)
        q = process_emb_query(query_emb, "gr_clip", query_mean)
        results[f"gr_clip_a{alpha:.1f}"] = evaluate_ranking(q @ corpus.T, q_gt, modes=modes)

    # 3. Global Z-Score
    for alpha in alphas:
        corpus = build_emb_corpus(text_emb, image_emb, modes, alpha,
                                  "global_zscore", text_mean, image_mean,
                                  text_std, image_std,
                                  mixed_emb=mixed_emb, mixed_mean=mixed_mean, mixed_std=mixed_std)
        q = process_emb_query(query_emb, "global_zscore", query_mean, query_std)
        results[f"zscore_a{alpha:.1f}"] = evaluate_ranking(q @ corpus.T, q_gt, modes=modes)

    # 4. 제안 — Z + Subspace Top-K (k=16)
    t_z = _l2norm_np((text_emb - text_mean) / text_std).astype("float32")
    i_z = _l2norm_np((image_emb - image_mean) / image_std).astype("float32")
    q_z = _l2norm_np((query_emb - query_mean) / query_std).astype("float32")
    # 옵션 B: mixed_emb 도 같은 Z-score 후 V projection
    if has_mixed:
        mx_z = _l2norm_np((mixed_emb - mixed_mean) / mixed_std).astype("float32")
    else:
        mx_z = None

    dirs = _compute_topk_directions(t_z, i_z, modes, K_FIXED)
    if dirs is not None:
        t_k = _l2norm_np(_project_out(t_z, dirs)).astype("float32")
        i_k = _l2norm_np(_project_out(i_z, dirs)).astype("float32")
        q_k = _l2norm_np(_project_out(q_z, dirs)).astype("float32")
        if mx_z is not None:
            mx_k = _l2norm_np(_project_out(mx_z, dirs)).astype("float32")
        else:
            mx_k = None
        for alpha in alphas:
            corpus = _build_from_precorrected(t_k, i_k, modes, alpha, mixed_corr=mx_k)
            results[f"z_subK_a{alpha:.1f}_k{K_FIXED}"] = evaluate_ranking(
                q_k @ corpus.T, q_gt, modes=modes)

    return results


def extract_best_per_method(results):
    """method prefix별 MRR 기준 best α를 골라서 메트릭 dict 반환."""
    out = {}
    for prefix, label in METHOD_DEFS:
        keys = [k for k in results if k.startswith(prefix + "_")]
        if not keys:
            continue
        best_key = max(keys, key=lambda k: results[k]["MRR"])
        out[label] = {m: results[best_key].get(m, 0.0) for m in CELL_METRICS}
    return out


def format_cell(metrics_dict):
    """CELL_METRICS 순서대로 "0.xxx/0.xxx/0.xxx" 문자열 생성."""
    if not metrics_dict:
        return "—/—/—"
    return "/".join(f"{metrics_dict[m]:.3f}" for m in CELL_METRICS)


def print_table1(all_results, backbones):
    """GR-CLIP 논문 Table 1 형식 출력.
    행: backbone × method (Raw/Z-Score/Ours 등)
    열: 4 datasets
    셀: CELL_METRICS 순서 (기본 NDCG@10 / R@10 / R@1)
    """
    print(f"\n{'='*120}")
    _cells = " / ".join(CELL_METRICS)
    print(f"  Table 1: Detailed results across all metrics on MixBench "
          f"({_cells})")
    print(f"{'='*120}")

    col_w = 22
    print(f"  {'Method':<28s}  " + "  ".join(f"{d:>{col_w}s}" for d in DATASETS))
    print("  " + "-" * (28 + (col_w + 2) * len(DATASETS)))

    # 1) Baselines: 각 backbone의 Raw
    for tag in backbones:
        display = BACKBONES[tag]["display"]
        row = f"  {display:<28s}"
        for ds in DATASETS:
            res = all_results.get((tag, ds), {})
            row += f"  {format_cell(res.get('Raw')):>{col_w}s}"
        print(row)

    print("  " + "-" * (28 + (col_w + 2) * len(DATASETS)))

    # 2) GR-CLIP per backbone (논문 표와 동일)
    for tag in backbones:
        display = BACKBONES[tag]["display"]
        row = f"  {'GR-' + display:<28s}"
        for ds in DATASETS:
            res = all_results.get((tag, ds), {})
            row += f"  {format_cell(res.get('GR-CLIP')):>{col_w}s}"
        print(row)

    print("  " + "-" * (28 + (col_w + 2) * len(DATASETS)))

    # 3) 제안 방법 per backbone — Z + Subspace Top-K
    for tag in backbones:
        display = BACKBONES[tag]["display"]
        row = f"  {'Ours-' + display:<28s}"
        for ds in DATASETS:
            res = all_results.get((tag, ds), {})
            row += f"  {format_cell(res.get('Ours (Z+Sub)')):>{col_w}s}"
        print(row)

    print("  " + "-" * (28 + (col_w + 2) * len(DATASETS)))


def print_mrr_summary(all_results, backbones):
    """추가 요약: backbone × method MRR — 데이터셋 평균."""
    print(f"\n{'='*100}")
    print(f"  요약: 데이터셋 평균 MRR (4개 dataset 평균)")
    print(f"{'='*100}")

    methods_to_show = ["Raw", "GR-CLIP", "Z-Score", "Ours (Z+Sub)"]
    print(f"  {'Backbone':<22s}" + "  ".join(f"{m:>14s}" for m in methods_to_show))
    print("  " + "-" * (22 + 16 * len(methods_to_show)))

    for tag in backbones:
        display = BACKBONES[tag]["display"]
        row = f"  {display:<22s}"
        for method in methods_to_show:
            mrrs = []
            for ds in DATASETS:
                res = all_results.get((tag, ds), {}).get(method)
                if res and "MRR" in res:
                    mrrs.append(res["MRR"])
            if mrrs:
                row += f"  {np.mean(mrrs):>14.4f}"
            else:
                row += f"  {'—':>14s}"
        print(row)
    print("  " + "-" * (22 + 16 * len(methods_to_show)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=DATASETS,
                    help="단일 dataset만 실행 (기본: 4개 전부)")
    ap.add_argument("--all", action="store_true",
                    help="4개 dataset × CLIP 5 backbone (default 동작)")
    ap.add_argument("--backbones", nargs="+",
                    default=["B16", "L14", "OC_B16", "OC_L14", "SigLIP400M"],
                    choices=list(BACKBONES.keys()),
                    help="포함할 backbone tag 리스트")
    ap.add_argument("--include-vlm2vec", action="store_true",
                    help="VLM2Vec_LN, VLM2Vec_QW, VLM2Vec_V2 추가 (캐시 있어야 함)")
    ap.add_argument("--alpha", type=float, default=None,
                    help="특정 α로 고정 (sweep 생략, 단일 α만). "
                         "출력 파일명에 _a0.5 접미사가 붙어 sweep 결과와 분리됨")
    ap.add_argument("--from-raw", action="store_true",
                    help="기존 backbone_comparison_raw.json 재사용해서 표만 재구성 "
                         "(임베딩/α-sweep 재계산 없음, CELL_METRICS 만 바꿀 때 편리)")
    args = ap.parse_args()

    backbones = list(args.backbones)
    if args.include_vlm2vec:
        backbones.extend(["VLM2Vec_LN", "VLM2Vec_QW", "VLM2Vec_V2"])

    subsets = [args.subset] if args.subset else DATASETS
    alphas = [args.alpha] if args.alpha is not None else ALPHAS

    np.random.seed(SEED)

    RESULT_DIR = _MIXBENCH_DIR.parent / "MixBench_results" / "backbone_comparison"
    RESULT_DIR.mkdir(exist_ok=True, parents=True)

    # α 고정 시 출력 파일명에 접미사를 붙여 sweep 결과를 덮어쓰지 않는다.
    sfx = "" if args.alpha is None else f"_a{args.alpha:g}"
    OUT_JSON = RESULT_DIR / f"backbone_comparison{sfx}.json"
    RAW_JSON = RESULT_DIR / f"backbone_comparison_raw{sfx}.json"

    # ── --from-raw: 기존 α-sweep 결과만 읽어 표 재구성 ──
    if args.from_raw:
        raw_path = RAW_JSON
        if not raw_path.exists():
            raise FileNotFoundError(
                f"{raw_path} 없음. 먼저 재계산 모드로 한 번 돌려서 raw json 을 만드세요."
            )
        with open(raw_path) as f:
            raw_dump = json.load(f)

        all_results = {}
        for key, results in raw_dump.items():
            tag, ds = key.split("__")
            if tag not in backbones or ds not in subsets:
                continue
            method_bests = extract_best_per_method(results)
            for prefix, label in METHOD_DEFS:
                keys = [k for k in results if k.startswith(prefix + "_")]
                if keys and label in method_bests:
                    best_key = max(keys, key=lambda k: results[k]["MRR"])
                    method_bests[label]["MRR"] = results[best_key]["MRR"]
            all_results[(tag, ds)] = method_bests

        print_table1(all_results, backbones)
        print_mrr_summary(all_results, backbones)

        # backbone_comparison.json 을 새 CELL_METRICS 로 다시 저장
        out_path = OUT_JSON
        summary = {f"{t}__{d}": v for (t, d), v in all_results.items()}
        with open(out_path, "w") as f:
            json.dump({
                "k_fixed": K_FIXED,
                "alphas": alphas,
                "cell_metrics": CELL_METRICS,
                "backbones": {t: BACKBONES[t]["display"] for t in backbones},
                "best_per_method": summary,
            }, f, indent=2)
        print(f"\n저장: {out_path}")
        return

    # all_results: {(backbone_tag, dataset): {method_label: {metric: val}}}
    all_results = {}
    # raw_results: {(backbone_tag, dataset): {key: {metric: val}}}  — 디스크 저장용
    raw_results = {}

    for backbone_tag in backbones:
        bcfg = BACKBONES[backbone_tag]
        print(f"\n{'#'*80}")
        print(f"  Backbone: {bcfg['display']} ({backbone_tag})  "
              f"family={bcfg['family']}  dim={bcfg['dim']}")
        print(f"{'#'*80}")

        # VLM2Vec family 면 mixed_emb 까지 로드 (옵션 B)
        use_mixed = bcfg["family"] == "vlm2vec"

        for subset in subsets:
            print(f"\n--- [{subset}] ---")
            if use_mixed:
                text_emb, image_emb, query_emb, modes, q_gt, mixed_emb = load_mixbench(
                    subset, backbone_tag=backbone_tag, return_mixed=True)
                if mixed_emb is None:
                    print(f"  ⚠ mixed_emb 없음 → 옵션 A (alpha-blend) 로 fallback")
            else:
                text_emb, image_emb, query_emb, modes, q_gt = load_mixbench(
                    subset, backbone_tag=backbone_tag)
                mixed_emb = None

            results = run_methods_one(text_emb, image_emb, query_emb, modes, q_gt, alphas,
                                       mixed_emb=mixed_emb)
            method_bests = extract_best_per_method(results)
            # MRR도 함께 저장 (요약용)
            for prefix, label in METHOD_DEFS:
                keys = [k for k in results if k.startswith(prefix + "_")]
                if keys:
                    best_key = max(keys, key=lambda k: results[k]["MRR"])
                    method_bests[label]["MRR"] = results[best_key]["MRR"]
            all_results[(backbone_tag, subset)] = method_bests

            # 디스크에는 raw α-sweep 결과 그대로 저장
            raw_save = {}
            for k, v in results.items():
                raw_save[k] = {kk: vv for kk, vv in v.items()
                               if isinstance(vv, (int, float))}
            raw_results[(backbone_tag, subset)] = raw_save

            # 진행상황 한 줄 출력
            r1 = method_bests.get("Raw", {}).get("NDCG@10", 0)
            r2 = method_bests.get("Ours (Z+Sub)", {}).get("NDCG@10", 0)
            delta = (r2 - r1) / r1 * 100 if r1 > 0 else 0
            print(f"  → NDCG@10  Raw={r1:.3f}  Ours={r2:.3f}  ({delta:+.1f}%)")

    # ── 최종 표 출력 ──
    print_table1(all_results, backbones)
    print_mrr_summary(all_results, backbones)

    # ── 저장 ──
    out_path = OUT_JSON
    summary = {}
    for (tag, ds), bests in all_results.items():
        key = f"{tag}__{ds}"
        summary[key] = bests
    with open(out_path, "w") as f:
        json.dump({
            "k_fixed": K_FIXED,
            "alphas": alphas,
            "cell_metrics": CELL_METRICS,
            "backbones": {t: BACKBONES[t]["display"] for t in backbones},
            "best_per_method": summary,
        }, f, indent=2)
    print(f"\n저장: {out_path}")

    raw_path = RAW_JSON
    raw_dump = {f"{t}__{d}": v for (t, d), v in raw_results.items()}
    with open(raw_path, "w") as f:
        json.dump(raw_dump, f, indent=2)
    print(f"전체 α-sweep 저장: {raw_path}")


if __name__ == "__main__":
    main()
