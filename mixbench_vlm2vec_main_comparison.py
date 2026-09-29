"""
MixBench2026 — 메인 실험 + VLM2Vec baseline 추가
==================================================================
mixbench_full_comparison_ext.py 의 5가지 방법 (CLIP-L/14) 결과에
VLM2Vec(LLaVaNext), VLM2Vec(Qwen) 을 추가 baseline 행으로 합쳐 표 출력.

비교 대상:
  [메인 — CLIP-L/14]
    1. Raw                          — baseline
    2. GR-CLIP                      — mean shift only
    3. Global Z-Score               — embedding-level z-score
    9. Z + Subspace Top-K ★ 제안    — paired diff SVD top-k 제거 (k=16 fixed)
    11. Z-Score + Adaptive          — score-level adaptive

  [추가 baseline — 별도 backbone]
    VLM2Vec(LLaVaNext)              — native multimodal embedding
    VLM2Vec(Qwen)                   — native multimodal embedding
    * 옵션 B: 'both' 항목은 alpha-blend 대신 corpus_mixed_emb.npy 직접 사용
      (encode_vlm2vec.py 가 생성한 캐시가 있을 때 자동).

전제:
  - CLIP 캐시: cache/cache_<subset>_L14/        (없으면 data_loader 가 자동 인코딩)
  - VLM2Vec 캐시: cache/cache_<subset>_VLM2Vec_{LN,QW}/
                  (별도 env 에서 encode_vlm2vec.py 로 미리 생성 필요)

실행:
  conda activate jiwoo
  python mixbench_vlm2vec_main_comparison.py --all
  python mixbench_vlm2vec_main_comparison.py --subset MSCOCO
  python mixbench_vlm2vec_main_comparison.py --all --skip-vlm2vec   # VLM2Vec 캐시 없을 때
  python mixbench_vlm2vec_main_comparison.py --all --alpha 0.5      # 메인 1~3,9 만 α 고정
"""

from __future__ import annotations

import os
# Windows + conda OpenMP 충돌 우회 (libiomp5md.dll 중복). 무거운 import 전에 설정.
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
    run_all_methods,
    METHOD_CONFIGS,
    _extract_method_bests,
)


# 메인 표에 추가할 VLM2Vec baseline backbone tag (data_loader.BACKBONES 키)
# V1 (LN/QW 7B) + V2 (Qwen 2B, 단일 변형). 캐시가 없는 tag 는 자동 스킵.
VLM2VEC_TAGS = ["VLM2Vec_LN", "VLM2Vec_QW", "VLM2Vec_V2"]

# 출력할 metric 컬럼 (full_comparison_ext.print_cross_dataset_table 과 동일)
METRICS_TO_SHOW = ["MRR", "R@1", "R@5", "R@10", "NDCG@10", "NDCG@100"]


# ─────────────────────────────────────────────
# VLM2Vec native baseline 평가
#   - data_loader 가 캐시(corpus_{text,image}_emb / query_emb / corpus_mixed_emb) 로드
#   - 'Raw' 스타일 평가: VLM2Vec 자체가 학습된 multimodal embedding 이므로
#     별도 mean-shift / z-score / subspace projection 없이 그대로 사용
#   - 'both' 항목: mixed_emb 가 있으면 옵션 B (직접 사용), 없으면 옵션 A (alpha-blend)
#   - α sweep: 옵션 B 인 경우 alpha 영향 없으나, fallback 호환 위해 sweep 유지
# ─────────────────────────────────────────────
def run_vlm2vec_baseline(subset, backbone_tag, alphas=None):
    if alphas is None:
        alphas = ALPHAS

    try:
        text_emb, image_emb, query_emb, modes, q_gt, mixed_emb = load_mixbench(
            subset, backbone_tag=backbone_tag, return_mixed=True
        )
    except Exception as e:
        print(f"  ⚠ {backbone_tag} 캐시 로드 실패: {e}")
        return None

    if len(query_emb) == 0 or len(text_emb) == 0:
        print(f"  비어있음 (queries={len(query_emb)}, corpus={len(text_emb)})")
        return None

    use_mixed = mixed_emb is not None
    if not use_mixed:
        print(f"  (mixed_emb 없음 → 옵션 A: alpha-blend fallback)")

    best = None
    for alpha in alphas:
        corpus = build_emb_corpus(text_emb, image_emb, modes, alpha,
                                  mixed_emb=mixed_emb if use_mixed else None)
        q = process_emb_query(query_emb)
        scores = q @ corpus.T
        res = evaluate_ranking(scores, q_gt, modes=modes)
        if best is None or res["MRR"] > best["MRR"]:
            best = dict(res)
            best["_alpha"] = float(alpha)
            best["_option"] = "B(mixed_emb)" if use_mixed else "A(alpha-blend)"
    return best


# ─────────────────────────────────────────────
# 통합 표 출력
#   메인 5 방법 (CLIP-L/14) + VLM2Vec 2 baseline 행
#   metric 별로 표 1개씩 (full_comparison_ext.print_cross_dataset_table 패턴)
# ─────────────────────────────────────────────
def print_main_table(all_dataset_results, vlm2vec_results, subsets):
    main_labels = [label for _, label in METHOD_CONFIGS]
    has_adaptive = any("zscore+adaptive" in r for r in all_dataset_results.values()
                       if r is not None)
    if has_adaptive:
        main_labels.append("11. Z-Score+Adaptive")

    # backbone_tag → display label
    vlm2vec_labels = [(tag, BACKBONES[tag]["display"]) for tag in VLM2VEC_TAGS
                       if any((tag, s) in vlm2vec_results for s in subsets)]

    label_w = max([len(m) for m in main_labels] +
                  [len(d) for _, d in vlm2vec_labels] + [22])

    for metric in METRICS_TO_SHOW:
        print(f"\n{'='*100}")
        print(f"  메인 실험 + VLM2Vec baseline — {metric}  "
              f"(메인 backbone: CLIP-L/14, k=16 fixed)")
        print(f"{'='*100}")
        hdr = f"  {'Method':<{label_w}s}"
        for s in subsets:
            hdr += f"  {s:>15s}"
        print(hdr)
        print("  " + "-" * (label_w + 17 * len(subsets)))

        # ── 메인 (CLIP-L/14) ──
        for m in main_labels:
            row = f"  {m:<{label_w}s}"
            for s in subsets:
                r = all_dataset_results.get(s)
                if r is None:
                    row += f"  {'N/A':>15s}"
                    continue
                bests = _extract_method_bests(r)
                v = bests.get(m, (None, None))
                val = v[0][metric] if v[0] and metric in v[0] else None
                row += f"  {val:>15.4f}" if val is not None else f"  {'N/A':>15s}"
            print(row)

        # ── 구분선 + VLM2Vec baseline ──
        if vlm2vec_labels:
            print("  " + "-" * (label_w + 17 * len(subsets)))
            for tag, display in vlm2vec_labels:
                row = f"  {display:<{label_w}s}"
                for s in subsets:
                    res = vlm2vec_results.get((tag, s))
                    val = res[metric] if res and metric in res else None
                    row += f"  {val:>15.4f}" if val is not None else f"  {'N/A':>15s}"
                print(row)
        print("  " + "-" * (label_w + 17 * len(subsets)))


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["MSCOCO", "VisualNews", "Google_WIT", "OVEN"])
    ap.add_argument("--all", action="store_true", help="4개 데이터셋 전부 실행")
    ap.add_argument("--alpha", type=float, default=None,
                    help="메인 1~3, 9번 α 고정 (sweep / 11번 생략). VLM2Vec sweep 은 유지.")
    ap.add_argument("--skip-vlm2vec", action="store_true",
                    help="VLM2Vec baseline 측정 생략 (캐시가 없을 때)")
    ap.add_argument("--vlm2vec-only", nargs="+", choices=VLM2VEC_TAGS,
                    help="VLM2Vec 중 일부 backbone tag만 평가 (기본: 양쪽 다)")
    args = ap.parse_args()

    if not args.all and not args.subset:
        ap.error("--subset 또는 --all 중 하나를 지정하세요")
    if args.alpha is not None and not (0.0 <= args.alpha <= 1.0):
        ap.error("--alpha 는 0.0 ~ 1.0 범위여야 합니다")

    alphas = [args.alpha] if args.alpha is not None else ALPHAS
    skip_adaptive = args.alpha is not None
    vlm2vec_tags = list(args.vlm2vec_only) if args.vlm2vec_only else list(VLM2VEC_TAGS)

    np.random.seed(SEED)

    RESULT_DIR = _MIXBENCH_DIR.parent / "MixBench_results"
    RESULT_DIR.mkdir(exist_ok=True, parents=True)

    subsets = ["MSCOCO", "VisualNews", "Google_WIT", "OVEN"] if args.all else [args.subset]

    all_dataset_results = {}     # {subset: results dict from run_all_methods}
    vlm2vec_results = {}         # {(tag, subset): best metric dict}

    for subset in subsets:
        print(f"\n{'#'*80}")
        print(f"  데이터셋: {subset}")
        print(f"{'#'*80}")

        # ── 메인 5 방법 (CLIP-L/14, mixbench_full_comparison_ext 재사용) ──
        results = run_all_methods(subset, inspect_n=0, alphas=alphas,
                                  skip_adaptive=skip_adaptive)
        all_dataset_results[subset] = results

        # ── VLM2Vec baseline 행 ──
        if not args.skip_vlm2vec:
            for tag in vlm2vec_tags:
                print(f"\n[VLM2Vec baseline] {BACKBONES[tag]['display']} on {subset}...")
                res = run_vlm2vec_baseline(subset, tag, alphas=alphas)
                if res is None:
                    continue
                vlm2vec_results[(tag, subset)] = res
                print(f"  best α={res['_alpha']:.1f}  option={res['_option']}  "
                      f"MRR={res['MRR']:.4f}  NDCG@10={res['NDCG@10']:.4f}")

    # ── 통합 표 ──
    print_main_table(all_dataset_results, vlm2vec_results, subsets)

    # ── 저장 ──
    save = {}
    for subset in subsets:
        block = {"main": {}, "vlm2vec": {}}
        r = all_dataset_results.get(subset) or {}
        for k, v in r.items():
            block["main"][k] = {kk: vv for kk, vv in v.items()
                                 if isinstance(vv, (int, float))}
        for tag in vlm2vec_tags:
            res = vlm2vec_results.get((tag, subset))
            if res is not None:
                block["vlm2vec"][tag] = {kk: vv for kk, vv in res.items()
                                          if isinstance(vv, (int, float, str))}
        save[subset] = block

    out_name = (f"vlm2vec_main_comparison_a{args.alpha:.1f}.json"
                if args.alpha is not None else "vlm2vec_main_comparison.json")
    out_path = RESULT_DIR / out_name
    with open(out_path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
