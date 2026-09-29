"""
MixBench2026 — Data Loader (mixbench_full_comparison.py의 통합 헬퍼 라이브러리)

주요 변경 사항 (vs MixBench/mixbench_full_comparison.py):
  - _load_wit_as_mixbench() 함수: v2 구조 (4,421 corpus + 1,000 queries)
    → MixBench2026 공식 Google_WIT split과 동일 규모 (이전: 990/990)
  - filter_by_experiment() 함수 추가
    → mixbench_three_experiments.py에서 통합 (의존성 일원화)

이 파일은 modality_gap_research_v1/ 내부의 다음 5개 파일에서 import 됨:
  - mixbench_full_comparison_ext.py    (제안 방법 9, 11 비교)
  - mixbench_three_experiments_ext.py  (§3 / §4 / §5 실험)
  - mixbench_ablation_5way_k16.py      (5-way ablation)
  - mixbench_k_sensitivity_v2.py       (k 민감도 분석)
  - multimodal_subspace_visualization.py

데이터셋: MixBench2026 (MSCOCO, VisualNews, OVEN) + 자체 구성 Google_WIT (v2)

데이터 로드 흐름:
  load_mixbench(subset) →
    - MSCOCO/VisualNews/OVEN: mixed-modality-search/MixBench2026 직접 다운로드
    - Google_WIT: wikimedia/wit_base에서 v2 (4,421 corpus + 1,000 queries) 구성
  → CLIP ViT-L/14 인코딩 → 캐시 저장 (cache_{subset}_L14/)

실행:
  보통 직접 실행하지 않고 다른 파일에서 import해서 사용.
  단독 실행 시 mixbench_full_comparison.py와 동일하게 5 방법 비교를 수행.

  conda activate jiwoo
  python data_loader.py --subset MSCOCO
  python data_loader.py --all
"""

import argparse
import json
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

TOP_KS = [1, 3, 5, 10, 20]
NDCG_KS = [5, 10, 20, 100]
RRF_K = 60
SEED = 42
K_LOCAL = 100

CLIP_MODEL = "ViT-L-14"
CLIP_PRETRAINED = "openai"
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# ─────────────────────────────────────────────
# Backbone 레지스트리 (mixbench_backbone_comparison.py 용)
#   tag         : 캐시 디렉토리 이름 (cache_{subset}_{tag}/)
#   family      : "open_clip" 또는 "vlm2vec"
#   model       : open_clip 모델명 또는 HuggingFace repo
#   pretrained  : open_clip pretrained tag (vlm2vec는 None)
#   dim         : embedding dimension
#   display     : 표 출력용 이름 (GR-CLIP paper Table 1 형식)
# ─────────────────────────────────────────────
BACKBONES = {
    "B16":        {"family": "open_clip", "model": "ViT-B-16",                  "pretrained": "openai",                "dim": 512,  "display": "CLIP-B/16"},
    "L14":        {"family": "open_clip", "model": "ViT-L-14",                  "pretrained": "openai",                "dim": 768,  "display": "CLIP-L/14"},
    "OC_B16":     {"family": "open_clip", "model": "ViT-B-16",                  "pretrained": "laion2b_s34b_b88k",     "dim": 512,  "display": "OpenCLIP-B/16"},
    "OC_L14":     {"family": "open_clip", "model": "ViT-L-14",                  "pretrained": "laion2b_s32b_b82k",     "dim": 768,  "display": "OpenCLIP-L/14"},
    "SigLIP400M": {"family": "open_clip", "model": "ViT-SO400M-14-SigLIP-384",  "pretrained": "webli",                 "dim": 1152, "display": "SigLIP-400m"},

    # VLM2Vec: MMEBModel 기반. encode_vlm2vec.py 에서 별도 env 로 인코딩 후 캐시 저장.
    # tablerag env에서 본 load_mixbench 는 캐시만 읽음 → MMEBModel 로딩 안 됨.
    "VLM2Vec_LN": {
        "family": "vlm2vec",
        "model": "TIGER-Lab/VLM2Vec-LLaVa-Next",   # GR-CLIP paper main result variant
        "model_backbone": "llava_next",
        "checkpoint_path": None,                    # LLaVa-Next 는 full weights (no LoRA)
        "lora": False,
        "image_token": "<image>",
        "pretrained": None, "dim": 4096,            # llava-v1.6-mistral-7b hidden dim
        "display": "VLM2Vec(LLaVaNext)",
    },
    "VLM2Vec_QW": {
        "family": "vlm2vec",
        "model": "Qwen/Qwen2-VL-7B-Instruct",       # base model
        "model_backbone": "qwen2_vl",
        "checkpoint_path": "TIGER-Lab/VLM2Vec-Qwen2VL-7B",  # LoRA adapter
        "lora": True,
        "image_token": "<|image_pad|>",
        "pretrained": None, "dim": 3584,            # Qwen2-VL-7B hidden dim
        "display": "VLM2Vec(Qwen)",
    },

    # VLM2Vec V2 — codebase 완전 overhaul, ../VLM2Vec_V2/ 별도 clone 필요.
    # 인코딩은 encode_vlm2vec_v2.py 에서 별도 env (vlm2vec_v2) 로 수행.
    # data_loader 는 그 결과 캐시 (cache/cache_<subset>_VLM2Vec_V2/) 만 읽음.
    "VLM2Vec_V2": {
        "family": "vlm2vec",
        "model": "Qwen/Qwen2-VL-2B-Instruct",       # V2 base (7B 가 아닌 2B!)
        "model_backbone": "qwen2_vl",
        "checkpoint_path": "VLM2Vec/VLM2Vec-V2.0",  # LoRA adapter (V2 단일 변형)
        "lora": True,
        "image_token": "<|image_pad|>",
        "pretrained": None, "dim": 1536,            # Qwen2-VL-2B hidden dim
        "display": "VLM2Vec-V2(Qwen-2B)",
    },
}


# ─────────────────────────────────────────────
# CLIP 임베딩 헬퍼
# ─────────────────────────────────────────────
def _encode_texts(model, tokenizer, texts, device, batch_size=128):
    import torch
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            tok = tokenizer(batch).to(device)
            f = model.encode_text(tok).float()
            f = f / (f.norm(dim=-1, keepdim=True) + 1e-8)
        embs.append(f.cpu().numpy())
    return np.concatenate(embs).astype("float32")


def _encode_pil_images(model, preprocess, images, device, batch_size=32):
    import torch
    model_dtype = next(model.parameters()).dtype
    embs = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        tensors = [preprocess(img) for img in batch]
        batch_tensor = torch.stack(tensors).to(device=device, dtype=model_dtype)
        with torch.no_grad():
            f = model.encode_image(batch_tensor).float()
            f = f / (f.norm(dim=-1, keepdim=True) + 1e-8)
        embs.append(f.cpu().numpy())
    return np.concatenate(embs).astype("float32")


# ─────────────────────────────────────────────
# WIT → MixBench 형식 변환
# ─────────────────────────────────────────────
def _parse_english_caption(cap):
    if not cap or "English:" not in cap:
        return None
    idx = cap.index("English:")
    en_part = cap[idx + len("English:"):]
    match = re.search(r"\s[A-ZÀ-Ž][a-zà-ž].*?:", en_part)
    if match:
        en_part = en_part[:match.start()]
    en_part = en_part.strip()
    return en_part if len(en_part) >= 5 else None


def _find_english_context(wit_features):
    urls = wit_features.get("page_url", [])
    ctxs = wit_features.get("context_page_description", [])
    for j, url in enumerate(urls):
        if "en.wikipedia.org" in str(url):
            if j < len(ctxs) and ctxs[j] and len(ctxs[j].strip()) >= 10:
                return ctxs[j].strip()
    return None


def _load_wit_as_mixbench(n_corpus=4421, n_queries=1000):
    """v2 WIT — MixBench2026/Google_WIT 공식 규모와 동일하게 구성.

    Args:
      n_corpus: corpus 항목 수 (default 4421 = MixBench2026 공식)
      n_queries: query 수 (default 1000 = MixBench2026 공식)

    Returns:
      ds_q_list: [{"id": str, "text": str}] — n_queries개
      ds_c_list: [{"id": str, "text": str|None, "image": PIL|None}] — n_corpus개
      modes: list of "text"|"image"|"both" — corpus modes

    Structure (MixBench2026/Google_WIT 공식과 일치):
      - 1,000 queries (text caption)
      - 4,421 corpus (1/3씩 text-only/image-only/both 모드)
      - 3,421 distractor (query 없는 corpus 항목)
    """
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wit_base", split="train", streaming=True)
    items = []
    scanned = 0
    for sample in ds:
        scanned += 1
        if len(items) >= n_corpus:
            break
        en_cap = _parse_english_caption(sample.get("caption_attribution_description"))
        if not en_cap:
            continue
        en_ctx = _find_english_context(sample.get("wit_features", {}))
        if not en_ctx:
            continue
        img = sample.get("image")
        if img is None:
            continue
        if img.mode != "RGB":
            img = img.convert("RGB")
        items.append((en_cap, en_ctx, img))

    print(f"  WIT 스캔: {scanned:,}개 → 유효: {len(items):,}개")
    N = len(items)
    third = N // 3
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(N)
    both_idx = set(perm[:third].tolist())
    text_idx = set(perm[third:2 * third].tolist())

    # ── Corpus 구성 (전체 N개 = 4,421) ──
    ds_c_list = []
    modes = []
    for i, (q_text, c_text, c_img) in enumerate(items):
        sid = str(i)
        if i in both_idx:
            ds_c_list.append({"id": sid, "text": c_text, "image": c_img})
            modes.append("both")
        elif i in text_idx:
            ds_c_list.append({"id": sid, "text": c_text, "image": None})
            modes.append("text")
        else:
            ds_c_list.append({"id": sid, "text": None, "image": c_img})
            modes.append("image")

    # ── Query 구성 (처음 n_queries개 = 1,000) ──
    n_q = min(n_queries, N)
    ds_q_list = []
    for i in range(n_q):
        sid = str(i)
        ds_q_list.append({"id": sid, "text": items[i][0]})

    print(f"  Corpus: {N:,}개 (text={modes.count('text'):,}, "
          f"image={modes.count('image'):,}, both={modes.count('both'):,})")
    print(f"  Queries: {len(ds_q_list):,}개 / Distractors: {N - n_q:,}개")
    return ds_q_list, ds_c_list, modes


# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────
def _load_open_clip(backbone_cfg, device):
    """open_clip 모델 로드 → (encode_texts_fn, encode_images_fn, dim)"""
    import open_clip
    import torch
    model, _, preprocess = open_clip.create_model_and_transforms(
        backbone_cfg["model"], pretrained=backbone_cfg["pretrained"], device=device
    )
    tokenizer = open_clip.get_tokenizer(backbone_cfg["model"])
    model = model.half().eval() if device == "cuda" else model.eval()

    def encode_texts_fn(texts):
        return _encode_texts(model, tokenizer, texts, device)

    def encode_images_fn(images):
        return _encode_pil_images(model, preprocess, images, device)

    return encode_texts_fn, encode_images_fn, backbone_cfg["dim"]


def _load_vlm2vec(backbone_cfg, device):
    """VLM2Vec 모델은 별도 env (`vlm2vec`) 에서 `encode_vlm2vec.py` 로 인코딩하고
    캐시 디렉토리에 결과를 저장하는 방식으로만 지원함.

    이 함수가 호출됐다는 건 cache miss 라는 뜻 → 명확한 안내 메시지 출력 후 raise.
    """
    raise RuntimeError(
        f"VLM2Vec 캐시가 없습니다 (cache_*_{backbone_cfg['display']} 폴더 비어있음).\n"
        f"본 env (tablerag) 에는 MMEBModel 의존성이 없습니다.\n"
        f"별도 env 에서 먼저 인코딩하세요:\n"
        f"    conda activate vlm2vec\n"
        f"    python encode_vlm2vec.py --backbone {backbone_cfg.get('model_backbone','?')} "
        f"--subset <MSCOCO|VisualNews|Google_WIT|OVEN|all>\n"
        f"인코딩이 끝나면 본 스크립트가 캐시를 자동 읽어들입니다."
    )


def load_mixbench(subset, backbone_tag="L14", return_mixed=False):
    """캐시 로드.

    Args:
      return_mixed: True 면 (..., mixed_emb) 6-tuple 반환. mixed_emb 파일이 없으면 None.
                    VLM2Vec backbone (옵션 B) 에서 'both' 항목 retrieval 에 사용.
    """
    from datasets import load_dataset
    import torch

    backbone_cfg = BACKBONES[backbone_tag]
    CACHE_DIR = Path(__file__).parent / "cache" / f"cache_{subset}_{backbone_tag}"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{subset} / {backbone_cfg['display']}] cache: {CACHE_DIR.name}")

    cache_files = {
        "text_emb": CACHE_DIR / "corpus_text_emb.npy",
        "image_emb": CACHE_DIR / "corpus_image_emb.npy",
        "query_emb": CACHE_DIR / "query_emb.npy",
        "modes": CACHE_DIR / "modes.json",
        "gt": CACHE_DIR / "q_gt.json",
    }
    mixed_path = CACHE_DIR / "corpus_mixed_emb.npy"

    if all(f.exists() for f in cache_files.values()):
        text_emb = np.load(cache_files["text_emb"])
        image_emb = np.load(cache_files["image_emb"])
        query_emb = np.load(cache_files["query_emb"])
        with open(cache_files["modes"]) as f:
            modes = json.load(f)
        with open(cache_files["gt"]) as f:
            q_gt = json.load(f)
        print(f"[{subset}] 캐시 로드 ({backbone_cfg['display']})... corpus: {len(text_emb)}, queries: {len(query_emb)}")
        if return_mixed:
            mixed_emb = np.load(mixed_path) if mixed_path.exists() else None
            return text_emb, image_emb, query_emb, modes, q_gt, mixed_emb
        return text_emb, image_emb, query_emb, modes, q_gt

    if subset == "Google_WIT":
        print(f"[Google_WIT] wikimedia/wit_base에서 로드...")
        ds_q, ds_c, modes = _load_wit_as_mixbench()
    else:
        print(f"[{subset}] MixBench2026 로드...")
        ds_q = load_dataset("mixed-modality-search/MixBench2026", name=subset, split="queries")
        ds_c = load_dataset("mixed-modality-search/MixBench2026", name=subset, split="mixed_corpus")
        modes = []
        for i in range(len(ds_c)):
            has_text = ds_c[i]["text"] is not None
            has_image = ds_c[i]["image"] is not None
            if has_text and has_image:
                modes.append("both")
            elif has_text:
                modes.append("text")
            else:
                modes.append("image")

    print(f"  모달리티: both={modes.count('both')}, text={modes.count('text')}, image={modes.count('image')}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if backbone_cfg["family"] == "open_clip":
        encode_texts_fn, encode_images_fn, dim = _load_open_clip(backbone_cfg, device)
    elif backbone_cfg["family"] == "vlm2vec":
        encode_texts_fn, encode_images_fn, dim = _load_vlm2vec(backbone_cfg, device)
    else:
        raise ValueError(f"Unknown backbone family: {backbone_cfg['family']}")

    N = len(ds_c)

    text_indices = [i for i, m in enumerate(modes) if m in ("text", "both")]
    texts = [ds_c[i]["text"] for i in text_indices]
    print(f"  텍스트 임베딩 ({len(texts)}개)...")
    text_embs = encode_texts_fn(texts)
    full_text_emb = np.zeros((N, dim), dtype="float32")
    for idx, i in enumerate(text_indices):
        full_text_emb[i] = text_embs[idx]

    image_indices = [i for i, m in enumerate(modes) if m in ("image", "both")]
    images = []
    for i in image_indices:
        img = ds_c[i]["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        images.append(img)
    print(f"  이미지 임베딩 ({len(images)}개)...")
    image_embs = encode_images_fn(images)
    full_image_emb = np.zeros((N, dim), dtype="float32")
    for idx, i in enumerate(image_indices):
        full_image_emb[i] = image_embs[idx]

    # 쿼리 임베딩
    # - 일반: text-only (MSCOCO, VisualNews, Google_WIT)
    # - OVEN: query에 image+text 모두 존재 → 두 modality 평균 후 L2 정규화
    query_has_image = (
        subset == "OVEN"
        and len(ds_q) > 0
        and ds_q[0].get("image") is not None
    )
    query_texts = [ds_q[i]["text"] for i in range(len(ds_q))]
    print(f"  쿼리 텍스트 임베딩 ({len(query_texts)}개)...")
    query_text_emb = encode_texts_fn(query_texts)

    if query_has_image:
        # OVEN: query 이미지도 인코딩 → text/image 평균 후 재정규화
        query_images = []
        for i in range(len(ds_q)):
            qimg = ds_q[i]["image"]
            if qimg.mode != "RGB":
                qimg = qimg.convert("RGB")
            query_images.append(qimg)
        print(f"  쿼리 이미지 임베딩 ({len(query_images)}개) — OVEN: image+text 결합")
        query_image_emb = encode_images_fn(query_images)

        # 0.5 평균 (단순) → 두 modality 가중치 동등
        query_emb = (query_text_emb + query_image_emb) / 2.0
        norms = np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-8
        query_emb = (query_emb / norms).astype("float32")
        del query_images
    else:
        query_emb = query_text_emb

    c_id_to_idx = {str(ds_c[i]["id"]): i for i in range(len(ds_c))}
    q_gt = []
    valid_q = []
    for qi in range(len(ds_q)):
        qid = str(ds_q[qi]["id"])
        if qid in c_id_to_idx:
            q_gt.append(c_id_to_idx[qid])
            valid_q.append(qi)
    query_emb = query_emb[valid_q]

    np.save(cache_files["text_emb"], full_text_emb)
    np.save(cache_files["image_emb"], full_image_emb)
    np.save(cache_files["query_emb"], query_emb)
    with open(cache_files["modes"], "w") as f:
        json.dump(modes, f)
    with open(cache_files["gt"], "w") as f:
        json.dump(q_gt, f)

    del images
    print(f"[{subset} / {backbone_cfg['display']}] corpus: {N}, queries: {len(query_emb)}")
    return full_text_emb, full_image_emb, query_emb, modes, q_gt


# ─────────────────────────────────────────────
# 공통 평가
# ─────────────────────────────────────────────
def compute_ndcg(ranked, gt_idx, k):
    dcg = 0.0
    for rank in range(min(k, len(ranked))):
        if ranked[rank] == gt_idx:
            dcg += 1.0 / np.log2(rank + 2)
    idcg = 1.0 / np.log2(2)
    return dcg / idcg if idcg > 0 else 0.0


def inspect_top5_modes(scores_matrix, q_gt, modes, label, n_queries=10):
    """첫 n_queries개 쿼리에 대해 top-5 결과의 id·modality·GT 매칭 여부 출력"""
    print(f"\n  [{label}] 샘플 쿼리 {n_queries}개의 Top-5 modality")
    print(f"  {'Q#':>4s}  {'GT':>5s}({'mode':>5s})  {'rank1':>5s}({'m':>5s})  "
          f"{'rank2':>5s}({'m':>5s})  {'rank3':>5s}({'m':>5s})  "
          f"{'rank4':>5s}({'m':>5s})  {'rank5':>5s}({'m':>5s})  hit@5")
    print("  " + "-" * 115)
    n = min(n_queries, len(scores_matrix))
    for qi in range(n):
        gt_idx = q_gt[qi]
        gt_mode = modes[gt_idx]
        order = np.argsort(-scores_matrix[qi])[:5]
        hit = "✓" if gt_idx in order else "✗"
        row = f"  {qi:>4d}  {gt_idx:>5d}({gt_mode:>5s})"
        for r_idx in order:
            row += f"  {int(r_idx):>5d}({modes[int(r_idx)]:>5s})"
        row += f"  {hit:>5s}"
        print(row)

    # 샘플 쿼리의 top-5 modality 분포 요약
    mode_count = {"text": 0, "image": 0, "both": 0}
    for qi in range(n):
        order = np.argsort(-scores_matrix[qi])[:5]
        for r_idx in order:
            m = modes[int(r_idx)]
            if m in mode_count:
                mode_count[m] += 1
    total = sum(mode_count.values())
    if total > 0:
        print(f"  → 샘플 {n}개 쿼리 × top-5 = {total}건 중: "
              f"text {mode_count['text']} ({mode_count['text']/total*100:.1f}%), "
              f"image {mode_count['image']} ({mode_count['image']/total*100:.1f}%), "
              f"both {mode_count['both']} ({mode_count['both']/total*100:.1f}%)")


def evaluate_ranking(item_scores_matrix, q_gt, modes=None):
    """(Q, N) score matrix → 랭킹 → 평가 + top-5 modality 분포 기록"""
    metrics = defaultdict(list)
    top5_mode_counts = {"text": 0, "image": 0, "both": 0}  # top-5 누적 카운트
    total_top5 = 0

    for qi in range(len(item_scores_matrix)):
        gt_idx = q_gt[qi]
        order = np.argsort(-item_scores_matrix[qi])

        mrr = 0.0
        for rank, idx in enumerate(order):
            if idx == gt_idx:
                mrr = 1.0 / (rank + 1)
                break
        metrics["MRR"].append(mrr)

        for k in TOP_KS:
            metrics[f"R@{k}"].append(1.0 if gt_idx in order[:k] else 0.0)
        for k in NDCG_KS:
            metrics[f"NDCG@{k}"].append(compute_ndcg(order, gt_idx, k))

        # top-5 결과의 modality 분포 기록
        if modes is not None:
            for idx in order[:5]:
                m = modes[idx]
                if m in top5_mode_counts:
                    top5_mode_counts[m] += 1
                    total_top5 += 1

    result = {k: float(np.mean(v)) for k, v in metrics.items()}
    if modes is not None and total_top5 > 0:
        result["top5_text_ratio"] = top5_mode_counts["text"] / total_top5
        result["top5_image_ratio"] = top5_mode_counts["image"] / total_top5
        result["top5_both_ratio"] = top5_mode_counts["both"] / total_top5
        result["top5_text_count"] = top5_mode_counts["text"]
        result["top5_image_count"] = top5_mode_counts["image"]
        result["top5_both_count"] = top5_mode_counts["both"]
    return result


# ─────────────────────────────────────────────
# Embedding-level 코퍼스 구성
# ─────────────────────────────────────────────
def build_emb_corpus(text_emb, image_emb, modes, alpha,
                     correction="raw",
                     text_mean=None, image_mean=None,
                     text_std=None, image_std=None,
                     mixed_emb=None, mixed_mean=None, mixed_std=None):
    """embedding-level 보정 → alpha blend → N개 코퍼스 벡터.

    옵션 B (VLM2Vec 등): mixed_emb 가 주어지면 'both' 항목 retrieval 에
    alpha-blend 대신 mixed_emb (같은 correction 적용) 를 사용. alpha 와 무관.
    mixed_mean/mixed_std 는 해당 correction 의 mixed_emb 통계.
    """
    t = text_emb.copy()
    im = image_emb.copy()

    if correction == "gr_clip":
        t = t - text_mean
        im = im - image_mean
    elif correction == "global_zscore":
        t = (t - text_mean) / text_std
        im = (im - image_mean) / image_std

    text_mask = np.array([m == "text" for m in modes])
    image_mask = np.array([m == "image" for m in modes])
    both_mask = np.array([m == "both" for m in modes])

    corpus = np.zeros_like(t)
    corpus[text_mask] = t[text_mask]
    corpus[image_mask] = im[image_mask]

    if mixed_emb is not None:
        # 옵션 B: 'both' 항목은 mixed_emb (보정 후) 사용 — alpha 무관
        mx = mixed_emb.copy()
        if correction == "gr_clip" and mixed_mean is not None:
            mx = mx - mixed_mean
        elif correction == "global_zscore" and mixed_mean is not None:
            mx = (mx - mixed_mean) / mixed_std
        corpus[both_mask] = mx[both_mask]
    else:
        # 기본 (CLIP 등): alpha-blend
        corpus[both_mask] = alpha * t[both_mask] + (1 - alpha) * im[both_mask]

    corpus = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8)
    return corpus.astype("float32")


def process_emb_query(query_emb, correction="raw",
                      query_mean=None, query_std=None):
    q = query_emb.copy()
    if correction == "gr_clip":
        q = q - query_mean
    elif correction == "global_zscore":
        q = (q - query_mean) / query_std
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
    return q.astype("float32")


# ─────────────────────────────────────────────
# Score-level: Adaptive Local Calibration
# ─────────────────────────────────────────────
def get_local_stats(t_scores, i_scores, K):
    N = len(t_scores)
    t_rank = (-t_scores).argsort().argsort()
    i_rank = (-i_scores).argsort().argsort()
    combined = np.concatenate([1.0 / (RRF_K + t_rank), 1.0 / (RRF_K + i_rank)])
    top = np.argsort(-combined)[:K]
    t_idx = top[top < N]
    i_idx = top[top >= N] - N

    tm = float(t_scores[t_idx].mean()) if len(t_idx) >= 2 else float(t_scores.mean())
    ts = float(t_scores[t_idx].std() + 1e-8) if len(t_idx) >= 2 else float(t_scores.std() + 1e-8)
    im = float(i_scores[i_idx].mean()) if len(i_idx) >= 2 else float(i_scores.mean())
    is_ = float(i_scores[i_idx].std() + 1e-8) if len(i_idx) >= 2 else float(i_scores.std() + 1e-8)
    return tm, ts, im, is_


def run_adaptive(text_emb, image_emb, query_emb, modes, q_gt, label="adaptive",
                 inspect_n=0):
    """Adaptive gap comp: per-query local z-score + adaptive α → N개 랭킹"""
    N = len(text_emb)
    all_item_scores = np.zeros((len(query_emb), N), dtype="float32")
    alpha_log = []

    for qi in tqdm(range(len(query_emb)), desc=label):
        q = query_emb[qi]
        t_scores = text_emb @ q
        i_scores = image_emb @ q

        tm, ts, im, is_ = get_local_stats(t_scores, i_scores, K_LOCAL)
        alpha = (tm - im) / ((ts + is_) / 2)
        alpha_log.append(alpha)

        t_z = (t_scores - tm) / ts
        i_z = (i_scores - im) / is_ + alpha

        for i in range(N):
            mode = modes[i]
            if mode == "text":
                all_item_scores[qi, i] = t_z[i]
            elif mode == "image":
                all_item_scores[qi, i] = i_z[i]
            else:
                all_item_scores[qi, i] = (t_z[i] + i_z[i]) / 2

    result = evaluate_ranking(all_item_scores, q_gt, modes=modes)
    result["avg_alpha"] = float(np.mean(alpha_log))
    if inspect_n > 0:
        inspect_top5_modes(all_item_scores, q_gt, modes, label, n_queries=inspect_n)
    return result


# ─────────────────────────────────────────────
# 전체 실험 실행
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# 실험별 필터링 (cross_modal / multimodal / mixed)
# 원래 mixbench_three_experiments.py에 있던 함수를 통합
# ─────────────────────────────────────────────
def filter_by_experiment(text_emb, image_emb, query_emb, modes, q_gt, experiment):
    """
    실험 세팅에 맞게 코퍼스/쿼리 서브셋 추출.

    - cross_modal: mode ∈ {text, image} 만 유지 (both 제외)
    - multimodal:  mode == "both" 만 유지
    - mixed:       전체 유지 (no-op)

    GT 인덱스를 새 코퍼스 인덱스로 재매핑하고, GT가 필터링된 쿼리는 제거.
    """
    if experiment == "mixed":
        return text_emb, image_emb, query_emb, list(modes), list(q_gt)

    if experiment == "cross_modal":
        allowed = {"text", "image"}
    elif experiment == "multimodal":
        allowed = {"both"}
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    keep_mask = np.array([m in allowed for m in modes])
    old_indices = np.where(keep_mask)[0]
    old_to_new = {int(old): new for new, old in enumerate(old_indices)}

    new_text_emb = text_emb[keep_mask]
    new_image_emb = image_emb[keep_mask]
    new_modes = [modes[i] for i in old_indices]

    new_q_gt = []
    keep_q = []
    for qi, gt_old in enumerate(q_gt):
        if int(gt_old) in old_to_new:
            new_q_gt.append(old_to_new[int(gt_old)])
            keep_q.append(qi)
    new_query_emb = query_emb[keep_q] if len(keep_q) > 0 else query_emb[:0]

    print(f"  [{experiment}] corpus: {len(modes)} → {len(new_modes)}, "
          f"queries: {len(q_gt)} → {len(new_q_gt)}")
    if len(new_modes) > 0:
        print(f"    분포: both={new_modes.count('both')}, "
              f"text={new_modes.count('text')}, image={new_modes.count('image')}")

    return new_text_emb, new_image_emb, new_query_emb, new_modes, new_q_gt


def run_all_methods(subset, inspect_n=0, alphas=None, skip_adaptive=False):
    """방법 실행 → 결과 dict 반환.
    - alphas: α 리스트 (None이면 ALPHAS 사용)
    - skip_adaptive: True면 1~3번(embedding-level)만 실행, 4~5번(adaptive) 생략
    - inspect_n > 0이면 각 방법의 best-α에서 N개 샘플 쿼리 top-5 검사"""
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

    # ── 1. Raw (baseline) ──
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
        best_alpha = float(best_raw.split("_a")[1])
        corpus = build_emb_corpus(text_emb, image_emb, modes, best_alpha)
        q = process_emb_query(query_emb)
        scores = q @ corpus.T
        inspect_top5_modes(scores, q_gt, modes, f"1. Raw (α={best_alpha})", inspect_n)

    # ── 2. GR-CLIP (기존 embedding-level) ──
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
        best_alpha = float(best_gr.split("_a")[1])
        corpus = build_emb_corpus(text_emb, image_emb, modes, best_alpha,
                                  "gr_clip", text_mean, image_mean)
        q = process_emb_query(query_emb, "gr_clip", query_mean)
        scores = q @ corpus.T
        inspect_top5_modes(scores, q_gt, modes, f"2. GR-CLIP (α={best_alpha})", inspect_n)

    # ── 3. Global Z-Score (제안 embedding-level) ──
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
        best_alpha = float(best_zs.split("_a")[1])
        corpus = build_emb_corpus(text_emb, image_emb, modes, best_alpha,
                                  "global_zscore", text_mean, image_mean,
                                  text_std, image_std)
        q = process_emb_query(query_emb, "global_zscore", query_mean, query_std)
        scores = q @ corpus.T
        inspect_top5_modes(scores, q_gt, modes, f"3. Global Z-Score (α={best_alpha})", inspect_n)

    if skip_adaptive:
        print(f"\n[--alpha 지정] 4~5번(Adaptive) 생략")
        return results

    # ── 4. Adaptive Local Calibration (제안 score-level) ──
    print(f"\n[4/5] Adaptive Local Calibration (raw 임베딩)...")
    res = run_adaptive(text_emb, image_emb, query_emb, modes, q_gt,
                       label="4. Adaptive", inspect_n=inspect_n)
    results["adaptive"] = res
    print(f"  MRR={res['MRR']:.4f}  avg_α={res['avg_alpha']:.2f}")

    # ── 5. Z-Score + Adaptive (제안 결합) ──
    print(f"\n[5/5] Z-Score + Adaptive (결합)...")
    t_z = (text_emb - text_mean) / text_std
    i_z = (image_emb - image_mean) / image_std
    q_z = (query_emb - query_mean) / query_std
    t_z = (t_z / (np.linalg.norm(t_z, axis=1, keepdims=True) + 1e-8)).astype("float32")
    i_z = (i_z / (np.linalg.norm(i_z, axis=1, keepdims=True) + 1e-8)).astype("float32")
    q_z = (q_z / (np.linalg.norm(q_z, axis=1, keepdims=True) + 1e-8)).astype("float32")

    res = run_adaptive(t_z, i_z, q_z, modes, q_gt,
                       label="5. Z-Score+Adaptive", inspect_n=inspect_n)
    results["zscore+adaptive"] = res
    print(f"  MRR={res['MRR']:.4f}  avg_α={res['avg_alpha']:.2f}")

    return results


# ─────────────────────────────────────────────
# 결과 출력
# ─────────────────────────────────────────────
def print_results(subset, results):
    """데이터셋 결과 테이블 출력"""
    print(f"\n{'='*110}")
    print(f"  [{subset}] 전체 방법론 비교 (ViT-L/14)")
    print(f"{'='*110}")

    # Embedding-level best per method
    method_bests = {}
    for prefix, label in [("raw_", "1. Raw"),
                           ("gr_clip_", "2. GR-CLIP"),
                           ("zscore_", "3. Global Z-Score")]:
        keys = [k for k in results if k.startswith(prefix)]
        if keys:
            best_key = max(keys, key=lambda k: results[k]["MRR"])
            best_alpha = float(best_key.split("_a")[1])
            method_bests[label] = (results[best_key], best_alpha)

    # Score-level
    if "adaptive" in results:
        method_bests["4. Adaptive (score)"] = (results["adaptive"], None)
    if "zscore+adaptive" in results:
        method_bests["5. Z-Score+Adaptive"] = (results["zscore+adaptive"], None)

    # 테이블
    print(f"  {'Method':<25s}  {'α':>5s}  {'MRR':>8s}  {'R@1':>8s}  {'R@5':>8s}  "
          f"{'R@10':>8s}  {'R@20':>8s}  {'NDCG@5':>8s}  {'NDCG@10':>8s}  {'NDCG@20':>8s}")
    print("  " + "-" * 105)

    for label, (res, alpha) in method_bests.items():
        a_str = f"{alpha:.1f}" if alpha is not None else "auto"
        print(f"  {label:<25s}  {a_str:>5s}  {res['MRR']:>8.4f}  {res['R@1']:>8.4f}  "
              f"{res.get('R@5',0):>8.4f}  {res['R@10']:>8.4f}  {res['R@20']:>8.4f}  "
              f"{res.get('NDCG@5',0):>8.4f}  {res.get('NDCG@10',0):>8.4f}  "
              f"{res.get('NDCG@20',0):>8.4f}")

    print("  " + "-" * 105)

    # Top-5 modality 분포
    print(f"\n  === Top-5 결과의 Modality 분포 (% / 건수) ===")
    print(f"  {'Method':<25s}  {'α':>5s}  {'text':>14s}  {'image':>14s}  {'both':>14s}")
    print("  " + "-" * 80)
    for label, (res, alpha) in method_bests.items():
        a_str = f"{alpha:.1f}" if alpha is not None else "auto"
        if "top5_text_ratio" in res:
            t_pct = res["top5_text_ratio"] * 100
            i_pct = res["top5_image_ratio"] * 100
            b_pct = res["top5_both_ratio"] * 100
            t_cnt = int(res["top5_text_count"])
            i_cnt = int(res["top5_image_count"])
            b_cnt = int(res["top5_both_count"])
            print(f"  {label:<25s}  {a_str:>5s}  "
                  f"{t_pct:>5.1f}% ({t_cnt:>5d})  "
                  f"{i_pct:>5.1f}% ({i_cnt:>5d})  "
                  f"{b_pct:>5.1f}% ({b_cnt:>5d})")
        else:
            print(f"  {label:<25s}  {a_str:>5s}  {'N/A':>14s}  {'N/A':>14s}  {'N/A':>14s}")
    print("  " + "-" * 80)

    # 개선율
    if "1. Raw" in method_bests and "5. Z-Score+Adaptive" in method_bests:
        raw_mrr = method_bests["1. Raw"][0]["MRR"]
        best_mrr = method_bests["5. Z-Score+Adaptive"][0]["MRR"]
        gr_mrr = method_bests.get("2. GR-CLIP", ({}, None))[0].get("MRR", 0)
        print(f"\n  vs Raw:     MRR +{(best_mrr - raw_mrr):.4f} ({(best_mrr/raw_mrr - 1)*100:+.1f}%)")
        if gr_mrr:
            print(f"  vs GR-CLIP: MRR +{(best_mrr - gr_mrr):.4f} ({(best_mrr/gr_mrr - 1)*100:+.1f}%)")


def print_cross_dataset_table(all_dataset_results):
    """데이터셋 간 비교 테이블"""
    subsets = list(all_dataset_results.keys())
    all_methods = ["1. Raw", "2. GR-CLIP", "3. Global Z-Score",
                   "4. Adaptive (score)", "5. Z-Score+Adaptive"]
    # 실제로 실행된 method만 표시 (--alpha 사용 시 4~5번 제외)
    has_adaptive = any("adaptive" in r or "zscore+adaptive" in r
                       for r in all_dataset_results.values())
    methods = all_methods if has_adaptive else all_methods[:3]

    # 각 데이터셋의 method_bests 추출
    dataset_bests = {}
    for subset, results in all_dataset_results.items():
        bests = {}
        for prefix, label in [("raw_", "1. Raw"),
                               ("gr_clip_", "2. GR-CLIP"),
                               ("zscore_", "3. Global Z-Score")]:
            keys = [k for k in results if k.startswith(prefix)]
            if keys:
                best_key = max(keys, key=lambda k: results[k]["MRR"])
                bests[label] = results[best_key]
        if "adaptive" in results:
            bests["4. Adaptive (score)"] = results["adaptive"]
        if "zscore+adaptive" in results:
            bests["5. Z-Score+Adaptive"] = results["zscore+adaptive"]
        dataset_bests[subset] = bests

    # MRR 비교표
    print(f"\n{'='*90}")
    print(f"  전체 데이터셋 × 방법론 MRR 비교 (ViT-L/14)")
    print(f"{'='*90}")

    header = f"  {'Method':<25s}"
    for subset in subsets:
        header += f"  {subset:>15s}"
    print(header)
    print("  " + "-" * (25 + 17 * len(subsets)))

    for method in methods:
        row = f"  {method:<25s}"
        for subset in subsets:
            val = dataset_bests[subset].get(method, {}).get("MRR", 0)
            row += f"  {val:>15.4f}"
        print(row)

    print("  " + "-" * (25 + 17 * len(subsets)))

    # NDCG@10 비교표
    print(f"\n  전체 데이터셋 × 방법론 NDCG@10 비교")
    print("  " + "-" * (25 + 17 * len(subsets)))

    header = f"  {'Method':<25s}"
    for subset in subsets:
        header += f"  {subset:>15s}"
    print(header)
    print("  " + "-" * (25 + 17 * len(subsets)))

    for method in methods:
        row = f"  {method:<25s}"
        for subset in subsets:
            val = dataset_bests[subset].get(method, {}).get("NDCG@10", 0)
            row += f"  {val:>15.4f}"
        print(row)

    print("  " + "-" * (25 + 17 * len(subsets)))

    # 개선율 요약 (4~5번이 실행된 경우에만)
    if has_adaptive:
        print(f"\n  Z-Score+Adaptive 개선율 (vs GR-CLIP)")
        print("  " + "-" * (25 + 17 * len(subsets)))
        row_mrr = f"  {'MRR 개선율':<25s}"
        row_ndcg = f"  {'NDCG@10 개선율':<25s}"
        for subset in subsets:
            gr = dataset_bests[subset].get("2. GR-CLIP", {}).get("MRR", 0)
            zsa = dataset_bests[subset].get("5. Z-Score+Adaptive", {}).get("MRR", 0)
            gr_n = dataset_bests[subset].get("2. GR-CLIP", {}).get("NDCG@10", 0)
            zsa_n = dataset_bests[subset].get("5. Z-Score+Adaptive", {}).get("NDCG@10", 0)
            if gr > 0:
                row_mrr += f"  {(zsa/gr - 1)*100:>+14.1f}%"
                row_ndcg += f"  {(zsa_n/gr_n - 1)*100:>+14.1f}%"
            else:
                row_mrr += f"  {'N/A':>15s}"
                row_ndcg += f"  {'N/A':>15s}"
        print(row_mrr)
        print(row_ndcg)


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["MSCOCO", "VisualNews", "Google_WIT"])
    ap.add_argument("--all", action="store_true", help="3개 데이터셋 전부 실행")
    ap.add_argument("--inspect", type=int, default=0,
                    help="N개 샘플 쿼리의 top-5 modality 분포 검사 (0=비활성)")
    ap.add_argument("--alpha", type=float, default=None,
                    help="특정 α로 고정 (1~3번만 실행, α sweep 및 4~5번 생략)")
    args = ap.parse_args()

    if not args.all and not args.subset:
        ap.error("--subset 또는 --all 중 하나를 지정하세요")

    if args.alpha is not None and not (0.0 <= args.alpha <= 1.0):
        ap.error("--alpha는 0.0~1.0 범위여야 합니다")

    alphas = [args.alpha] if args.alpha is not None else ALPHAS
    skip_adaptive = args.alpha is not None

    np.random.seed(SEED)

    RESULT_DIR = Path(__file__).parent.parent / "MixBench_results"
    RESULT_DIR.mkdir(exist_ok=True, parents=True)

    subsets = ["MSCOCO", "VisualNews", "Google_WIT"] if args.all else [args.subset]
    all_dataset_results = {}

    for subset in subsets:
        print(f"\n{'#'*80}")
        print(f"  데이터셋: {subset}")
        print(f"{'#'*80}")

        results = run_all_methods(subset, inspect_n=args.inspect,
                                  alphas=alphas, skip_adaptive=skip_adaptive)
        all_dataset_results[subset] = results

        # 데이터셋별 결과 출력
        print_results(subset, results)

        # 저장
        save_results = {}
        for k, v in results.items():
            save_results[k] = {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float))}
        out_dir = RESULT_DIR / subset
        out_dir.mkdir(exist_ok=True)
        out_name = (f"full_comparison_a{args.alpha:.1f}.json"
                    if args.alpha is not None else "full_comparison.json")
        with open(out_dir / out_name, "w") as f:
            json.dump(save_results, f, indent=2)
        print(f"\n저장: {out_dir / out_name}")

    # 전체 데이터셋 비교 (--all일 때)
    if len(all_dataset_results) > 1:
        print_cross_dataset_table(all_dataset_results)

        # 전체 결과 저장
        with open(RESULT_DIR / "cross_dataset_comparison.json", "w") as f:
            summary = {}
            for subset, results in all_dataset_results.items():
                summary[subset] = {}
                for k, v in results.items():
                    summary[subset][k] = {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float))}
            json.dump(summary, f, indent=2)
        print(f"\n전체 저장: {RESULT_DIR / 'cross_dataset_comparison.json'}")


if __name__ == "__main__":
    main()