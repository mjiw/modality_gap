"""
VLM2Vec 인코딩 스크립트 — 별도 env (`vlm2vec`) 에서 실행.

GR-CLIP 논문 Table 1 의 VLM2Vec(LLaVaNext), VLM2Vec(Qwen) 행을 재현하기 위해
MixBench2026 의 모든 corpus / query 항목을 VLM2Vec embedding 으로 변환하여
`cache_{subset}_VLM2Vec_{LN|QW}/` 에 numpy 캐시로 저장.

저장된 캐시는 본 프로젝트 메인 env (tablerag) 에서 다음 스크립트가 읽어 활용:
    python mixbench_backbone_comparison.py --include-vlm2vec

GR-CLIP 의 vlm_example/embedding.py (https://github.com/yuhui-zh15/MixedModalitySearch)
를 그대로 따라 instruction template 4 종을 사용:
  - query (text):       "Retrieve a relevant item that represents: {caption}\n"
  - corpus text-only:   caption alone
  - corpus image-only:  "{image_token}\nRepresent the given image\n"
  - corpus multimodal:  "{image_token}\nRepresent the given image with related text: {caption}\n"

전제:
  - VLM2Vec 공식 repo 가 ../VLM2Vec/ 에 clone 되어 있고 src/ import 가능
  - 의존성: transformers==4.52.3, peft, qwen-vl-utils (Qwen 변형용), torch, datasets

실행:
  conda activate vlm2vec
  python encode_vlm2vec.py --backbone LN --subset MSCOCO
  python encode_vlm2vec.py --backbone QW --subset all
  python encode_vlm2vec.py --backbone LN --subset all       # 4 dataset 전부
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────
# flash_attn stub — Windows 에서 flash_attn 컴파일 불가.
# VLM2Vec 의 InternVideo2 backbone 이 모듈 레벨에서 flash_attn 을 import 하지만,
# 우리가 쓰는 LLaVA-Next / Qwen2-VL 경로는 use_flash_attn=False 라 실행은 안 됨.
# import 시점에 이름만 존재하면 통과되므로 fake module 로 우회.
# ─────────────────────────────────────────────
def _install_flash_attn_stub():
    if "flash_attn" in sys.modules:
        return
    import importlib.util
    def _mk(name):
        m = types.ModuleType(name)
        # transformers 가 find_spec 으로 검사하므로 __spec__ 필수.
        m.__spec__ = importlib.util.spec_from_loader(name, loader=None)
        return m
    fake = _mk("flash_attn")
    fake.bert_padding = _mk("flash_attn.bert_padding")
    fake.flash_attn_interface = _mk("flash_attn.flash_attn_interface")
    def _err(*_a, **_kw):
        raise RuntimeError("flash_attn was stubbed at import time — "
                           "called at runtime means an unsupported code path.")
    fake.bert_padding.unpad_input = _err
    fake.bert_padding.pad_input = _err
    fake.flash_attn_interface.flash_attn_varlen_qkvpacked_func = _err
    # transformers.is_flash_attn_2_available() 는 추가로 version 도 검사 → False 반환되게.
    fake.__version__ = "0.0.0"
    sys.modules["flash_attn"] = fake
    sys.modules["flash_attn.bert_padding"] = fake.bert_padding
    sys.modules["flash_attn.flash_attn_interface"] = fake.flash_attn_interface

_install_flash_attn_stub()


# ─────────────────────────────────────────────
# skimage stub — scikit-image 가 numpy 2.x 를 요구해서 torch 2.4 (numpy 1.x 빌드)
# 와 충돌. VLM2Vec 의 qwen2_vl_tokenselection / qwen2_5_vl_tokenselection backbone 만
# mark_boundaries 를 사용 — 우리가 쓰는 llava_next / qwen2_vl backbone 은 호출 안 함.
# import 시점에 이름만 통과시키면 됨.
# ─────────────────────────────────────────────
def _install_skimage_stub():
    if "skimage" in sys.modules:
        return
    import importlib.util
    def _mk(name):
        m = types.ModuleType(name)
        m.__spec__ = importlib.util.spec_from_loader(name, loader=None)
        return m
    fake = _mk("skimage")
    fake.segmentation = _mk("skimage.segmentation")
    def _err(*_a, **_kw):
        raise RuntimeError("skimage was stubbed at import time — "
                           "called means qwen2_vl_tokenselection code path entered "
                           "(should not happen for llava_next / qwen2_vl backbones).")
    fake.segmentation.mark_boundaries = _err
    fake.__version__ = "0.0.0"
    sys.modules["skimage"] = fake
    sys.modules["skimage.segmentation"] = fake.segmentation

_install_skimage_stub()


# ─────────────────────────────────────────────
# datasets / pyarrow 를 VLM2Vec 모델 로드 전에 먼저 import.
# datasets 4.x 는 pyarrow C extension 을 로드하는데, 이를 VLM2Vec (torch + transformers
# + PEFT 등 무거운 스택) 이 메모리에 자리 잡은 뒤에 import 하면 C-level segfault 가 발생.
# 미리 import 해서 C 라이브러리 초기화 순서를 정리.
# ─────────────────────────────────────────────
import datasets as _datasets_preload  # noqa: F401
import pyarrow as _pyarrow_preload    # noqa: F401


# 동일 디렉토리의 data_loader 에서 데이터 로드 헬퍼만 재사용
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# VLM2Vec 공식 repo (../VLM2Vec/) 를 import path 에 추가
_VLM2VEC_REPO = _HERE.parent / "VLM2Vec"
if not _VLM2VEC_REPO.exists():
    raise RuntimeError(
        f"VLM2Vec 공식 repo 가 {_VLM2VEC_REPO} 에 없습니다.\n"
        f"다음 명령으로 clone 하세요:\n"
        f"    cd {_HERE.parent}\n"
        f"    git clone https://github.com/TIGER-AI-Lab/VLM2Vec.git"
    )
sys.path.insert(0, str(_VLM2VEC_REPO))


# ─────────────────────────────────────────────
# Backbone 설정 (data_loader.BACKBONES 와 일치)
# ─────────────────────────────────────────────
BACKBONE_CFG = {
    "LN": {
        "tag": "VLM2Vec_LN",
        "model_name": "TIGER-Lab/VLM2Vec-LLaVa-Next",
        "model_backbone": "llava_next",
        "checkpoint_path": None,
        "lora": False,
        "image_token": "<image>",
        "display": "VLM2Vec(LLaVaNext)",
        "dim": 4096,  # llava-v1.6-mistral-7b hidden dim (data_loader.BACKBONES 와 일치)
    },
    "QW": {
        "tag": "VLM2Vec_QW",
        "model_name": "Qwen/Qwen2-VL-7B-Instruct",
        "model_backbone": "qwen2_vl",
        "checkpoint_path": "TIGER-Lab/VLM2Vec-Qwen2VL-7B",
        "lora": True,
        "image_token": "<|image_pad|>",
        "display": "VLM2Vec(Qwen)",
        "dim": 3584,  # Qwen2-VL-7B hidden dim
    },
}

SUBSETS = ["MSCOCO", "VisualNews", "Google_WIT", "OVEN"]
IMAGE_SIZE = 1344  # GR-CLIP vlm_example 과 동일


# ─────────────────────────────────────────────
# VLM2Vec 모델 로드 (공식 MMEBModel)
# ─────────────────────────────────────────────
def load_vlm2vec(cfg):
    import torch
    from src.model.model import MMEBModel
    from src.arguments import ModelArguments
    from src.model.processor import load_processor

    print(f"[VLM2Vec 로드] {cfg['display']} ({cfg['model_name']})")
    kwargs = dict(
        model_name=cfg["model_name"],
        pooling="last",
        normalize=True,
        model_backbone=cfg["model_backbone"],
    )
    if cfg["checkpoint_path"]:
        kwargs["checkpoint_path"] = cfg["checkpoint_path"]
    if cfg["lora"]:
        kwargs["lora"] = True

    model_args = ModelArguments(**kwargs)
    processor = load_processor(model_args)

    # VLM2Vec/src/model/processor.py:121-134 의 load_processor 가 QW backbone 에서
    # data_args=None 이면 size={"shortest_edge": None, "longest_edge": None} 를 강제로
    # from_pretrained 에 주입해 preprocessor_config.json 의 정상 min/max_pixels 를
    # 덮어쓰는 버그가 있음 (transformers>=4.52 에서 발현). 체크포인트에 저장된 값으로 복원.
    if cfg["model_backbone"] == "qwen2_vl":
        ip = processor.image_processor
        if ip.size.get("longest_edge") is None or ip.size.get("shortest_edge") is None:
            ip.min_pixels = 3136        # VLM2Vec-Qwen2VL-7B preprocessor_config.json
            ip.max_pixels = 12_845_056  # = 3584x3584 px 한도
            ip.size = {"shortest_edge": ip.min_pixels, "longest_edge": ip.max_pixels}

    model = MMEBModel.load(model_args)
    model = model.to("cuda", dtype=torch.bfloat16).eval()
    return model, processor


# ─────────────────────────────────────────────
# Encode 한 항목 (GR-CLIP vlm_example/embedding.py 패턴)
# ─────────────────────────────────────────────
def encode_one(model, processor, *, text=None, image=None):
    """GR-CLIP vlm_example/embedding.py 의 encode() 와 1:1 동일 구조."""
    import torch
    from PIL import Image as PILImage

    if image is not None:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), PILImage.BICUBIC)

    # QW backbone 전용 경로: VLM2Vec fork 의 modeling_qwen2_vl.py:1633-1646 가
    # pixel_values / image_grid_thw 를 batch element 별 list (원소=텐서) 로 받도록 패치돼
    # 있어서, processor raw 출력 (shape=(1,3) 텐서) 을 그대로 넘기면 rot_pos_emb 에서
    # 0-d iteration 에러 발생. 공식 Qwen2_VL_process_fn 으로 포맷을 맞춰서 통과.
    # LN backbone 은 아래 else 그대로 (기존 동작 유지).
    if getattr(model, "model_backbone", None) == "qwen2_vl":
        from src.model.processor import Qwen2_VL_process_fn
        model_inputs = {
            "text": [text],
            "images": [image] if image is not None else [None],
        }
        inputs = Qwen2_VL_process_fn(model_inputs, processor, max_length=None)
        inputs = {
            k: (v.to("cuda") if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }
    else:
        inputs = processor(text=text, images=image, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        reps = model(tgt=inputs)["tgt_reps"]
    return reps.squeeze().float().cpu().numpy()


# ─────────────────────────────────────────────
# 데이터셋 로드 (data_loader.py 와 동일한 분기)
# ─────────────────────────────────────────────
def load_raw_dataset(subset):
    """모든 dataset 의 raw items 를 (ds_q_list, ds_c_list, modes) 형태로 통일.
    data_loader.load_mixbench 의 데이터 로딩 부분만 발췌 (인코딩 제외).
    """
    print(f"[DEBUG] load_raw_dataset({subset}) 진입", flush=True)
    from datasets import load_dataset
    print(f"[DEBUG] datasets import OK", flush=True)
    from data_loader import _load_wit_as_mixbench
    print(f"[DEBUG] data_loader import OK", flush=True)

    if subset == "Google_WIT":
        print(f"[Google_WIT] wikimedia/wit_base 에서 로드 ...", flush=True)
        ds_q_list, ds_c_list, modes = _load_wit_as_mixbench()
        print(f"[DEBUG] _load_wit_as_mixbench OK", flush=True)
        return ds_q_list, ds_c_list, modes

    print(f"[{subset}] MixBench2026 로드 ...")
    ds_q = load_dataset("mixed-modality-search/MixBench2026", name=subset, split="queries")
    ds_c = load_dataset("mixed-modality-search/MixBench2026", name=subset, split="mixed_corpus")
    ds_q_list = [dict(ds_q[i]) for i in range(len(ds_q))]
    ds_c_list = [dict(ds_c[i]) for i in range(len(ds_c))]
    modes = []
    for c in ds_c_list:
        has_text = c.get("text") is not None
        has_image = c.get("image") is not None
        if has_text and has_image:
            modes.append("both")
        elif has_text:
            modes.append("text")
        else:
            modes.append("image")
    return ds_q_list, ds_c_list, modes


# ─────────────────────────────────────────────
# Main encoding loop — 한 (subset, backbone) 조합
# ─────────────────────────────────────────────
def encode_subset(model, processor, cfg, subset, batch_size=1):
    """corpus + query 를 모두 VLM2Vec 임베딩으로 변환하여 캐시 디렉토리에 저장.

    저장 형식 (data_loader.load_mixbench 와 호환):
      corpus_text_emb.npy  — N×D, "text"/"both" 항목에 임베딩 (나머지 0)
      corpus_image_emb.npy — N×D, "image"/"both" 항목에 임베딩 (나머지 0)
      query_emb.npy        — Q×D
      modes.json, q_gt.json — 평가 메타데이터
    """
    import torch
    from tqdm import tqdm

    print(f"[DEBUG] encode_subset 진입: {subset}", flush=True)
    cache_dir = _HERE / "cache" / f"cache_{subset}_{cfg['tag']}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DEBUG] cache_dir 생성/확인: {cache_dir}", flush=True)

    # 옵션 B: 6개 파일 (mixed_emb 포함)
    expected = ["corpus_text_emb.npy", "corpus_image_emb.npy",
                "corpus_mixed_emb.npy",  # 옵션 B 추가
                "query_emb.npy", "modes.json", "q_gt.json"]
    if all((cache_dir / f).exists() for f in expected):
        print(f"[{subset}/{cfg['display']}] 캐시 (6 파일) 이미 존재 → 스킵", flush=True)
        return

    # Incremental 모드 판단: 기존 5개 파일 있고 mixed_emb 만 없는 경우
    basic_files = ["corpus_text_emb.npy", "corpus_image_emb.npy",
                   "query_emb.npy", "modes.json", "q_gt.json"]
    basic_done = all((cache_dir / f).exists() for f in basic_files)
    mixed_done = (cache_dir / "corpus_mixed_emb.npy").exists()
    incremental = basic_done and not mixed_done
    if incremental:
        print(f"[{subset}/{cfg['display']}] basic 5 files 존재 → mixed_emb 만 추가 인코딩 ('both' 항목만)", flush=True)

    print(f"[DEBUG] load_raw_dataset 호출 직전 ({subset})", flush=True)
    ds_q, ds_c, modes = load_raw_dataset(subset)
    print(f"[DEBUG] load_raw_dataset 반환 OK: ds_c={len(ds_c)}, ds_q={len(ds_q)}", flush=True)
    print(f"  corpus={len(ds_c)} (text={modes.count('text')}, "
          f"image={modes.count('image')}, both={modes.count('both')})", flush=True)
    print(f"  queries={len(ds_q)}", flush=True)

    dim = cfg["dim"]
    print(f"  embedding dim = {dim} (cfg)", flush=True)

    N = len(ds_c)
    Q = len(ds_q)

    image_token = cfg["image_token"]
    image_only_prompt_tmpl = f"{image_token}\nRepresent the given image\n"
    # ★ 옵션 B 의 mixed 프롬프트 — GR-CLIP vlm_example/embedding.py 와 정확히 동일 문구
    mixed_prompt_tmpl = f"{image_token}\nRepresent the given image with related text information: {{caption}}\n"

    if incremental:
        # ── Incremental: 기존 text/image/query 그대로, 'both' 항목 mixed_emb 만 추가 ──
        full_text_emb  = np.load(cache_dir / "corpus_text_emb.npy")
        full_image_emb = np.load(cache_dir / "corpus_image_emb.npy")
        query_emb_full = np.load(cache_dir / "query_emb.npy")
        with open(cache_dir / "q_gt.json") as f:
            q_gt = json.load(f)
        full_mixed_emb = np.zeros((N, dim), dtype="float32")

        both_indices = [i for i, m in enumerate(modes) if m == "both"]
        print(f"  mixed_emb 인코딩 ({len(both_indices)} 'both' 항목)", flush=True)
        for i in tqdm(both_indices, desc=f"{subset}/{cfg['tag']} mixed"):
            c_text  = ds_c[i].get("text")
            c_image = ds_c[i].get("image")
            mixed_prompt = mixed_prompt_tmpl.format(caption=c_text)
            full_mixed_emb[i] = encode_one(model, processor,
                                           text=mixed_prompt, image=c_image)
    else:
        # ── Full encoding: text/image/mixed 모두 처음부터 ──
        full_text_emb  = np.zeros((N, dim), dtype="float32")
        full_image_emb = np.zeros((N, dim), dtype="float32")
        full_mixed_emb = np.zeros((N, dim), dtype="float32")

        print(f"  corpus 인코딩 시작 (template: image_token={image_token!r}) ...", flush=True)
        print(f"  - text mode: text-only prompt (1 encoding)", flush=True)
        print(f"  - image mode: image-only prompt (1 encoding)", flush=True)
        print(f"  - both mode: text-only + image-only + mixed (3 encodings, 옵션 B)", flush=True)
        for i in tqdm(range(N), desc=f"{subset}/{cfg['tag']} corpus"):
            m = modes[i]
            c_text  = ds_c[i].get("text")
            c_image = ds_c[i].get("image")

            if m == "text":
                full_text_emb[i] = encode_one(model, processor, text=c_text)
            elif m == "image":
                full_image_emb[i] = encode_one(model, processor,
                                               text=image_only_prompt_tmpl, image=c_image)
            else:  # both
                # paired (SVD 입력용): text-only + image-only
                full_text_emb[i]  = encode_one(model, processor, text=c_text)
                full_image_emb[i] = encode_one(model, processor,
                                               text=image_only_prompt_tmpl, image=c_image)
                # mixed (raw 'both' retrieval 용): image + caption 함께
                mixed_prompt = mixed_prompt_tmpl.format(caption=c_text)
                full_mixed_emb[i] = encode_one(model, processor,
                                               text=mixed_prompt, image=c_image)

        # ── Query 인코딩 ──
        query_has_image = (
            subset == "OVEN"
            and len(ds_q) > 0
            and ds_q[0].get("image") is not None
        )
        query_emb_full = np.zeros((Q, dim), dtype="float32")
        print(f"  query 인코딩 시작 (text-only template)" +
              (" + image (OVEN)" if query_has_image else "") + " ...", flush=True)
        for qi in tqdm(range(Q), desc=f"{subset}/{cfg['tag']} queries"):
            q_text = ds_q[qi]["text"]
            if query_has_image:
                qimg = ds_q[qi]["image"]
                qprompt = f"{image_token}\nRetrieve a relevant item that represents: {q_text}\n"
                emb = encode_one(model, processor, text=qprompt, image=qimg)
            else:
                qprompt = f"Retrieve a relevant item that represents: {q_text}\n"
                emb = encode_one(model, processor, text=qprompt)
            query_emb_full[qi] = emb

        # ── GT 매핑 ──
        c_id_to_idx = {str(ds_c[i].get("id", i)): i for i in range(N)}
        q_gt = []
        valid_q = []
        for qi in range(Q):
            qid = str(ds_q[qi].get("id", qi))
            if qid in c_id_to_idx:
                q_gt.append(c_id_to_idx[qid])
                valid_q.append(qi)
        query_emb_full = query_emb_full[valid_q]

    # ── Atomic 저장: 모두 임시 폴더에 쓰고 마지막에 rename ──
    tmp_dir = cache_dir.parent / f".{cache_dir.name}.tmp"
    if tmp_dir.exists():
        import shutil
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()

    np.save(tmp_dir / "corpus_text_emb.npy",  full_text_emb)
    np.save(tmp_dir / "corpus_image_emb.npy", full_image_emb)
    np.save(tmp_dir / "corpus_mixed_emb.npy", full_mixed_emb)  # ★ 옵션 B 신규
    np.save(tmp_dir / "query_emb.npy",        query_emb_full)
    with open(tmp_dir / "modes.json", "w") as f:
        json.dump(modes, f)
    with open(tmp_dir / "q_gt.json", "w") as f:
        json.dump(q_gt, f)

    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
    tmp_dir.rename(cache_dir)
    print(f"  ✓ 저장 완료: {cache_dir}", flush=True)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["LN", "QW"], required=True,
                    help="LN=VLM2Vec-LLaVa-Next, QW=VLM2Vec-Qwen2VL-7B")
    ap.add_argument("--subset", default="all",
                    help="MSCOCO / VisualNews / Google_WIT / OVEN / all")
    args = ap.parse_args()

    cfg = BACKBONE_CFG[args.backbone]
    subsets = SUBSETS if args.subset == "all" else [args.subset]

    # 메모리 절약: 모델은 한 번만 로드
    model, processor = load_vlm2vec(cfg)
    for subset in subsets:
        print(f"\n{'#'*70}")
        print(f"  [{subset}] × {cfg['display']}")
        print(f"{'#'*70}")
        encode_subset(model, processor, cfg, subset)


if __name__ == "__main__":
    main()
