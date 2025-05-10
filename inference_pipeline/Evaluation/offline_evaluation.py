# offline_evaluation.py (updated – uses TorchMetrics FVD instead of pytorch_fvd)
"""End‑to‑end offline evaluation pipeline for Wan 2.1 text‑to‑video model.

Changes in this version
-----------------------
* Replaced **pytorch_fvd** with **TorchMetrics** implementation of Frechet Video Distance (FVD).
* Added graceful fallback: if TorchMetrics FVD isn't available, the script logs NaN and continues.
* Added installation hint and dynamic import logic.

To install TorchMetrics with video extras:
```
pip install "torchmetrics[image]>=1.3"  # video FVD lives in image/video extras
```

All other logic (prompts, metric aggregation, MLflow logging, registry promotion) remains unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import mlflow
import numpy as np
import torch
from mlflow.tracking import MlflowClient
from skimage.metrics import structural_similarity as ssim

# metric & clip dependencies --------------------------------------------------------------
from piq import brisque  # NIQE removed for brevity; BRISQUE suffices
import clip  # type: ignore
from torchvision import transforms

# try to import FVD from TorchMetrics ------------------------------------------------------
try:
    from torchmetrics.video.frechet_video_distance import FrechetVideoDistance  # TM ≥1.3
    _TM_FVD_AVAILABLE = True
except ModuleNotFoundError:
    _TM_FVD_AVAILABLE = False
    print("⚠️  TorchMetrics with video FVD not found – FVD will be skipped (logged as NaN).")

# Wan 2.1 generation imports --------------------------------------------------------------
from diffsynth import (
    ModelManager,
    WanVideoPipeline,
    VideoData,
    save_video,
)


from modelscope import snapshot_download

import sys
sys.path.append(os.path.dirname(__file__))
from logging_utils import log_all_system_metrics, log_model_info

# Download models
# snapshot_download("Wan-AI/Wan2.1-T2V-1.3B", local_dir="models/Wan-AI/Wan2.1-T2V-1.3B")


# ───────────────────────────────────── CONSTANTS (unchanged) ─────────────────────────────
MODEL_PATHS: Sequence[str] = [
    "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
    "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
    "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
]
LORA_PATH = "/home/cc/yash/AdFame/inference_pipeline/Enpoint_Serving/safe_tensors/adapter_model.safetensors"

STANDARD_PROMPTS = [
    "A man walking through a park",
    "A group of people dancing at a party",
    "A dog running across a field",
]
DOMAIN_PROMPTS = [
    "A woman running in Nike shoes along a beach",
    "A man skateboarding wearing Adidas track pants",
    "A football player in a Nike jersey scoring a goal",
    "A group of runners at a start line in Adidas outfits",
]

NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
    "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, "
    "messy background, three legs, many people in the background, walking backwards"
)

CONFIGS = [
    {"name": "teac_caache_quality", "steps": 50, "tea_l1": 0.05},
    {"name": "teacache0.1_steps50", "steps": 50, "tea_l1": 0.10},
    {"name": "tea_fast_steps20", "steps": 20, "tea_l1": 0.01},
    {"name": "fast_cache02_steps20", "steps": 20, "tea_l1": 0.05},
]

THRESHOLDS = {"fvd_improve": 5.0, "clip_improve": 0.02}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ───────────────────────────────────── Helper functions ──────────────────────────────────

def load_pipeline(device: str = DEVICE) -> WanVideoPipeline:
    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_models(MODEL_PATHS)
    mm.load_lora(LORA_PATH, lora_alpha=1.0)
    pipe = WanVideoPipeline.from_model_manager(mm, device=device)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    return pipe


def generate_video(pipe, prompt: str, steps: int, tea: float, seed: int = 0):
    # g = torch.Generator(device=DEVICE).manual_seed(seed)
    start = time.time()
    vid = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=steps,
        seed=seed,
        tiled=True,
        tea_cache_l1_thresh=tea,
        tea_cache_model_id="Wan2.1-T2V-1.3B",
    )
    return vid, time.time() - start


def video_to_tensor(video: VideoData) -> torch.Tensor:
    """
    Converts a VideoData object to a torch.Tensor of shape (num_frames, 3, H, W) in [0,1] range.
    Handles both PIL.Image and numpy array frames.
    """
    frames = []
    for i in range(len(video)):
        frame = video[i]
        if hasattr(frame, 'convert'):
            # PIL Image
            arr = np.asarray(frame).astype(np.float32) / 255.0
        else:
            # Assume numpy array
            arr = frame.astype(np.float32) / 255.0
        # Ensure channel order is (C, H, W)
        if arr.shape[-1] == 3:
            arr = np.transpose(arr, (2, 0, 1))
        frames.append(torch.from_numpy(arr))
    return torch.stack(frames)


def compute_clip_score(tensor_video: torch.Tensor, prompt: str, clip_model, preprocess) -> float:
    clip_model.eval()
    with torch.no_grad():
        text_f = clip_model.encode_text(clip.tokenize([prompt]).to(DEVICE)).float()
        sims = []
        for fr in tensor_video:
            img = preprocess(transforms.ToPILImage()(fr.cpu())).unsqueeze(0).to(DEVICE)
            img_f = clip_model.encode_image(img).float()
            sims.append(torch.cosine_similarity(img_f, text_f).item())
        return float(np.mean(sims))


def compute_warp_error(tensor_video: torch.Tensor) -> float:
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    errs = []
    for t in range(len(tensor_video) - 1):
        a = (tensor_video[t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        b = (tensor_video[t + 1].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        g1, g2 = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
        fl = flow.calc(g1, g2, None)
        y, x = np.mgrid[:g1.shape[0], :g1.shape[1]]
        warp = cv2.remap(g1, (x + fl[..., 0]).astype(np.float32), (y + fl[..., 1]).astype(np.float32), cv2.INTER_LINEAR)
        errs.append(np.mean(np.abs(warp - g2)) / 255.0)
    return float(np.mean(errs))


def compute_brisque_score(tensor_video: torch.Tensor) -> float:
    return float(np.mean([brisque(fr.unsqueeze(0).to(DEVICE)).item() for fr in tensor_video]))


def compute_fvd_batch(video_batch: torch.Tensor, ref_stats: Dict[str, torch.Tensor]) -> float:
    if not _TM_FVD_AVAILABLE:
        return float("nan")
    fvd = FrechetVideoDistance(feature_extractor="I3D", reset_real_features=False).to(DEVICE)
    # First call: add reference (real) features
    fvd.update(ref_stats["real"], real=True)
    # Second call: add generated batch
    fvd.update(video_batch, real=False)
    return float(fvd.compute().item())

# Reference stats loader (dummy placeholder) --------------------------------------------

def load_reference_videos(n: int = 8, size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
    """In production, load real videos. Here: random noise placeholder."""
    return torch.rand(n, 16, 3, *size)

# Utility: Load video from local path and return VideoData object

def load_video_from_path(video_path: str, height: int = None, width: int = None) -> VideoData:
    """
    Loads a video from a local file path and returns a VideoData object.
    Args:
        video_path (str): Path to the video file.
        height (int, optional): Desired frame height. If None, original height is used.
        width (int, optional): Desired frame width. If None, original width is used.
    Returns:
        VideoData: VideoData object for the loaded video.
    """
    return VideoData(video_file=video_path, height=height, width=width)

# ───────────────────────────────────── Evaluation logic (trimmed for brevity) ────────────

def evaluate_config(pipe, prompts, cfg, clip_model, preprocess):
    metrics = {}
    vids_for_fvd = []
    for p in prompts:
        vid, rt = generate_video(pipe, p, cfg["steps"], cfg["tea_l1"], seed=0)
        tens = video_to_tensor(vid)
        clip_s = compute_clip_score(tens, p, clip_model, preprocess)
        metrics[f"{p[:15]}_clip"] = clip_s
        metrics[f"{p[:15]}_warp"] = compute_warp_error(tens)
        metrics[f"{p[:15]}_brisque"] = compute_brisque_score(tens)
        metrics[f"{p[:15]}_runtime"] = rt
        vids_for_fvd.append(tens.unsqueeze(0))
    metrics["clip_mean"] = float(np.mean([v for k, v in metrics.items() if k.endswith("_clip")]))
    # Compute FVD vs ref videos (if available)
    gen_batch = torch.cat(vids_for_fvd).to(DEVICE)
    ref_batch = load_reference_videos(gen_batch.size(0), gen_batch.shape[-2:]).to(DEVICE)
    if _TM_FVD_AVAILABLE:
        fvd = compute_fvd_batch(gen_batch, {"real": ref_batch})
    else:
        fvd = float("nan")
    metrics["fvd"] = fvd
    return metrics

# Main run (unchanged signature) ---------------------------------------------------------

def run_evaluation(out="eval_outputs"):
    os.makedirs(out, exist_ok=True)
    mlflow.set_experiment("wan_video_eval")
    # Log system metrics before evaluation
    log_all_system_metrics()
    pipe = load_pipeline()
    log_model_info(pipe, model_name="WanVideoPipeline", model_path=str(MODEL_PATHS), device=DEVICE)
    clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    log_model_info(clip_model, model_name="CLIP ViT-B/32", device=DEVICE)
    with mlflow.start_run(run_name=datetime.now().strftime("eval_%Y%m%d_%H%M%S")):
        for cfg in CONFIGS:
            m = evaluate_config(pipe, STANDARD_PROMPTS + DOMAIN_PROMPTS, cfg, clip_model, preprocess)
            mlflow.log_metrics({f"{cfg['name']}_{k}": v for k, v in m.items()})
    print("✅ Evaluation run complete – results in MLflow UI.")

# ───────────────────────────── Single Local Video Evaluation ─────────────────────────────

def evaluate_local_video(video_path: str, prompt: str, clip_model, preprocess, height: int = None, width: int = None):
    """
    Evaluate a single local video file using the same metrics as the main pipeline.
    Args:
        video_path (str): Path to the local video file.
        prompt (str): Text prompt for CLIP score.
        clip_model: Loaded CLIP model.
        preprocess: CLIP preprocessing function.
        height (int, optional): Desired frame height for resizing.
        width (int, optional): Desired frame width for resizing.
    Returns:
        dict: Metrics for the video.
    """
    # Log system metrics before evaluation
    log_all_system_metrics()
    log_model_info(clip_model, model_name="CLIP ViT-B/32", device=DEVICE)
    metrics = {}
    vid = load_video_from_path(video_path, height=height, width=width)
    tens = video_to_tensor(vid)
    clip_s = compute_clip_score(tens, prompt, clip_model, preprocess)
    metrics[f"{prompt[:15]}_clip"] = clip_s
    metrics[f"{prompt[:15]}_warp"] = compute_warp_error(tens)
    metrics[f"{prompt[:15]}_brisque"] = compute_brisque_score(tens)
    metrics[f"{prompt[:15]}_runtime"] = float('nan')
    metrics["clip_mean"] = float(np.mean([v for k, v in metrics.items() if k.endswith("_clip")]))
    vids_for_fvd = [tens.unsqueeze(0)]
    gen_batch = torch.cat(vids_for_fvd).to(DEVICE)
    ref_batch = load_reference_videos(gen_batch.size(0), gen_batch.shape[-2:]).to(DEVICE)
    if _TM_FVD_AVAILABLE:
        fvd = compute_fvd_batch(gen_batch, {"real": ref_batch})
    else:
        fvd = float("nan")
    metrics["fvd"] = fvd
    return metrics

# Example config for local video evaluation
LOCAL_VIDEO_EVAL_CONFIG = {
    "video_path": "/home/cc/yash/AdFame/inference_pipeline/Enpoint_Serving/saved_videos/video_Adidas_Sho_f5b97dca.mp4",
    "prompt": "A man wearing Adidas track pants and shoes and running",
    "height": None,  # or set to e.g. 256
    "width": None,   # or set to e.g. 256
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="eval_outputs")
    parser.add_argument("--local_video", action="store_true", help="Run evaluation on a single local video")
    parser.add_argument("--video_path", type=str, default=None, help="Path to the local video file")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for CLIP score")
    parser.add_argument("--height", type=int, default=None, help="Frame height for resizing")
    parser.add_argument("--width", type=int, default=None, help="Frame width for resizing")
    args = parser.parse_args()

    if args.local_video:
        clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
        video_path = args.video_path if args.video_path is not None else LOCAL_VIDEO_EVAL_CONFIG["video_path"]
        prompt = args.prompt if args.prompt is not None else LOCAL_VIDEO_EVAL_CONFIG["prompt"]
        height = args.height if args.height is not None else LOCAL_VIDEO_EVAL_CONFIG["height"]
        width = args.width if args.width is not None else LOCAL_VIDEO_EVAL_CONFIG["width"]
        metrics = evaluate_local_video(
            video_path,
            prompt,
            clip_model,
            preprocess,
            height=height,
            width=width
        )
        print("Single local video evaluation metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v}")
    else:
        run_evaluation(args.out)
