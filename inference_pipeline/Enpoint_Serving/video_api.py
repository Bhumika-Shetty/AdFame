from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
import torch
import uuid
import asyncio
import os
from diffsynth import ModelManager, WanVideoPipeline, save_video

# --- FastAPI app setup ---
app = FastAPI(
    title="Text-to-Video Generation API",
    description="API for generating videos from text prompts using Wan2.1-T2V-1.3B",
    version="1.0.0"
)
Instrumentator().instrument(app).expose(app)

# --- Request/Response Models ---
class VideoRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for video generation.")

class VideoResponse(BaseModel):
    video_path: str = Field(..., description="Path to the generated video file.")

# --- GPU serialization lock ---
gpu_lock = asyncio.Lock()

# --- Model and pipeline setup (load once at startup) ---
MODEL_PATHS = [
    "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
    "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
    "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
]
LORA_PATH = "./safe_tensors/adapter_model.safetensors"
NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, "
    "worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, "
    "deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
)

# Use GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
model_manager.load_models(MODEL_PATHS)
model_manager.load_lora(LORA_PATH, lora_alpha=1.0)
pipe = WanVideoPipeline.from_model_manager(model_manager, device=DEVICE)
pipe.enable_vram_management(num_persistent_param_in_dit=None)

# --- Video generation endpoint ---
@app.post("/generate", response_model=VideoResponse)
async def generate_video(request: VideoRequest):
    async with gpu_lock:
        try:
            prompt = request.prompt
            # Generate a unique filename
            safe_prompt = prompt[:10].replace(' ', '_')
            unique_id = str(uuid.uuid4())[:8]
            video_dir = "saved_videos"
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, f"video_{safe_prompt}_{unique_id}.mp4")

            # Run inference
            video = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=50,
                seed=0,
                tiled=True,
                tea_cache_l1_thresh=0.05,
                tea_cache_model_id="Wan2.1-T2V-1.3B",
            )
            save_video(video, video_path, fps=30, quality=5)
            return VideoResponse(video_path=video_path)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Video generation error: {str(e)}") 