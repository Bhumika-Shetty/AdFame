import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData


model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
model_manager.load_models([
    "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
    "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
    "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
])
model_manager.load_lora("./safe_tensors/adapter_model.safetensors", lora_alpha=1.0)
pipe = WanVideoPipeline.from_model_manager(model_manager, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=None)

prompt = "A man wearing adidas shoes"
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=50,
    seed=0, tiled=True,
    # TeaCache parameters
    tea_cache_l1_thresh=0.05, # The larger this value is, the faster the speed, but the worse the visual quality.
    tea_cache_model_id="Wan2.1-T2V-1.3B", # Choose one in (Wan2.1-T2V-1.3B, Wan2.1-T2V-14B, Wan2.1-I2V-14B-480P, Wan2.1-I2V-14B-720P).
)
save_video(video, f"saved_videos/video_{prompt[:10].replace(' ', '_')}.mp4", fps=15, quality=5)
