# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import numpy as np
import os
import torch
import uuid

# Triton Python Model
import triton_python_backend_utils as pb_utils

# DiffSynth specific imports
from diffsynth import ModelManager, WanVideoPipeline, save_video

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have an `initialize` method and an `execute`
    method.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` is optional. This function allows the
        model to intialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self.model_config = json.loads(args["model_config"])

        # Get GPGPU device ID
        if args["model_instance_kind"] == "GPU":
            self.device_id = int(args["model_instance_device_id"])
            self.device = f"cuda:{self.device_id}"
        else:
            self.device = "cpu"

        # Model and pipeline setup
        # Paths should be relative to the model version directory or absolute if models are centrally located.
        # For now, assume they are in a 'model_files' subdirectory within the version dir.
        model_version_dir = os.path.join(args["model_repository"], args["model_version"])
        
        # Define model paths relative to the model version directory
        # These paths need to be adjusted based on where the actual model files are placed
        # within the Triton model repository structure.
        # Example: model_files/Wan-AI/Wan2.1-T2V-1.3B/...
        self.model_files_dir = os.path.join(model_version_dir, "model_files")

        # Check if model files exist, if not, this will fail. 
        # In a real scenario, these might be downloaded here or pre-packaged.
        # For now, we assume they are present in self.model_files_dir structure.
        model_paths_relative = [
            "Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
            "Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
            "Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
        ]
        self.model_paths = [os.path.join(self.model_files_dir, p) for p in model_paths_relative]
        self.lora_path = os.path.join(self.model_files_dir, "safe_tensors/adapter_model.safetensors")

        # Ensure model directories exist before trying to load (or handle download here)
        # For this example, we assume files are already in place.
        # For example, the Dockerfile for Triton would need to ensure these files are copied
        # into the model repository structure, or this initialize script downloads them.

        # Default negative prompt from the original video_api.py
        self.default_negative_prompt = (
            "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, "
            "worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, "
            "deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        )
        self.default_num_inference_steps = 50
        self.default_seed = 0

        # Output directory for videos - this should be a shared volume if FastAPI needs to access files directly
        # Or Triton could return the video bytes (more complex for large files)
        self.video_output_dir = "/shared_videos/triton_generated" # Example path, configure as shared volume
        os.makedirs(self.video_output_dir, exist_ok=True)

        try:
            model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu") # Load on CPU first
            model_manager.load_models(self.model_paths)
            if os.path.exists(self.lora_path):
                 model_manager.load_lora(self.lora_path, lora_alpha=1.0)
            else:
                print(f"LORA file not found at {self.lora_path}, skipping LORA loading.")

            self.pipe = WanVideoPipeline.from_model_manager(model_manager, device=self.device)
            self.pipe.enable_vram_management(num_persistent_param_in_dit=None)
            print("DiffSynth Model initialized successfully on device: ", self.device)
        except Exception as e:
            print(f"Error initializing DiffSynth model: {e}")
            raise e

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model.
        `execute` function receives a list of pb_utils.InferenceRequest as
        input. Each InferenceRequest has parameters:
          * request_id: The ID of the request, generated by Triton.
          * inputs: A list of pb_utils.Tensor
          * requested_output_names: A list of strings containing the names of
            the requested outputs
          * correlation_id: The correlation ID of the request. This ID will
            be zero if the request does not have a correlation ID.
        After processing a group of requests, `execute` is responsible for
        returning a list of pb_utils.InferenceResponse objects. Each
        InferenceResponse should be created TritonPythonModel
        `InferenceResponse(output_tensors=..., error=...)` initialized
        with a list of output tensors for that request and an error message
        (zero if not error).

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse objects
        """
        responses = []

        for request in requests:
            try:
                # Get inputs
                prompt_tensor = pb_utils.get_input_tensor_by_name(request, "PROMPT")
                prompt = prompt_tensor.as_numpy()[0].decode("utf-8")

                negative_prompt_tensor = pb_utils.get_input_tensor_by_name(request, "NEGATIVE_PROMPT")
                if negative_prompt_tensor is not None:
                    negative_prompt = negative_prompt_tensor.as_numpy()[0].decode("utf-8")
                else:
                    negative_prompt = self.default_negative_prompt

                num_steps_tensor = pb_utils.get_input_tensor_by_name(request, "NUM_INFERENCE_STEPS")
                if num_steps_tensor is not None:
                    num_inference_steps = int(num_steps_tensor.as_numpy()[0])
                else:
                    num_inference_steps = self.default_num_inference_steps
                
                seed_tensor = pb_utils.get_input_tensor_by_name(request, "SEED")
                if seed_tensor is not None:
                    seed = int(seed_tensor.as_numpy()[0])
                else:
                    seed = self.default_seed

                # Generate unique filename
                safe_prompt_prefix = "".join(filter(str.isalnum, prompt))[:20] # Sanitize prompt for filename
                unique_id = str(uuid.uuid4())[:8]
                video_filename = f"video_{safe_prompt_prefix}_{unique_id}.mp4"
                video_path_output = os.path.join(self.video_output_dir, video_filename)

                # Run inference
                video_frames = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    seed=seed,
                    tiled=True, # From original video_api.py
                    tea_cache_l1_thresh=0.05, # From original video_api.py
                    tea_cache_model_id="Wan2.1-T2V-1.3B", # From original video_api.py
                )
                save_video(video_frames, video_path_output, fps=30, quality=5) # fps and quality from original

                # Create output tensor
                output_tensor = pb_utils.Tensor(
                    "GENERATED_VIDEO_PATH",
                    np.array([video_path_output.encode("utf-8")], dtype=np.object_)
                )
                inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
                responses.append(inference_response)

            except Exception as e:
                error = pb_utils.TritonError(f"Error during inference: {str(e)}")
                inference_response = pb_utils.InferenceResponse(output_tensors=[], error=error)
                responses.append(inference_response)
                print(f"Error processing request: {e}")

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` is optional. This function allows the
        model to perform any necessary clean ups before exit.
        """
        print("Cleaning up DiffSynth model...")
        del self.pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Finalize complete.")


