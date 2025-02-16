from diffusers import StableDiffusionPipeline
import torch

# Defining the model ID
model_id = "CompVis/stable-diffusion-v1-4"

# Initialize Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_auth_token="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"  # Replace with your Hugging Face token
)
pipe = pipe.to("cuda")  # Moving pipeline to GPU for faster inference
