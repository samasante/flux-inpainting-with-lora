from cog import BasePredictor, Input, Path
import torch
from PIL import Image
import numpy as np
from diffusers import FluxInpaintPipeline
from huggingface_hub import login
import os

class Predictor(BasePredictor):
    def setup(self):
        HF_TOKEN = os.environ.get("HF_TOKEN")
        login(token=HF_TOKEN)
        self.pipe = FluxInpaintPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16
        ).to("cuda")

    def predict(
        self,
        image: Path = Input(description="Input image for inpainting"),
        mask: Path = Input(description="Mask image"),
        prompt: str = Input(description="Text prompt for inpainting"),
        lora_path: str = Input(description="Lora model path", default="XLabs-AI/flux-RealismLora"),
        lora_weights: str = Input(description="Lora weights name", default="lora.safetensors"),
        lora_scale: float = Input(description="Lora scale", default=0.9),
        trigger_word: str = Input(description="Lora trigger word", default="a photo of TOK"),
        seed: int = Input(description="Random seed", default=42),
        strength: float = Input(description="Strength of the inpainting", default=0.85),
        num_inference_steps: int = Input(description="Number of inference steps", default=28),
    ) -> Path:
        image = Image.open(image).convert("RGB")
        mask = Image.open(mask).convert("RGB")

        # Resize images
        width, height = self.resize_image_dimensions(image.size)
        image = image.resize((width, height), Image.LANCZOS)
        mask = mask.resize((width, height), Image.LANCZOS)

        # Load LoRA weights
        self.pipe.load_lora_weights(lora_path, weight_name=lora_weights)

        generator = torch.Generator(device="cuda").manual_seed(seed)

        result = self.pipe(
            prompt=f"{prompt} {trigger_word}",
            image=image,
            mask_image=mask,
            width=width,
            height=height,
            strength=strength,
            generator=generator,
            num_inference_steps=num_inference_steps,
            max_sequence_length=256,
            joint_attention_kwargs={"scale": lora_scale},
        ).images[0]

        output_path = Path("output.png")
        result.save(output_path)
        return output_path

    def resize_image_dimensions(self, original_resolution_wh, maximum_dimension=1024):
        width, height = original_resolution_wh

        if width > height:
            scaling_factor = maximum_dimension / width
        else:
            scaling_factor = maximum_dimension / height

        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)

        new_width = new_width - (new_width % 32)
        new_height = new_height - (new_height % 32)

        return new_width, new_height
