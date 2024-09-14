import sys
import logging
from typing import Tuple

import torch
from PIL import Image
from cog import BasePredictor, Input, Path
from diffusers import FluxInpaintPipeline
from huggingface_hub import login

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Predictor(BasePredictor):
    def setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Python version: {sys.version}")

    def predict(
        self,
        hf_token: str = Input(description="Hugging Face API token"),
        image: Path = Input(description="Input image for inpainting"),
        mask: Path = Input(description="Mask image"),
        prompt: str = Input(description="Text prompt for inpainting"),
        lora_path: str = Input(description="Lora model path", default="XLabs-AI/flux-RealismLora"),
        lora_weights: str = Input(description="Lora weights name", default="lora.safetensors"),
        lora_scale: float = Input(description="Lora scale", default=0.9, ge=0, le=1),
        trigger_word: str = Input(description="Lora trigger word", default="a photo of TOK"),
        seed: int = Input(description="Random seed", default=42, ge=0),
        strength: float = Input(description="Strength of the inpainting", default=0.85, ge=0, le=1),
        num_inference_steps: int = Input(description="Number of inference steps", default=28, ge=1, le=100),
    ) -> Path:
        try:
            # Authenticate with Hugging Face
            login(token=hf_token)
            logger.info("Logged in to Hugging Face")

            # Load the model
            if not hasattr(self, 'pipe'):
                self.pipe = FluxInpaintPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-dev",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                ).to(self.device)

            image = self.load_image(image)
            mask = self.load_image(mask)

            width, height = self.resize_image_dimensions(image.size)
            image = image.resize((width, height), Image.LANCZOS)
            mask = mask.resize((width, height), Image.LANCZOS)

            self.pipe.load_lora_weights(lora_path, weight_name=lora_weights)
            logger.info(f"LoRA weights loaded from {lora_path}")

            generator = torch.Generator(device=self.device).manual_seed(seed)

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
            logger.info(f"Output image saved to {output_path}")

            return output_path

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise ValueError(f"Prediction failed: {str(e)}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def resize_image_dimensions(self, original_resolution_wh: Tuple[int, int], maximum_dimension: int = 1024) -> Tuple[int, int]:
        width, height = original_resolution_wh
        scaling_factor = maximum_dimension / max(width, height)
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        return new_width - (new_width % 32), new_height - (new_height % 32)

    def load_image(self, image_path: Path) -> Image.Image:
        try:
            image = Image.open(image_path)
            return image.convert("RGB")
        except Exception as e:
            raise ValueError(f"Error loading image: {str(e)}")
