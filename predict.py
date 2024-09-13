import os
import logging
from io import BytesIO

import torch
import requests
from PIL import Image
from cog import BasePredictor, Input, Path
from diffusers import FluxInpaintPipeline
from huggingface_hub import login

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Predictor(BasePredictor):
    def setup(self):
        logger.info("Setting up the predictor...")
        try:
            HF_TOKEN = os.environ.get("HF_TOKEN")
            if not HF_TOKEN:
                raise ValueError("HF_TOKEN environment variable is not set")
            login(token=HF_TOKEN)
            logger.info("Successfully logged in to Hugging Face")

            self.pipe = FluxInpaintPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=torch.bfloat16
            ).to("cuda")
            logger.info("FluxInpaintPipeline loaded successfully")
        except Exception as e:
            logger.error(f"Error during setup: {str(e)}")
            raise

    def predict(
        self,
        image: Path = Input(description="Input image for inpainting"),
        mask: Path = Input(description="Mask image"),
        image_url: str = Input(description="URL of input image (optional)", default=None),
        mask_url: str = Input(description="URL of mask image (optional)", default=None),
        prompt: str = Input(description="Text prompt for inpainting"),
        lora_path: str = Input(description="Lora model path", default="XLabs-AI/flux-RealismLora"),
        lora_weights: str = Input(description="Lora weights name", default="lora.safetensors"),
        lora_scale: float = Input(description="Lora scale", default=0.9),
        trigger_word: str = Input(description="Lora trigger word", default="a photo of TOK"),
        seed: int = Input(description="Random seed", default=42),
        strength: float = Input(description="Strength of the inpainting", default=0.85),
        num_inference_steps: int = Input(description="Number of inference steps", default=28),
        blur_mask: bool = Input(description="Whether to blur the mask", default=False),
        blur_factor: int = Input(description="Blur factor for mask", default=33),
    ) -> Path:
        try:
            logger.info("Starting prediction process...")

            # Load image and mask (from file or URL)
            image = self.load_image(image, image_url)
            mask = self.load_image(mask, mask_url)

            if image is None or mask is None:
                raise ValueError("Both image and mask must be provided (either as file or URL)")

            # Resize images
            width, height = self.resize_image_dimensions(image.size)
            image = image.resize((width, height), Image.LANCZOS)
            mask = mask.resize((width, height), Image.LANCZOS)
            logger.info(f"Images resized to {width}x{height}")

            # Blur mask if requested
            if blur_mask:
                mask = self.pipe.mask_processor.blur(mask, blur_factor=blur_factor)
                logger.info(f"Mask blurred with factor {blur_factor}")

            # Load LoRA weights
            logger.info(f"Loading LoRA weights from {lora_path}")
            self.pipe.load_lora_weights(lora_path, weight_name=lora_weights)

            generator = torch.Generator(device="cuda").manual_seed(seed)
            logger.info(f"Using seed: {seed}")

            logger.info("Starting image generation...")
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
            logger.info("Image generation completed")

            output_path = Path("output.png")
            result.save(output_path)
            logger.info(f"Output image saved to {output_path}")

            return output_path

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

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

    def load_image(self, image_path, image_url):
        try:
            if image_path:
                logger.info(f"Loading image from file: {image_path}")
                return Image.open(image_path).convert("RGB")
            elif image_url:
                logger.info(f"Loading image from URL: {image_url}")
                response = requests.get(image_url)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert("RGB")
            return None
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            raise
