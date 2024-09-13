from typing import Tuple

import requests
import random
import numpy as np
import gradio as gr
import spaces
import torch
from PIL import Image
from diffusers import FluxInpaintPipeline
from huggingface_hub import login
import os
import time
from gradio_imageslider import ImageSlider

from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from transformers import CLIPTextModel, CLIPTokenizer,T5EncoderModel, T5TokenizerFast
import requests
from io import BytesIO
import PIL.Image
import requests

MARKDOWN = """
# FLUX.1 Inpainting with lora
"""

MAX_SEED = np.iinfo(np.int32).max
IMAGE_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = os.environ.get("HF_TOKEN")

login(token=HF_TOKEN)

bfl_repo="black-forest-labs/FLUX.1-dev"

class calculateDuration:
    def __init__(self, activity_name=""):
        self.activity_name = activity_name

    def __enter__(self):
        self.start_time = time.time()
        self.start_time_formatted = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time))
        print(f"Activity: {self.activity_name}, Start time: {self.start_time_formatted}")
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        self.end_time_formatted = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.end_time))
        
        if self.activity_name:
            print(f"Elapsed time for {self.activity_name}: {self.elapsed_time:.6f} seconds")
        else:
            print(f"Elapsed time: {self.elapsed_time:.6f} seconds")
        
        print(f"Activity: {self.activity_name}, End time: {self.start_time_formatted}")


def remove_background(image: Image.Image, threshold: int = 50) -> Image.Image:
    image = image.convert("RGBA")
    data = image.getdata()
    new_data = []
    for item in data:
        avg = sum(item[:3]) / 3
        if avg < threshold:
            new_data.append((0, 0, 0, 0))
        else:
            new_data.append(item)

    image.putdata(new_data)
    return image

# text_encoder = CLIPTextModel.from_pretrained(os.path.join(os.getcwd(), "flux_text_encoders/clip_l.safetensors"), torch_dtype=dtype)
# tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
# text_encoder_2 = T5EncoderModel.from_pretrained(os.path.join(os.getcwd(), "flux_text_encoders/t5xxl_fp8_e4m3fn.safetensors"), torch_dtype=dtype)
# tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2", torch_dtype=dtype)
# vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype)
# transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype)


pipe = FluxInpaintPipeline.from_pretrained(bfl_repo, torch_dtype=torch.bfloat16).to(DEVICE)


def resize_image_dimensions(
    original_resolution_wh: Tuple[int, int],
    maximum_dimension: int = IMAGE_SIZE
) -> Tuple[int, int]:
    width, height = original_resolution_wh

    # if width <= maximum_dimension and height <= maximum_dimension:
    #     width = width - (width % 32)
    #     height = height - (height % 32)
    #     return width, height

    if width > height:
        scaling_factor = maximum_dimension / width
    else:
        scaling_factor = maximum_dimension / height

    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)

    new_width = new_width - (new_width % 32)
    new_height = new_height - (new_height % 32)

    return new_width, new_height


@spaces.GPU(duration=100)
def process(
    input_image_editor: dict,
    image_url: str,
    mask_url: str,
    blur_mask: bool,
    blur_factor: int,
    lora_path: str,
    lora_weights: str,
    lora_scale: float,
    trigger_word: str,
    input_text: str,
    seed_slicer: int,
    randomize_seed_checkbox: bool,
    strength_slider: float,
    num_inference_steps_slider: int,
    progress=gr.Progress(track_tqdm=True)
):
    if not input_text:
        gr.Info("Please enter a text prompt.")
        return None, None

    # default image edtiro
    image = input_image_editor['background']
    mask = input_image_editor['layers'][0]

    if image_url:
        print("start to fetch image from url", image_url)
        response = requests.get(image_url)
        response.raise_for_status()
        image = PIL.Image.open(BytesIO(response.content))
        print("fetch image success")

    if mask_url:
        print("start to fetch mask from url", mask_url)
        response = requests.get(mask_url)
        response.raise_for_status()
        mask = PIL.Image.open(BytesIO(response.content))
        print("fetch mask success")

    if not image:
        gr.Info("Please upload an image.")
        return None, None

    if not mask:
        gr.Info("Please draw a mask on the image.")
        return None, None
    if blur_mask:
        mask = pipe.mask_processor.blur(mask, blur_factor=blur_factor)

    with calculateDuration("resize image"):
        width, height = resize_image_dimensions(original_resolution_wh=image.size)
        resized_image = image.resize((width, height), Image.LANCZOS)
        resized_mask = mask.resize((width, height), Image.LANCZOS)
    
    with calculateDuration("load lora"):
        print(lora_path, lora_weights)
        pipe.load_lora_weights(lora_path, weight_name=lora_weights)
    
    if randomize_seed_checkbox:
        seed_slicer = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed_slicer)

    with calculateDuration("run pipe"):
        print(input_text, width, height, strength_slider, num_inference_steps_slider, lora_scale)
        result = pipe(
            prompt=f"{input_text} {trigger_word}",
            image=resized_image,
            mask_image=resized_mask,
            width=width,
            height=height,
            strength=strength_slider,
            generator=generator,
            num_inference_steps=num_inference_steps_slider,
            max_sequence_length=256,
            joint_attention_kwargs={"scale": lora_scale},
        ).images[0]
    
    return [resized_image, result], resized_mask


with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    with gr.Row():
        with gr.Column():
            input_image_editor_component = gr.ImageEditor(
                label='Image',
                type='pil',
                sources=["upload", "webcam"],
                image_mode='RGB',
                layers=False,
                brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed"))
                
            image_url =  gr.Textbox(
                    label="image url",
                    show_label=True,
                    max_lines=1,
                    placeholder="Enter your image url (Optional)",
                )
            mask_url =  gr.Textbox(
                    label="Mask image url",
                    show_label=True,
                    max_lines=1,
                    placeholder="Enter your mask image url (Optional)",
                )
        
            with gr.Accordion("Prompt Settings", open=True):

                input_text_component = gr.Textbox(
                    label="Inpaint prompt",
                    show_label=True,
                    max_lines=1,
                    placeholder="Enter your prompt",
                )
                trigger_word = gr.Textbox(
                    label="Lora trigger word",
                    show_label=True,
                    max_lines=1,
                    placeholder="Enter your lora trigger word here",
                    value="a photo of TOK"
                    
                )

                submit_button_component = gr.Button(
                    value='Submit', variant='primary', scale=0)

            with gr.Accordion("Lora Settings", open=True):
                lora_path = gr.Textbox(
                    label="Lora model path",
                    show_label=True,
                    max_lines=1,
                    placeholder="Enter your model path",
                    info="Currently, only LoRA hosted on Hugging Face'model can be loaded properly.",
                    value="XLabs-AI/flux-RealismLora"
                )
                lora_weights = gr.Textbox(
                    label="Lora weights",
                    show_label=True,
                    max_lines=1,
                    placeholder="Enter your lora weights name",
                    value="lora.safetensors"
                )
                lora_scale = gr.Slider(
                    label="Lora scale",
                    show_label=True,
                    minimum=0,
                    maximum=1,
                    step=0.1,
                    value=0.9,
                )
                
            with gr.Accordion("Advanced Settings", open=True):
                
                
                seed_slicer_component = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=42,
                )

                randomize_seed_checkbox_component = gr.Checkbox(
                    label="Randomize seed", value=True)

                blur_mask = gr.Checkbox(
                    label="if blur mask", value=False)
                blur_factor =  gr.Slider(
                    label="blur factor",
                    minimum=0,
                    maximum=50,
                    step=1,
                    value=33,
                )
                with gr.Row():
                    strength_slider_component = gr.Slider(
                        label="Strength",
                        info="Indicates extent to transform the reference `image`. "
                             "Must be between 0 and 1. `image` is used as a starting "
                             "point and more noise is added the higher the `strength`.",
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        value=0.85,
                    )

                    num_inference_steps_slider_component = gr.Slider(
                        label="Number of inference steps",
                        info="The number of denoising steps. More denoising steps "
                             "usually lead to a higher quality image at the",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=28,
                    )
        with gr.Column():
            output_image_component = ImageSlider(label="Generate image", type="pil", slider_color="pink")
            
            with gr.Accordion("Debug", open=False):
                output_mask_component = gr.Image(
                    type='pil', image_mode='RGB', label='Input mask', format="png")

    submit_button_component.click(
        fn=process,
        inputs=[
            input_image_editor_component,
            image_url,
            mask_url,
            blur_mask,
            blur_factor,
            lora_path,
            lora_weights,
            lora_scale,
            trigger_word,
            input_text_component,
            seed_slicer_component,
            randomize_seed_checkbox_component,
            strength_slider_component,
            num_inference_steps_slider_component
        ],
        outputs=[
            output_image_component,
            output_mask_component
        ]
    )

demo.launch(debug=False, show_error=True)
