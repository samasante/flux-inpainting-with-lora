# Flux Inpainting with LoRA

This project implements an AI-powered image inpainting model using the Flux architecture with LoRA (Low-Rank Adaptation) fine-tuning. It's designed to run on Replicate, allowing users to easily perform image inpainting tasks with custom prompts and LoRA models.

## Overview

The Flux Inpainting model can intelligently fill in masked areas of an image based on the surrounding context and a text prompt. This implementation also supports the use of custom LoRA models for fine-tuned results.

## Features

- Image inpainting using the Flux architecture
- Support for custom LoRA models
- Adjustable parameters for fine-tuned control
- Easy deployment on Replicate

## Usage

To use this model on Replicate, you'll need to provide the following inputs:

- `hf_token`: Your Hugging Face API token for accessing the model
- `image`: The input image for inpainting
- `mask`: A mask image indicating the area to be inpainted
- `prompt`: A text prompt describing the desired inpainting result
- `lora_path`: Path to the LoRA model (default: "XLabs-AI/flux-RealismLora")
- `lora_weights`: Name of the LoRA weights file (default: "lora.safetensors")
- `lora_scale`: Scale factor for LoRA (default: 0.9, range: 0-1)
- `trigger_word`: LoRA trigger word (default: "a photo of TOK")
- `seed`: Random seed for reproducibility (default: 42)
- `strength`: Strength of the inpainting effect (default: 0.85, range: 0-1)
- `num_inference_steps`: Number of inference steps (default: 28, range: 1-100)

The model will return an output image with the inpainted result.

## Development

To modify or extend this project:

1. Update the `predict.py` file to change the model's behavior or add new features.
2. Modify the `cog.yaml` file if you need to change the build configuration or add new dependencies.
3. Test your changes locally using the Cog CLI before deploying to Replicate.

## Deployment

To deploy this model to Replicate:

1. Ensure you have the Cog CLI installed and configured.
2. Run `cog push r8.im/your-username/your-model-name` to build and push the model.

## Acknowledgements

This project is based on the work by jiuface on Hugging Face Spaces.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
