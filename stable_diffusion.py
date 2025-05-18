import gradio as gr
import torch
import random
import sys
import os
from diffusers import SanaSprintPipeline
from PIL import Image

# Model setup
model_size = "0.6B"
MAX_IMAGE_SIZE = 1024
model_id = f"Efficient-Large-Model/Sana_Sprint_{model_size}_{MAX_IMAGE_SIZE}px_diffusers"
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pipeline
pipe = SanaSprintPipeline.from_pretrained(
    model_id,
    torch_dtype=dtype,
).to(device)

# Image generation function
def generate_images(prompt, resolution):
    num_images = 3
    num_inference_steps = 12
    guidance_scale = 5.0
    width, height = map(int, resolution.split("x"))

    images = []
    paths = []

    for i in range(num_images):
        seed = random.randint(0, sys.maxsize)
        generator = torch.Generator(device).manual_seed(seed)

        image = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            intermediate_timesteps=None,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
            output_type="pil"
        ).images[0]

        path = f"generated_image_{i+1}.jpg"
        image.save(path)

        images.append(image)
        paths.append(path)

    return images[0], paths[0], images[1], paths[1], images[2], paths[2], ""

# Reset function
def reset_fields():
    return ["", None, None, None, None, None, ""]

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🎨 Stable Diffusion Image Generator")

    with gr.Row():
        prompt = gr.Textbox(label="Prompt", placeholder="e.g., a boy swimming on cloud", lines=1)
        resolution = gr.Dropdown(
            label="Image Resolution",
            choices=["512x512", "768x768", "1024x1024", "1920x1080", "3840x2160"],
            value="1024x1024"
        )

    with gr.Row():
        generate_button = gr.Button("Generate")
        reset_button = gr.Button("Reset")

    # Each image followed by its download button
    with gr.Row():
        with gr.Column():
            output_image1 = gr.Image(label="Variation 1", interactive=False)
            download1 = gr.DownloadButton(label="Download", value=None)

        with gr.Column():
            output_image2 = gr.Image(label="Variation 2", interactive=False)
            download2 = gr.DownloadButton(label="Download", value=None)

        with gr.Column():
            output_image3 = gr.Image(label="Variation 3", interactive=False)
            download3 = gr.DownloadButton(label="Download", value=None)

    # Button actions
    generate_button.click(
        fn=generate_images,
        inputs=[prompt, resolution],
        outputs=[output_image1, download1, output_image2, download2, output_image3, download3, prompt]
    )

    reset_button.click(
        fn=reset_fields,
        inputs=[],
        outputs=[prompt, output_image1, output_image2, output_image3, download1, download2, download3, prompt]
    )

if __name__ == "__main__":
    demo.launch()
