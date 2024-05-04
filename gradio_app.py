import logging
import os
import tempfile
import time

import gradio as gr
import numpy as np
import rembg
import torch
from PIL import Image
from functools import partial

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation

import argparse
from diffusers import DiffusionPipeline
import subprocess



if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

tsr_model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)

# image_generation_model = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
image_generation_model = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo")
image_generation_model.enable_model_cpu_offload()
# image_generation_model.to(device)
print(device)

# adjust the chunk size to balance between speed and memory usage
tsr_model.renderer.set_chunk_size(8192)
tsr_model.to(device)


rembg_session = rembg.new_session()


def check_input(prompt):
    if prompt is None or prompt.strip() == "":
        raise gr.Error("Please write a prompt!")

def image_generation(text_input):
    text_input = "Professional 3d model of " + text_input + ", dramatic lighting, highly detailed, volumetric, cartoon"
    return image_generation_model(prompt=text_input, num_inference_steps=4, guidance_scale=0.0).images[0]


def preprocess(input_image, do_remove_background, foreground_ratio):
    torch.cuda.empty_cache()
    def fill_background(image):
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        return image

    if do_remove_background:
        image = input_image.convert("RGB")
        image = remove_background(image, rembg_session)
        image = resize_foreground(image, foreground_ratio)
        image = fill_background(image)
    else:
        image = input_image
        if image.mode == "RGBA":
            image = fill_background(image)
    return image


def generate(image, mc_resolution, formats=["obj"]):
    scene_codes = tsr_model(image, device=device)
    mesh = tsr_model.extract_mesh(scene_codes, resolution=mc_resolution)[0]
    mesh = to_gradio_3d_orientation(mesh)
    mesh_path = tempfile.NamedTemporaryFile(suffix=f".obj", delete=False)
    mesh.export(mesh_path.name)
    return mesh_path.name

def sliceObj(obj3D, size):
    if int(size) > 30:
        config_type = "./prusaConfig_big.ini"
    else: config_type = "./prusaConfig.ini"
    command = f"prusa-slicer --load {config_type} --rotate-x 90 --scale-to-fit {size},{size},{size} --slice --output 3dObj.bgcode {obj3D}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    output, _ = process.communicate()
    return output.decode().strip()


def run_example(image_pil):
    preprocessed = preprocess(image_pil, False, 0.9)
    mesh_name_obj = generate(preprocessed, 256, ["obj"])
    return preprocessed, mesh_name_obj


with gr.Blocks(title="TripoSR") as interface:
    gr.Markdown(
        """
    # Generative 3D Demo using Stable Diffusion and TripoSR
    
    **Tips:**
    1. If you find the result is unsatisfied, please try to change the foreground ratio. It might improve the results.
    2. It's better to disable "Remove Background" for the provided examples (except fot the last one) since they have been already preprocessed.
    3. Otherwise, please disable "Remove Background" option only if your input image is RGBA with transparent background, image contents are centered and occupy more than 70% of image width or height.
    """
    )
    with gr.Row(variant="panel"):
        with gr.Column():
            input_text = gr.Textbox(
                label="Prompt",
                placeholder="What do you want to generate in 3D?"
            )
            with gr.Row():
                set_size = gr.Slider(
                    label="Set output model size in mm",
                    minimum=15,
                    maximum=360, # Maximum size is limited by the PrusaSlicerXL
                    value=80,
                    step=5
                )
            
            with gr.Row():
                input_image = gr.Image(
                    label="Input Image",
                    image_mode="RGBA",
                    sources=['upload', 'webcam', 'clipboard'],
                    type="pil",
                    elem_id="content_image",
                )
                processed_image = gr.Image(label="Processed Image", interactive=False)
            with gr.Row(visible = False): # Hidden for now since it's not used
                with gr.Group():
                    do_remove_background = gr.Checkbox(
                        label="Remove Background", value=True
                    )
                    foreground_ratio = gr.Slider(
                        label="Foreground Ratio",
                        minimum=0.5,
                        maximum=1.0,
                        value=0.85,
                        step=0.05,
                    )
                    mc_resolution = gr.Slider(
                        label="Marching Cubes Resolution",
                        minimum=32,
                        maximum=320,
                        value=256,
                        step=32
                    )
                slicer_output = gr.Textbox(label="Slicer output")
            with gr.Row():
                submit = gr.Button("Generate", elem_id="generate", variant="primary")
        with gr.Column():
            output_model_obj = gr.Model3D(
                label="Output Model (OBJ Format)",
                interactive=False,
                scale=1,
            )
            gr.Markdown("Note: The model shown here is flipped. Download to get correct results.")
            with gr.Row():
                download = gr.File("./3dObj.bgcode")
    
    # When user presses enter on the Text input, we check it's content input 
    #  and continue with the 3D pipeline
    input_text.submit(fn=check_input, inputs=[input_text]).success(
        fn=image_generation,
        inputs=[input_text],
        outputs=[input_image]
    ).success(
        fn=preprocess,
        inputs=[input_image, do_remove_background, foreground_ratio],
        outputs=[processed_image],
    ).success(
        fn=generate,
        inputs=[processed_image, mc_resolution],
        outputs=[output_model_obj],
    ).success(
        fn=sliceObj,
        inputs=[output_model_obj, set_size],
        outputs=[slicer_output]
    ),


    submit.click(fn=check_input, inputs=[input_text, set_size]).success(
        fn=image_generation,
        inputs=[input_text],
        outputs=[input_image]
    ).success(
        fn=preprocess,
        inputs=[input_image, do_remove_background, foreground_ratio],
        outputs=[processed_image],
    ).success(
        fn=generate,
        inputs=[processed_image, mc_resolution],
        outputs=[output_model_obj],
    ).success(
        fn=sliceObj,
        inputs=[output_model_obj, set_size],
        outputs=[slicer_output]
    ),





if __name__ == '__main__':
    interface.queue(max_size=1)
    interface.launch(
        auth= None,
        share="store_true",
        server_name= None, 
        server_port= 7865
    )