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
import pymeshlab



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
file_time_format = '%d%b%y_%H-%M'


def check_input(prompt):
    if prompt is None or prompt.strip() == "":
        raise gr.Error("Please write a prompt!")

def image_generation(text_input):
    mod_text_input = "Professional 3d model of " + text_input + ", dramatic lighting, highly detailed, volumetric, cartoon"
    image = image_generation_model(prompt=mod_text_input, num_inference_steps=4, guidance_scale=0.0).images[0]

    # Save image to the model_outputs folder
    output_filename = f"{time.strftime(file_time_format)}_{text_input.replace(' ', '_')}.png"
    image.save("./model_outputs/" + output_filename)
    return image

def save_user_image(image):
    # Save image to the model_outputs folder
    output_filename = f"{time.strftime(file_time_format)}_user_input.png"
    image.save("./model_outputs/" + output_filename)
    return "user_input"

def preprocess(input_image, do_remove_background, foreground_ratio):
    # Unload image generation model from GPU (Probably..)
    #  This should help to avoid memory issues
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


def generate(image, mc_resolution, text_prompt, formats=["obj"]):
    # Generate mesh using the TSR model
    scene_codes = tsr_model(image, device=device)
    mesh = tsr_model.extract_mesh(scene_codes, resolution=mc_resolution)[0]
    mesh = to_gradio_3d_orientation(mesh)

    # Export the mesh to the model_outputs folder
    mesh_name = f"{time.strftime(file_time_format)}_{text_prompt.replace(' ', '_')}.obj"
    mesh.export("./model_outputs/" + mesh_name)

    # Remove isolated pieces with pymeshlab
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh("./model_outputs/" + mesh_name)
    ms.meshing_remove_connected_component_by_diameter(mincomponentdiag = pymeshlab.PercentageValue(25.0))
    ms.save_current_mesh("./model_outputs/" + mesh_name)

    return "./model_outputs/" + mesh_name

def sliceObj(obj3D, size, text_prompt):
    # Determine the printer configuration based on the size of the model in mm
    if int(size) > 30:
        config_type = "./prusaConfig_big.ini"
    else:
        config_type = "./prusaConfig.ini"

    # Construct the command to slice the 3D model using PrusaSlicer
    file_name = "./model_outputs/" + f"{time.strftime(file_time_format)}_{text_prompt.replace(' ', '_')}.bgcode"
    command = f"prusa-slicer --load {config_type} --rotate-x 90 --scale-to-fit {size},{size},{size} --slice --output {file_name} {obj3D}"

    # Execute the slicing command and capture the output
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    output, _ = process.communicate()

    # Return the output as a string
    return output.decode().strip(), file_name

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
            with gr.Row(): # Hidden for now since it's not used
                with gr.Group(visible = False):
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
                download = gr.File(label="Download the generated .bgcode file")
    
    # When user presses enter on the Text input or presses the submit button, 
    #  we check it's content input and continue with the 3D pipeline
    gr.on(
        triggers=[input_text.submit, submit.click],
        fn=check_input, 
        inputs=[input_text]
    ).success(
        fn=image_generation,
        inputs=[input_text],
        outputs=[input_image]
    ).success(
        fn=preprocess,
        inputs=[input_image, do_remove_background, foreground_ratio],
        outputs=[processed_image],
    ).success(
        fn=generate,
        inputs=[processed_image, mc_resolution, input_text],
        outputs=[output_model_obj],
    ).success(
        fn=sliceObj,
        inputs=[output_model_obj, set_size, input_text],
        outputs=[slicer_output, download]
    ),

    # When user loads an image into input_image, we pass this image through the
    #  pipeline, skipping the image generation step
    input_image.upload(
        fn=save_user_image,
        inputs=[input_image],
        outputs=[input_text]
    ).success(
        fn=preprocess, 
        inputs=[input_image, do_remove_background, foreground_ratio],
        outputs=[processed_image]
    ).success(
        fn=generate,
        inputs=[processed_image, mc_resolution, input_text],
        outputs=[output_model_obj],
    ).success(
        fn=sliceObj,
        inputs=[output_model_obj, set_size, input_text],
        outputs=[slicer_output, download]
    ),



if __name__ == '__main__':
    interface.queue(max_size=1)
    interface.launch(
        auth= None,
        share="store_true",
        server_name= None, 
        server_port= 7865
    )