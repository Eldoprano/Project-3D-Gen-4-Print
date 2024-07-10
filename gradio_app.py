import time
import gradio as gr
import numpy as np
import rembg
import torch
from PIL import Image
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation
from diffusers import DiffusionPipeline
import subprocess
import pymeshlab
from transformers import pipeline
from functools import partial
from PIL import Image


if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

# Enter your HF token here
hf_access_token = "HF-Token"
file_time_format = '%y%b%d_%H-%M'

###################
# Load the models #
###################

### Load the image generation model ###
# image_generation_model = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo")
image_generation_model = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
image_generation_model.load_lora_weights("artificialguybr/3DRedmond-V1")
image_generation_model.to("cuda:0")
model_specific_prompt = "3D Render Style, 3DRenderAF, "
# image_generation_model.enable_model_cpu_offload()
# image_generation_model.to(device)


### Load TripoSR ###
tsr_model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)
# adjust the chunk size to balance between speed and memory usage
tsr_model.renderer.set_chunk_size(8192)
tsr_model.to(device)

### Load the background removal model ###
rembg_session = rembg.new_session()

### Load the speech recognition model ###
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-medium", generate_kwargs={"task":"translate"})



######################
# Pipeline Functions #
######################

# ----------------------------------------------
# This functions are here to change the visibility of some elements
# It is so that we can hide elements when user clicks on generate
#  and show them when user clicks on feeling lucky (and vice versa)
def change_visibility_of_single_output(button):
    return gr.Image(label="Input Image", image_mode="RGBA", sources=['upload', 'webcam', 'clipboard'], type="pil", elem_id="content_image", visible=button == "I'm Feeling Lucky")

def change_visibility_of_preprocessing_output(button):
    return gr.Image(label="Processed Image", interactive=False, visible=button == "I'm Feeling Lucky")

def change_visibility_of_gallery_outputs(button):
    return gr.Gallery(label="Image Gallery", visible=button == "Generate", allow_preview=False)
# ----------------------------------------------

####################
# Helper Functions #
####################
# Check if the prompt is empty
def check_input(prompt):
    if prompt is None or prompt.strip() == "":
        raise gr.Error("Please write a prompt!")

# Remove special characters from the file name
def sanitize_file_name(file_name, size=250):
    sanitized = "".join([c for c in file_name.replace(" ","_") if c.isalpha() or c.isdigit() or c in "._-"]).rstrip()
    return sanitized[:size]

# Save image to the model_outputs folder
def save_user_image(image):
    output_filename = f"{time.strftime(file_time_format)}_user_input"
    output_filename = sanitize_file_name(output_filename)
    image.save("./model_outputs/" + output_filename + ".png")
    return "user_input"

# Remove background and resize the image
def preprocess_image(input_image, do_remove_background, foreground_ratio):
    # Unload image generation model from GPU
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

# Remove background and resize multiple images
def preprocess_multiple_images(input_images, do_remove_background, foreground_ratio):
    processed_images = []
    for input_image, _ in input_images:
        image = Image.open(input_image)
        processed_image = preprocess_image(image, do_remove_background, foreground_ratio)
        processed_images.append(processed_image)
    return processed_images


# Remove isolated pieces with pymeshlab
def refine_mesh(mesh_path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_path)
    ms.meshing_remove_connected_component_by_diameter(mincomponentdiag = pymeshlab.PercentageValue(25.0))
    ms.save_current_mesh(mesh_path)
    return mesh_path

# Slice the 3D model using PrusaSlicer
def sliceObj(obj3D, size, text_prompt):
    # Determine the printer configuration based on the size of the model in mm
    if int(size) > 30:
        config_type = "./prusaConfig_big.ini"
    else:
        config_type = "./prusaConfig.ini"

    # Construct the command to slice the 3D model using PrusaSlicer
    text_prompt = f"{time.strftime(file_time_format)}_{text_prompt}"
    text_prompt = sanitize_file_name(text_prompt)
    file_name = "./model_outputs/" + text_prompt + ".bgcode"
    command = f"prusa-slicer --load {config_type} --rotate-x 90 --scale-to-fit {size},{size},{size} --slice --output {file_name} {obj3D}"

    # Execute the slicing command and capture the output
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    output, _ = process.communicate()

    # Return the output as a string
    return output.decode().strip(), file_name

# It returns a Gaussian Splatt for a given example video
def runSplatExample(video):
    video = str(video)
    video = video.split('/')[-1].replace("muted_", "")
    return "examples/" + video.replace('.mp4', '.splat')

#################
# Use AI Models #
#################

# Transcribe the audio to text and translate it to English
def audio_to_text(audio):
    sr, y = audio
    # Check if the audio has more than one channel and convert it to mono if necessary
    if y.ndim > 1 and y.shape[1] > 1:
        y = np.mean(y, axis=1)  # Convert to mono by averaging the channels
    y = y.astype(np.float32)
    y /= np.max(np.abs(y), axis=0, keepdims=True) + 1e-9  # Normalize audio

    return transcriber({"sampling_rate": sr, "raw": y})["text"]

# Generate images using the image generation model
# `button_clicked` decides whether to generate 1 or 4 images (sorry for that, it's not a nice solution)
def image_generation(text_input, button_clicked):
    torch.cuda.empty_cache()
    if button_clicked == "I'm Feeling Lucky":
        num_images = 1
    elif button_clicked == "Generate":
        num_images = 4
    mod_text_input = f"{model_specific_prompt} Professional 3d model of {text_input}, dramatic lighting, highly detailed, volumetric, cartoon"
    images = image_generation_model(
        prompt=mod_text_input, 
        num_inference_steps=28, 
        guidance_scale=7.0, 
        num_images_per_prompt=num_images).images

    # Save image to the model_outputs folder
    for i, image in enumerate(images):
        output_filename = f"{time.strftime(file_time_format)}_{text_input}_{i}"
        output_filename = sanitize_file_name(output_filename)
        image.save("./model_outputs/" + output_filename + ".png")
    if button_clicked == "I'm Feeling Lucky":
        return images[0]
    return images

# Generate 3D model using TripoSR
def generate(image, mc_resolution, text_prompt):
    # Convert numpy.ndarray to PIL Image if necessary
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    # Generate mesh using the TSR model
    scene_codes = tsr_model(image.convert("RGB"), device=device)
    mesh = tsr_model.extract_mesh(scene_codes, resolution=mc_resolution)[0]
    mesh = to_gradio_3d_orientation(mesh)

    # Export the mesh to the model_outputs folder
    mesh_name = f"{time.strftime(file_time_format)}_{text_prompt}"
    mesh_name = sanitize_file_name(mesh_name)
    mesh.export("./model_outputs/" + mesh_name + ".obj")

    refined_mesh = refine_mesh("./model_outputs/" + mesh_name + ".obj")

    return refined_mesh

####################
# Gradio Interface #
####################
with gr.Blocks(title="TripoSR") as interface:
    gr.Markdown(
        """
    # Generative 3D Demo using Stable Diffusion and TripoSR
    """
    )
    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                input_text = gr.Textbox(
                    label="Prompt",
                    placeholder="What do you want to generate in 3D?",
                    scale=3,
                )
                input_voice = gr.Audio(
                    sources=["microphone"],
                    scale=1,
                    editable=False,
                    container=False,
                    waveform_options=gr.WaveformOptions(
                        waveform_color="#01C6FF",
                        waveform_progress_color="#0066B4",
                        show_controls=False,
                    ),
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
                image_gallery = gr.Gallery(
                    label="Image Gallery",
                    visible=False,
                    allow_preview=False,
                )
            
            with gr.Row():
                input_image = gr.Image(
                    label="Input Image",
                    image_mode="RGBA",
                    sources=['upload', 'webcam', 'clipboard'],
                    type="pil",
                    elem_id="content_image",
                )
                processed_image = gr.Image(
                    label="Processed Image", 
                    interactive=False
                )
                input_video = gr.Video(
                    label="Input Image",
                    sources=[],
                    elem_id="content_video",
                    visible=False
                )

            with gr.Row(): # Hidden since to make the interface more simple. 
                           # Please refer to https://huggingface.co/spaces/stabilityai/TripoSR
                           # to see how it changes 3D Model generation
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
                submit = gr.Button("Generate", elem_id="generate", variant="primary", scale=3)
                lucky = gr.Button("I'm Feeling Lucky", elem_id="lucky", variant="secondary", scale=1)

        with gr.Column():
            output_model_obj = gr.Model3D(
                label="Output Model (OBJ Format)",
                interactive=False,
                scale=1,
            )
            gr.Markdown("Note: The model shown here is flipped. Download to get correct results.")
            with gr.Row():
                download = gr.File(label="Download the generated .bgcode file")
    
    # Proof of concept for Gaussian Splatting
    with gr.Row(variant="panel"):
        gr.Examples(
            examples=[
                "examples/kemptenAuto.mp4",
                # "examples/kemptenStatue.mp4", # Too large to upload to github..
                "examples/valhala.mp4",
            ],
            label="Gaussian Splatting input examples",
            inputs=[input_video],
            fn=partial(runSplatExample),
            outputs=[output_model_obj],
            run_on_click=True,
        )


################################################
#       The pipelines are defined here         #
################################################

# Multiple image selection Pipeline (We generate 4 images and let the user select one)
    gr.on(
        triggers=[submit.click, input_text.submit],
        fn=check_input,
        inputs=[input_text]
    # ---------------------------------------------------------
    # All this code is to change the visibility of the elements
    ).success(
        fn=change_visibility_of_single_output, 
        inputs=[submit], 
        outputs=[input_image]
    ).success(
        fn=change_visibility_of_preprocessing_output, 
        inputs=[submit], 
        outputs=[processed_image]
    ).success(
        fn=change_visibility_of_gallery_outputs, 
        inputs=[submit], 
        outputs=[image_gallery]
    # ---------------------------------------------------------
    ).success(
        fn=image_generation,
        inputs=[input_text, submit],
        outputs=[image_gallery]
    ).success(
        fn=preprocess_multiple_images,
        inputs=[image_gallery, do_remove_background, foreground_ratio],
        outputs=[image_gallery],
    )
    # Here the pipeline continues once the user selects an image
    def get_selected_image(images, evt: gr.SelectData):
        return images[evt.index][0]
    image_gallery.select(
        get_selected_image,
        inputs=[image_gallery],
        outputs=input_image
    ).success(
        fn=generate,
        inputs=[input_image, mc_resolution, input_text],
        outputs=[output_model_obj],
    ).success(
        fn=sliceObj,
        inputs=[output_model_obj, set_size, input_text],
        outputs=[slicer_output, download]
    )


# Lucky button Pipeline (We generate only one image)
    gr.on(
        triggers=[lucky.click],
        fn=check_input, 
        inputs=[input_text]
    # ---------------------------------------------------------
    # All this code is to change the visibility of the elements
    ).success(
        fn=change_visibility_of_single_output, 
        inputs=[lucky], 
        outputs=[input_image]
    ).success(
        fn=change_visibility_of_preprocessing_output, 
        inputs=[lucky], 
        outputs=[processed_image]
    ).success(
        fn=change_visibility_of_gallery_outputs, 
        inputs=[lucky], 
        outputs=[image_gallery]
    # ---------------------------------------------------------
    ).success(
        fn=image_generation,
        inputs=[input_text, lucky],
        outputs=[input_image]
    ).success(
        fn=preprocess_image,
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

# Voice input pipeline
    input_voice.stop_recording(
        fn=audio_to_text,
        inputs=[input_voice],
        outputs=[input_text]
    # ---------------------------------------------------------
    # All this code is to change the visibility of the elements
    ).success(
        fn=change_visibility_of_single_output, 
        inputs=[lucky], 
        outputs=[input_image]
    ).success(
        fn=change_visibility_of_preprocessing_output, 
        inputs=[lucky], 
        outputs=[processed_image]
    ).success(
        fn=change_visibility_of_gallery_outputs, 
        inputs=[lucky], 
        outputs=[image_gallery]
    # ---------------------------------------------------------
    ).success(
        fn=image_generation,
        inputs=[input_text, lucky],
        outputs=[input_image]
    ).success(
        fn=preprocess_image,
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


# Image input pipeline (When uploads its own image)
    input_image.upload(
        fn=save_user_image,
        inputs=[input_image],
        outputs=[input_text]
    ).success(
        fn=preprocess_image, 
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

# Start the interface
if __name__ == '__main__':
    interface.queue(max_size=15)
    interface.launch(
        auth= None,
        share="store_true",
        server_name= None, 
        server_port= 7860
    )