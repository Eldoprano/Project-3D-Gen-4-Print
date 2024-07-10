# 3DGen4Print
This project aims to provide a simple and fast way to generate printable 3D models from voice, text prompts or images.

It was created using Gradio as a frontend, which makes it easy to change the models used and gives a simple interface to the user.

## Features

- **Voice to Text**: Convert and translate voice prompts to english text using Whisper Medium.
- **Text to Image**: Generate 3D styled images from text prompts using Stable Diffusion.
- **Image to 3D Model**: Convert images to 3D models using TripoSR.
- **Background Removal**: Remove and resize backgrounds in images.
- **Mesh Refinement**: Refine generated meshes with PyMeshLab.
- **3D Model Slicing**: Slice 3D models for printing using PrusaSlicer.

## Models Used
- Voice to text: [Whisper Medium](https://huggingface.co/openai/whisper-medium)
- Text to image: [StableDiffusionXL-Base+LORA trained for 3D Style Images](https://huggingface.co/artificialguybr/3DRedmond-V1)
- Image to 3D: [TripoSR](https://github.com/VAST-AI-Research/TripoSR)

---

## Installation

### Prerequisites

- Docker (for containerized setup)
- CUDA-compatible GPU (if available)

### Steps
To get the project up and running, you just need to follow these simple steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/Eldoprano/Project-3D-Gen-4-Print
    cd Project-3D-Gen-4-Print
    ```

2. **Build and start container**:

    - Build the Docker container:
        ```bash
        docker-compose build
        ```
    - Start the Docker container:
        ```bash
        docker-compose up -d
        ```
> Note: GPU 0 is used by default. You can change this in the `docker-compose.yml` file.
---

## Usage

### Running the Gradio Interface
Once the container is running, it will automatically start the Gradio interface and download all necessary models. You can access the interface by going to: `http://localhost:7890`.

We recommend using Visual Studio Code and it's extension [Remote Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) to open the container if you are having problems with port forwarding.

## Troubleshooting

### Known Issues
- Automatic model slicing appears to not be working anymore. Because of this, we recommend manually slicing the models using PrusaSlicer. [This is the version that used to work](https://github.com/prusa3d/PrusaSlicer/releases/download/version_2.7.4/PrusaSlicer-2.7.4+linux-x64-GTK2-202404050940.tar.bz2).