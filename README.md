# 3DGen4Print
This project aims to provide a simple and fast way to generate printable 3D models from text prompts. 

For the time being it uses StableDiffusionXL-Turbo to generate images from text prompts and TripoSR to generate 3D models from images.

## Getting Started
### Installation
- Install Docker
- Install CUDA 12.3 (We didn't test with other versions)
- To build the container, run `docker-compose build` (It uses the GPU 0 by default, you can change it in the `docker-compose.yml` file)
- (For now) To get Prusa Slicer inside the container download and extract https://github.com/prusa3d/PrusaSlicer/releases/download/version_2.7.4/PrusaSlicer-2.7.4+linux-x64-GTK2-202404050940.tar.bz2
  
  And add the folder to the path using `export PATH=$PATH:[/path/to/]PrusaSlicer-2.7.4+linux-x64-GTK2-202404050940`\

### Run the gradio app
- Run `python3 gradio_app.py`

### Manual TripoSR Inference
```sh
python run.py examples/chair.png --output-dir output/
```
This will save the reconstructed 3D model to `output/`. You can also specify more than one image path separated by spaces. The default options takes about **6GB VRAM** for a single image input.

For detailed usage of this script, use `python run.py --help`.

## Troubleshooting
> AttributeError: module 'torchmcubes_module' has no attribute 'mcubes_cuda'

or

> torchmcubes was not compiled with CUDA support, use CPU version instead.

This is because `torchmcubes` is compiled without CUDA support. Please make sure that 

- The locally-installed CUDA major version matches the PyTorch-shipped CUDA major version. For example if you have CUDA 11.x installed, make sure to install PyTorch compiled with CUDA 11.x.
- `setuptools>=49.6.0`. If not, upgrade by `pip install --upgrade setuptools`.

Then re-install `torchmcubes` by:

```sh
pip uninstall torchmcubes
pip install git+https://github.com/tatsy/torchmcubes.git
```

## Citation
```BibTeX
@article{TripoSR2024,
  title={TripoSR: Fast 3D Object Reconstruction from a Single Image},
  author={Tochilkin, Dmitry and Pankratz, David and Liu, Zexiang and Huang, Zixuan and and Letts, Adam and Li, Yangguang and Liang, Ding and Laforte, Christian and Jampani, Varun and Cao, Yan-Pei},
  journal={arXiv preprint arXiv:2403.02151},
  year={2024}
}
```
