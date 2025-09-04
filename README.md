# Cultural Transfer of Indian Art Styles with CycleGAN

This project applies CycleGAN-based image-to-image translation to render input photos in traditional Indian art styles such as Madhubani, Tanjavur, Mughal miniature, and Warli.

## Features
- Streamlit web UI to upload an image and select an art style
- CycleGAN generator for fast inference on CPU/GPU
- Pluggable weights per style (drop-in `.pth` files)

## Quickstart

1) Create a virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Prepare weights. Place pretrained generator weights here:
```
weights/
  madhubani/G_A2B.pth
  tanjavur/G_A2B.pth
  mughal/G_A2B.pth
  warli/G_A2B.pth
```

3) Run the UI:
```bash
streamlit run app.py
```

4) Upload an image and select a target art style.

## Weight format
The app expects PyTorch `.pth` files containing the state dict for the generator (`netG_A2B`). If your checkpoint is a dict (e.g., `{'netG_A2B': state_dict, ...}`) the loader will attempt to locate the first tensor-only state dict. If names are nested, pass through as-is; the loader strips common `module.` prefixes.

## Training (optional)
A minimal training skeleton is provided under `src/train/`. You will need paired/unpaired datasets per style. Suggested structure:
```
data/
  madhubani/
    trainA/  # photos
    trainB/  # madhubani images
  tanjavur/
  mughal/
  warli/
```
Follow the references in `src/train/README_train.md` and consider starting from existing public CycleGAN weights.

## License
For research and educational use. 