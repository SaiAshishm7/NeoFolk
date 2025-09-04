# Training Skeleton (CycleGAN)

This folder holds optional scripts for training CycleGAN models per art style.

## Data layout
```
data/
  <style>/
    trainA/  # domain A (photos)
    trainB/  # domain B (art images for <style>)
    testA/
    testB/
```

## Steps
- Curate datasets for each style (Madhubani, Tanjavur, Mughal, Warli)
- Use public CycleGAN training recipes (e.g., pytorch-CycleGAN-and-pix2pix) or adapt your own
- Export generator `G_A2B.pth` per style and place under `weights/<style>/`

## Notes
- Start with 256px or 512px resolution for faster convergence
- Consider augmentation and color jitter
- Monitor FID and sample visuals 