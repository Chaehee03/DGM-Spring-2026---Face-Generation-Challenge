# DGM Face Generation Challenge

This repository contains the reproducibility code for my DGM Spring 2026 Face Generation Challenge submission.

## Method

The final approach fine-tunes StyleGAN2-ADA from an FFHQ 256x256 pretrained checkpoint on CelebV-HQ raw frame datasets. The submitted images were generated using fixed checkpoints, fixed seed ranges, and truncation/checkpoint mixture strategies.

## Final Best Submission

Best public submission:
- FID: 32.9529
- IS: 4.2546
- KID: 0.0058
- TopPR: 0.8284

The best submission used a sample-level checkpoint mixture:
- 900 images from the 3-frame k5500 checkpoint
- 100 images from the 3-frame k5868 checkpoint
- overall truncation mixture: 750 images at psi=0.90 and 250 images at psi=0.95

## Reproducibility

The generation scripts use fixed seed ranges and deterministic StyleGAN2-ADA inference with `noise-mode=const`.

Large files such as checkpoints, datasets, generated images, and submission zip files are not included in this repository.
