# robust-image-matching

A small experimental project for matching and comparing two images using classical computer vision techniques with OpenCV.

The project consists of a single Python script and implements:
- image preprocessing (white balance, denoising, contrast enhancement, sharpening),
- local feature extraction (SIFT or ORB),
- descriptor matching,
- match filtering using Lowe’s ratio test,
- geometric verification with homography (RANSAC),
- a simple heuristic probability score based on inliers,
- visualization of matches and inliers.

This code is intended for educational and experimental purposes.

## Requirements

- Python 3.x  
- OpenCV  
- NumPy  

```bash
pip install opencv-python opencv-contrib-python numpy
