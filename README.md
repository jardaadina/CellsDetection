# Cell Detection and Counting in Microscopic Samples

A computer vision project for automated detection and counting of cells in microscopic images using classical image processing techniques.

## Overview

This project implements two approaches for cell segmentation:
1. **Classical Method**: Basic contour detection without cell separation
2. **Watershed Algorithm**: Advanced method for separating overlapping/touching cells

## Features

- Automated cell detection in microscopic images
- Support for overlapping cell separation using watershed transform
- Complete image processing pipeline from preprocessing to final counting
- Comparative analysis between different segmentation methods

## Methods

### Classical Approach
```
Color Image → Grayscale → Median Filter → Adaptive Binarization → 
Connected Components → Contour Detection → Final Result
```

### Watershed Approach
```
Color Image → Mean Shift Filtering → Grayscale → Binarization → 
Dilation (Sure Background) → Distance Transform → 
Marker Detection → Watershed Algorithm → Region Coloring → Final Result
```

## Key Components

1. **Color to Grayscale Conversion**: `Gray(i,j) = (B(i,j) + G(i,j) + R(i,j)) / 3`
2. **Median Filtering**: Noise reduction using median filter
3. **Adaptive Binarization**: Iterative threshold determination
4. **Connected Component Labeling**: BFS-based component identification
5. **Watershed Transform**: Distance-based cell separation

## Results

- **Classical Method**: 100% detection accuracy on test images
- **Watershed Method**: 73% detection accuracy due to preprocessing limitations

### Identified Issues with Watershed Implementation
- Aggressive preprocessing (mean shift filtering)
- Global OTSU binarization
- Restrictive threshold (0.4) in distance transform
- Loss of small and irregular cells

## Future Improvements

- Integration of deep learning techniques for marker detection
- Extension to temporal analysis (time-lapse) for cell tracking
- Real-time processing optimization
- Applications in automated pathological tissue diagnosis

## Technical Requirements

- OpenCV library
- C++ development environment
- Image processing capabilities
