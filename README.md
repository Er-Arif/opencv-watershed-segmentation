# OpenCV Watershed Image Segmentation

An educational computer-vision project demonstrating classical image segmentation with **OpenCV**, **NumPy** and marker-based watershed.

## What this project covers

1. Grayscale conversion and Otsu's automatic binary threshold
2. Morphological opening to clean foreground noise
3. Distance transform for probable object centers
4. Connected-component markers and unknown-region detection
5. Watershed boundaries drawn over the input image

This is **classical computer vision**, not a trained deep-learning model. The original college experiment is retained in Git history and as `exp3.py`; `main.py` is a portable CLI version.

## Setup and run

Python 3.10+ recommended.

```bash
python -m pip install -r requirements.txt
python main.py --input path/to/objects.jpg --output output/segmented.png --mask output/threshold.png
python -m unittest discover -s tests -v
```

The default thresholding expects relatively dark objects on a light background. Try `--no-invert` for light objects on dark backgrounds, and tune `--foreground-ratio` if watershed markers merge or split too aggressively.

## Outputs

- `segmented.png`: original image with watershed boundaries highlighted in red
- `threshold.png`: the Otsu-threshold binary image
- Existing `thresh.png`: output of the earlier notebook-style experiment, preserved for reference

## Reproducibility

Images are supplied using CLI paths instead of hardcoded local Windows directories. No proprietary dataset, pretrained weights or GPU is required. Results depend on image contrast and threshold choice.

## Future work

Compare morphological parameters on multiple samples, report segmentation quality using labeled masks, and add a small curated example dataset with reuse permission.
