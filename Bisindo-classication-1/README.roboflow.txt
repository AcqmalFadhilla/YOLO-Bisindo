
Bisindo classication - v1 2024-01-01 7:31pm
==============================

This dataset was exported via roboflow.com on January 1, 2024 at 1:40 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 100 images.
A are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* Random brigthness adjustment of between -27 and +27 percent

The following transformations were applied to the bounding boxes of each image:
* Random rotation of between -5 and +5 degrees
* Salt and pepper noise was applied to 0.11 percent of pixels


