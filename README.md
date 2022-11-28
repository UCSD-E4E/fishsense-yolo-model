Hello, this is the repo for the detection code used in FishSense. It is based on a tensorflow implementation of the Yolov4 R-CNN model. A majority of the code is setup, so that if you follow the steps below you should have a model which can detect fish on your computer. Currently the model works on a CPU, but it can be configured for a GPU. I did not do this, and reccomend looking here for help with that: https://github.com/theAIGuysCode/yolov4-custom-functions

First Install Anaconda here: https://www.anaconda.com/

After that open up the anaconda command prompt. Navigate to the directory where you pulled this repo. Once in the repo, run the follow
```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```
If you do not have anaconda, you could use Pip.

### Pip
```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```
### Nvidia Driver (For GPU, if you are not using Conda Environment and haven't set up CUDA yet)
Make sure to use CUDA Toolkit version 10.1 as it is the proper version for the TensorFlow version used in this repository.
https://developer.nvidia.com/cuda-10.1-download-archive-update2

## Now you need to download the weights for doing detection
Those weights are stored here:
https://drive.google.com/drive/folders/12-AV6rze4hCgDCYr0jh4wJzZSCHnMPQt?usp=share_link

There are 3 different set of weights there, best, region_use and species_fit .weights.

best is the a mix of various datasets to create the most adaptable weights I could
region_use is trained from using the birch aquarium dataset.
Species_fit is trained from various pictures of the same fish in an aquarium.

Once you download the weights, place them in the data folder

These two videos are helpful for understanding what is going on:

## Using Custom Trained YOLOv4 Weights
<strong>Learn How To Train Custom YOLOv4 Weights here: https://www.youtube.com/watch?v=mmj3nxGT2YQ </strong>

<strong>Watch me Walk-Through using Custom Model in TensorFlow :https://www.youtube.com/watch?v=nOIVxi5yurE </strong>

There will already be a .names folder called fish.names in the data/classes folder, but make sure it is there.
Likewise, there will a code change that shouldn't be needed, but is vital to the code running well, so check it as well. In core/config on line 14, change the .names file to fish.names if it is not already.

We also need to put images in the images directory. Navigate to data/images and place the image you wish to perform detections on.
You can also run the code on videos so place your video in the data/videos folder if you need.

Now you should have a weights file, .names file, input images and conda installed

Now we navigate back to the Conda terminal and run the following commands.

```bash
# Convert darknet weights to tensorflow
## yolov4
python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4

```
Now run the following the command with test_image changed to your image path.

```bash
# Run yolov4 tensorflow model
python detect.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --images ./data/images/test_image.jpg

# Run yolov4 on video
python detect_video.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --video ./data/video/test_video.mp4 --output ./detections/results.avi

# Run yolov4 on webcam
python detect_video.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --video 0 --output ./detections/results.avi

```
<strong>Note:</strong> You can also run the detector on multiple images at once by changing the --images flag like such ``--images "./data/images/kite.jpg, ./data/images/dog.jpg"``

<strong>Note:</strong> You can also run the detector on an entire directory using the --dir flag like such ``--dir "./data/images/"``


### Result Image(s) (Regular TensorFlow)
You can find the outputted image(s) showing the detections saved within the 'detections' folder.

### Result Video
Video saves wherever you point --output flag to. If you don't set the flag then your video will not be saved with detections on it.

Please look at the other flags in detect for more customization like conf and iou scores.

Let me know if something breaks at raghavmaddukuri@gmail.com