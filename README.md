# ONNX-Jetson

## Description
Examples of deploying inference models through onnx on a Jetson device.

## Usage
Build a new image from the Dockerfile:
```
docker build -t jetson-onnxruntime-yolov4
```
### Example 1 - Object Detection
1. Download the YOLOv4 model:
```
wget https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/yolov4/model/yolov4.onnx
```

2. Run the application:
```
nvidia-docker run -it --rm -v $PWD:/workspace/ --workdir=/workspace/  jetson-onnxruntime-yolov4 python3 main.py
```

### Example 2 - Classifier
1. Download the fruits dataset [here](https://www.kaggle.com/datasets/moltean/fruits)

2. Extract the dataset in ./data/fruits

3. Download the ViT model [here](https://drive.google.com/file/d/1FQvn3N3JTgeTZUc91_vXJaGZnjnAsj5N/view?usp=sharing)

4. Run the application:
```
nvidia-docker run -it --rm -v $PWD:/workspace/ --workdir=/workspace/  jetson-onnxruntime-yolov4 python3 vit_fruits_man_onnx.py
```
