# ONNX-Jetson

## Description
Examples of deploying inference models through onnx on a Jetson device.

## Usage
### Building an Image
Build a new image from the Dockerfile:
```bash
./BUILD-DOCKER-IMAGE.sh # (recommended)
```
or
```bash
docker build -t jetson-onnxruntime-yolov4
```
### Running Scripts
TWO options:
1. Run the scripts in standalone mode
2. Execute ``` ./RUN-DOCKER.sh ``` to start a container to run the scripts

### Example 1 - Object Detection

1. Download the YOLOv4 model [here](https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/yolov4/model/yolov4.onnx), save to ./onnx/

2. Run the application
- Standalone:
```bash
cd yolov4/
nvidia-docker run -it --rm -v $PWD:/workspace/ --workdir=/workspace/  jetson-onnxruntime-yolov4 python3 yolov4.py
```
- In the container:
```bash
cd ~/ros_ws/src/glozzom/yolov4/
python3 yolov4.py
```

### Example 2 - Classifier
1. Download the fruits dataset [here](https://www.kaggle.com/datasets/moltean/fruits), save to ./data/

2. Extract the dataset into ./data/fruits

3. Download the ViT model [here](https://drive.google.com/file/d/1FQvn3N3JTgeTZUc91_vXJaGZnjnAsj5N/view?usp=sharing), save to ./onnx/

4. Run the application
- Standalone:
```bash
cd vit/
nvidia-docker run -it --rm -v $PWD:/workspace/ --workdir=/workspace/  jetson-onnxruntime-yolov4 python3 vit_fruits_man_onnx.py
```
- In the container:
```bash
cd ~/ros_ws/src/glozzom/vit/
python3 vit_fruits_man_onnx.py
```
