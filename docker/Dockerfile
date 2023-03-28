FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
WORKDIR .

RUN apt-get update && apt-get install -y python3-pip libprotobuf-dev protobuf-compiler
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install cmake
RUN python3 -m pip install onnx==1.6.0 easydict matplotlib scipy
COPY *.whl /workspace/
RUN cd /workspace && python3 -m pip install *.whl

CMD ["/bin/bash"]
