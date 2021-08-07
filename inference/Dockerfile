FROM nvcr.io/nvidia/l4t-tensorflow:r32.5.0-tf1.15-py3


RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install Pillow
RUN pip3 install scipy
RUN pip3 install sklearn
RUN pip3 install matplotlib
RUN apt-get install -y python3-opencv
RUN pip3 install configparser
RUN pip3 install paho-mqtt

ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

WORKDIR /CrowdDetection

CMD [ "python3.6", "./trt.py" ]
