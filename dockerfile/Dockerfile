FROM registry.cn-hangzhou.aliyuncs.com/feidieufo/zhr_rl:base
# Add symbol link for libcuda.so
RUN ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so \
          /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so.1

RUN apt-get update && \
    apt-get -y install software-properties-common && \
    add-apt-repository -y ppa:jonathonf/python-3.6 && \
    apt-get update && \
    apt-get -y install python3.6 && \
    cd /usr/bin/ && \
    rm python3 && \
    ln -s python3.6 python3 && \
    apt-get -y install python3-pip
RUN pip3 install torch==1.1.0 torchvision==0.3.0 -f https://download.pytorch.org/whl/cu90/torch_stable.html
RUN pip3 install imgaug
ENV LC_ALL C

WORKDIR /rl