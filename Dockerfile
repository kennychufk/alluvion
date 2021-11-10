FROM nvidia/cuda:11.4.2-devel-ubuntu20.04 AS build

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      wget \
      python3 \
      python3-pip \
      python3-dev \
      python3-pybind11 \
      libeigen3-dev \
      doctest-dev \
      pybind11-dev \
      libglfw3-dev \
      libglm-dev \
      libfreetype-dev && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null && \
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null && \
    apt-get update && apt-get install -y --no-install-recommends \
      cmake && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY requirements.txt /work
RUN pip3 install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html

COPY . /work

RUN python3 setup.py install

FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 \
      python3 \
      python3-pkg-resources \
      libglfw3 \
      libfreetype6 && \
    rm -rf /var/lib/apt/lists/*
## copy all python packages
COPY --from=build /usr/local/lib/python3.8/dist-packages/ /usr/local/lib/python3.8/dist-packages/
