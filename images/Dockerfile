FROM ubuntu:22.04
LABEL Description="Build environment for neural network compilers"

ENV HOME /root

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get -y --no-install-recommends install \
    build-essential \
    clang \
    cmake \
    wget \
    git \
    python3 \
    python3-pip

RUN python3 -m pip install numpy tensorflow

# inspired by https://github.com/Dobiasd/frugally-deep/blob/master/INSTALL.md
RUN mkdir /my_ws
WORKDIR /my_ws
RUN git clone -c http.sslverify=false -b 'v0.2.23' --single-branch --depth 1 https://github.com/Dobiasd/FunctionalPlus && \
    git clone -c http.sslverify=false -b 'v3.11.3' --single-branch --depth 1 https://github.com/nlohmann/json && \
    git clone -c http.sslverify=false https://github.com/Dobiasd/frugally-deep && \
    git clone -c http.sslverify=false -b '3.4.0' --single-branch --depth 1 https://gitlab.com/libeigen/eigen.git

# create build folders
RUN mkdir -p /my_ws/FunctionalPlus/build && mkdir -p /my_ws/eigen/build && \
    mkdir -p /my_ws/json/build && mkdir -p /my_ws/frugally-deep/build

# compile fplus
WORKDIR /my_ws/FunctionalPlus/build
RUN cmake ..
RUN make && make install

# compile eigen
WORKDIR /my_ws/eigen/build
RUN cmake ..
RUN make && make install && ln -s /usr/local/include/eigen3/Eigen /usr/local/include/Eigen

# compile json
WORKDIR /my_ws/json/build
RUN cmake -DJSON_BuildTests=OFF ..
RUN make && make install

# compile frugally deep
WORKDIR /my_ws/frugally-deep/build
RUN cmake ..
RUN make && make install

# TODO fix final Workdir
# TODO make conversion script useable
WORKDIR /opt/
RUN cp /my_ws/frugally-deep/keras_export/convert_model.py /opt
COPY create_test_model.py /opt/
RUN python3 create_test_model.py

# TODO make it possible to run inference as test