#!/usr/bin/env bash

module load \
    cmake/cmake-3.11.1 \
    opencv/opencv-4.2.0-openmpi-3.1.3-openblas \
    openmpi/openmpi-3.1.3

cmake ..

cmake --build . --config Debug -- -j10
