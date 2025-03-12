#!/bin/bash

echo "Converting depth_anything onnx model to tensorrt ..."
/usr/src/tensorrt/bin/trtexec --onnx=/workspace/models/depth_anything_v2_vits.onnx \
                              --saveEngine=/workspace/models/depth_anything_v2_vits.engine \
                              --fp16