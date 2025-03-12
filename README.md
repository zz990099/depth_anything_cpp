# depth_anything_cpp

## About This Project

该项目是`DepthAnything`算法的c++实现，包括`TensorRT`、`RKNN`、`OnnxRuntime`三种硬件平台(推理引擎).

## Features

1. 支持多种推理引擎: `TensorRT`、`RKNN`、`OnnxRuntime`
2. 支持异步、多核推理，算法吞吐量较高，特别是`RK3588`平台

## Demo

| <img src="./assets/image.png" alt="1" width="500"> | <img src="./assets/depth_color.png" alt="1" width="500"> |
|:----------------------------------------:|:----:|
| **left image**  | **disp in color** |

|  nvidia-3080-laptop   |   qps   |  cpu   |
|:---------:|:---------:|:----------------:|
|  depth_anything_v2(fp16)   |   **197**   |  120%   |
|  depth_anything_v2(fp16) - ***async***  |   **208**   |  180%   |


|  jetson-orin-nx-16GB   |   qps   |  cpu   |
|:---------:|:---------:|:----------------:|
|  depth_anything_v2(fp16)   |   **39**   |  27%   |
|  depth_anything_v2(fp16) - ***async***  |   **41**   |  34%   |


|  orangepi-5-plus-16GB   |   qps   |  cpu   |
|:---------:|:---------:|:----------------:|
|  depth_anything_v2(fp16)   |   **1.2**   |  14%   |
|  depth_anything_v2(fp16) - ***async***  |   **3.2**   |  33%   |

|  intel-i7-11800H   |   qps   |  cpu   |
|:---------:|:---------:|:----------------:|
|  depth_anything_v2(fp32)   |   **3.9**   |  800%   |

## Usage

### Download Project

下载git项目
```bash
git clone git@github.com:zz990099/depth_anything_cpp.git
cd depth_anything_cpp
git submodule init && git submodule update
```

### Build Enviroment

使用docker构建工作环境
```bash
cd depth_anything_cpp
bash easy_deploy_tool/docker/build_docker.sh --platform=jetson_trt8_u2004 # or jetson_trt8_u2204, nvidia_gpu, rk3588
bash easy_deploy_tool/docker/into_docker.sh
```

### Compile Codes

在docker容器内，编译工程. 使用 `-DENABLE_*`宏来启用某种推理框架，可用的有: `-DENABLE_TENSORRT=ON`、`-DENABLE_RKNN=ON`、`-DENABLE_ORT=ON`，可以兼容。 
```bash
cd /workspace
mdkir build && cd build
cmake .. -DBUILD_TESTING=ON -DENABLE_TENSORRT=ON
make -j
```

### Convert Model

在docker容器内，运行模型转换脚本
```bash
cd /workspace
bash tools/cvt_onnx2trt.sh
```

### Run Test Cases

运行测试用例，具体测试用例请参考代码。
```bash
cd /workspace/build
./bin/simple_tests --gtest_filter=*correctness
# 限制GLOG输出
GLOG_minloglevel=1 ./bin/simple_tests --gtest_filter=*speed
```

## References

- [DepthAnythingV1](https://github.com/LiheYoung/Depth-Anything)
- [DepthAnythingV1](https://github.com/LiheYoung/Depth-Anything)
- [EasyDeployTool](https://github.com/zz990099/EasyDeployTool)