# Fast Inference

这是一个用于快速推理的C++项目，使用YOLOv5作为示例模型。该项目使用Conan进行依赖管理，支持在Linux开发板上运行。

## 依赖要求

- CMake >= 3.15
- Conan >= 2.0
- C++17 兼容的编译器
- Linux 操作系统

## 安装步骤

1. 安装Conan（如果尚未安装）：
```bash
pip install conan
```

2. 安装依赖：
```bash
conan install . --output-folder=build --build=missing
```

3. 构建项目：
```bash
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

## 使用方法

1. 首先需要将YOLOv5模型转换为TorchScript格式：
```python
import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.eval()
traced_script_module = torch.jit.script(model)
traced_script_module.save("yolov5s.torchscript")
```

2. 运行推理程序：
```bash
./fast_inference yolov5s.torchscript test_image.jpg
```

## 注意事项

- 确保开发板有足够的内存运行模型
- 可以根据需要调整输入图像大小和推理参数
- 建议使用Release模式编译以获得更好的性能

## 性能优化

- 使用OpenMP进行并行计算
- 使用CUDA进行GPU加速（如果硬件支持）
- 使用量化技术减小模型大小
- 使用TensorRT进行优化（如果硬件支持） 