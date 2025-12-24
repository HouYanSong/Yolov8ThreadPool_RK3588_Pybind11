# YOLOv8 ThreadPool Pybind11 

`RK3588 YOLOv8 Pybind11 C++`线程池推理，使用`Pybind11`对外提供`Python`调用接口，`YOLOv8s`量化模型视频检测帧率超过`60FPS`。

![](./result.gif)

### 1. 构建项目
```bash
cd Yolov8ThreadPool_RK3588_Pybind11
sudo apt install pybind11-dev
pip install pybind11 
rm -fr build
cmake -S . -B build
cmake --build build
```
```bash
[  7%] Building CXX object CMakeFiles/nn_process.dir/src/process/preprocess.cpp.o
[ 14%] Building CXX object CMakeFiles/nn_process.dir/src/process/postprocess.cpp.o
[ 21%] Linking CXX shared library libnn_process.so
[ 21%] Built target nn_process
[ 28%] Building CXX object CMakeFiles/rknn_engine.dir/src/engine/rknn_engine.cpp.o
[ 35%] Linking CXX shared library librknn_engine.so
[ 35%] Built target rknn_engine
[ 42%] Building CXX object CMakeFiles/yolov8_lib.dir/src/task/yolov8_custom.cpp.o
[ 50%] Linking CXX shared library libyolov8_lib.so
[ 50%] Built target yolov8_lib
[ 57%] Building CXX object CMakeFiles/draw_lib.dir/src/draw/cv_draw.cpp.o
[ 64%] Linking CXX shared library libdraw_lib.so
[ 64%] Built target draw_lib
[ 71%] Building CXX object CMakeFiles/yolov8_pybind11.dir/src/yolov8_pybind11.cpp.o
[ 78%] Linking CXX shared module yolov8_pybind11.cpython-310-aarch64-linux-gnu.so
[ 78%] Built target yolov8_pybind11
[ 85%] Building CXX object CMakeFiles/yolov8_threadPool_pybind11.dir/src/yolov8_threadPool_pybind11.cpp.o
[ 92%] Building CXX object CMakeFiles/yolov8_threadPool_pybind11.dir/src/task/yolov8_thread_pool.cpp.o
[100%] Linking CXX shared module yolov8_threadPool_pybind11.cpython-310-aarch64-linux-gnu.so
[100%] Built target yolov8_threadPool_pybind11
```

### 2. 线程池推理
```bash
python ./yolov8_threadPool.py
```
```bash
[NN_INFO] rknn_init success!
[NN_INFO] RKNN API version: 1.5.3b6 (181ec8d8b@2023-09-12T17:11:43)
[NN_INFO] RKNN Driver version: 0.9.6
[NN_INFO] model input num: 1, output num: 6
[NN_INFO] input tensors:
[NN_INFO]   index=0, name=data, n_dims=4, dims=[1, 640, 640, 3], n_elems=1228800, size=1228800, fmt=NHWC, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003922
[NN_INFO] output tensors:
[NN_INFO]   index=0, name=reg1, n_dims=4, dims=[1, 1, 4, 6400], n_elems=25600, size=25600, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.048087
[NN_INFO]   index=1, name=cls1, n_dims=4, dims=[1, 80, 80, 80], n_elems=512000, size=512000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=115, scale=0.123865
[NN_INFO]   index=2, name=reg2, n_dims=4, dims=[1, 1, 4, 1600], n_elems=6400, size=6400, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.053342
[NN_INFO]   index=3, name=cls2, n_dims=4, dims=[1, 80, 40, 40], n_elems=128000, size=128000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=114, scale=0.178546
[NN_INFO]   index=4, name=reg3, n_dims=4, dims=[1, 1, 4, 400], n_elems=1600, size=1600, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.054902
[NN_INFO]   index=5, name=cls3, n_dims=4, dims=[1, 80, 20, 20], n_elems=32000, size=32000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=114, scale=0.206036
[NN_INFO] rknn_init success!
[NN_INFO] RKNN API version: 1.5.3b6 (181ec8d8b@2023-09-12T17:11:43)
[NN_INFO] RKNN Driver version: 0.9.6
[NN_INFO] model input num: 1, output num: 6
[NN_INFO] input tensors:
[NN_INFO]   index=0, name=data, n_dims=4, dims=[1, 640, 640, 3], n_elems=1228800, size=1228800, fmt=NHWC, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003922
[NN_INFO] output tensors:
[NN_INFO]   index=0, name=reg1, n_dims=4, dims=[1, 1, 4, 6400], n_elems=25600, size=25600, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.048087
[NN_INFO]   index=1, name=cls1, n_dims=4, dims=[1, 80, 80, 80], n_elems=512000, size=512000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=115, scale=0.123865
[NN_INFO]   index=2, name=reg2, n_dims=4, dims=[1, 1, 4, 1600], n_elems=6400, size=6400, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.053342
[NN_INFO]   index=3, name=cls2, n_dims=4, dims=[1, 80, 40, 40], n_elems=128000, size=128000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=114, scale=0.178546
[NN_INFO]   index=4, name=reg3, n_dims=4, dims=[1, 1, 4, 400], n_elems=1600, size=1600, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.054902
[NN_INFO]   index=5, name=cls3, n_dims=4, dims=[1, 80, 20, 20], n_elems=32000, size=32000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=114, scale=0.206036
[NN_INFO] rknn_init success!
[NN_INFO] RKNN API version: 1.5.3b6 (181ec8d8b@2023-09-12T17:11:43)
[NN_INFO] RKNN Driver version: 0.9.6
[NN_INFO] model input num: 1, output num: 6
[NN_INFO] input tensors:
[NN_INFO]   index=0, name=data, n_dims=4, dims=[1, 640, 640, 3], n_elems=1228800, size=1228800, fmt=NHWC, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003922
[NN_INFO] output tensors:
[NN_INFO]   index=0, name=reg1, n_dims=4, dims=[1, 1, 4, 6400], n_elems=25600, size=25600, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.048087
[NN_INFO]   index=1, name=cls1, n_dims=4, dims=[1, 80, 80, 80], n_elems=512000, size=512000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=115, scale=0.123865
[NN_INFO]   index=2, name=reg2, n_dims=4, dims=[1, 1, 4, 1600], n_elems=6400, size=6400, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.053342
[NN_INFO]   index=3, name=cls2, n_dims=4, dims=[1, 80, 40, 40], n_elems=128000, size=128000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=114, scale=0.178546
[NN_INFO]   index=4, name=reg3, n_dims=4, dims=[1, 1, 4, 400], n_elems=1600, size=1600, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.054902
[NN_INFO]   index=5, name=cls3, n_dims=4, dims=[1, 80, 20, 20], n_elems=32000, size=32000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=114, scale=0.206036
[NN_INFO] rknn_init success!
[NN_INFO] RKNN API version: 1.5.3b6 (181ec8d8b@2023-09-12T17:11:43)
[NN_INFO] RKNN Driver version: 0.9.6
[NN_INFO] model input num: 1, output num: 6
[NN_INFO] input tensors:
[NN_INFO]   index=0, name=data, n_dims=4, dims=[1, 640, 640, 3], n_elems=1228800, size=1228800, fmt=NHWC, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003922
[NN_INFO] output tensors:
[NN_INFO]   index=0, name=reg1, n_dims=4, dims=[1, 1, 4, 6400], n_elems=25600, size=25600, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.048087
[NN_INFO]   index=1, name=cls1, n_dims=4, dims=[1, 80, 80, 80], n_elems=512000, size=512000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=115, scale=0.123865
[NN_INFO]   index=2, name=reg2, n_dims=4, dims=[1, 1, 4, 1600], n_elems=6400, size=6400, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.053342
[NN_INFO]   index=3, name=cls2, n_dims=4, dims=[1, 80, 40, 40], n_elems=128000, size=128000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=114, scale=0.178546
[NN_INFO]   index=4, name=reg3, n_dims=4, dims=[1, 1, 4, 400], n_elems=1600, size=1600, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.054902
[NN_INFO]   index=5, name=cls3, n_dims=4, dims=[1, 80, 20, 20], n_elems=32000, size=32000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=114, scale=0.206036
[NN_INFO] rknn_init success!
[NN_INFO] RKNN API version: 1.5.3b6 (181ec8d8b@2023-09-12T17:11:43)
[NN_INFO] RKNN Driver version: 0.9.6
[NN_INFO] model input num: 1, output num: 6
[NN_INFO] input tensors:
[NN_INFO]   index=0, name=data, n_dims=4, dims=[1, 640, 640, 3], n_elems=1228800, size=1228800, fmt=NHWC, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003922
[NN_INFO] output tensors:
[NN_INFO]   index=0, name=reg1, n_dims=4, dims=[1, 1, 4, 6400], n_elems=25600, size=25600, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.048087
[NN_INFO]   index=1, name=cls1, n_dims=4, dims=[1, 80, 80, 80], n_elems=512000, size=512000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=115, scale=0.123865
[NN_INFO]   index=2, name=reg2, n_dims=4, dims=[1, 1, 4, 1600], n_elems=6400, size=6400, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.053342
[NN_INFO]   index=3, name=cls2, n_dims=4, dims=[1, 80, 40, 40], n_elems=128000, size=128000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=114, scale=0.178546
[NN_INFO]   index=4, name=reg3, n_dims=4, dims=[1, 1, 4, 400], n_elems=1600, size=1600, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.054902
[NN_INFO]   index=5, name=cls3, n_dims=4, dims=[1, 80, 20, 20], n_elems=32000, size=32000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=114, scale=0.206036
[NN_INFO] rknn_init success!
[NN_INFO] RKNN API version: 1.5.3b6 (181ec8d8b@2023-09-12T17:11:43)
[NN_INFO] RKNN Driver version: 0.9.6
[NN_INFO] model input num: 1, output num: 6
[NN_INFO] input tensors:
[NN_INFO]   index=0, name=data, n_dims=4, dims=[1, 640, 640, 3], n_elems=1228800, size=1228800, fmt=NHWC, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003922
[NN_INFO] output tensors:
[NN_INFO]   index=0, name=reg1, n_dims=4, dims=[1, 1, 4, 6400], n_elems=25600, size=25600, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.048087
[NN_INFO]   index=1, name=cls1, n_dims=4, dims=[1, 80, 80, 80], n_elems=512000, size=512000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=115, scale=0.123865
[NN_INFO]   index=2, name=reg2, n_dims=4, dims=[1, 1, 4, 1600], n_elems=6400, size=6400, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.053342
[NN_INFO]   index=3, name=cls2, n_dims=4, dims=[1, 80, 40, 40], n_elems=128000, size=128000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=114, scale=0.178546
[NN_INFO]   index=4, name=reg3, n_dims=4, dims=[1, 1, 4, 400], n_elems=1600, size=1600, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.054902
[NN_INFO]   index=5, name=cls3, n_dims=4, dims=[1, 80, 20, 20], n_elems=32000, size=32000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=114, scale=0.206036
[NN_INFO] rknn_init success!
[NN_INFO] RKNN API version: 1.5.3b6 (181ec8d8b@2023-09-12T17:11:43)
[NN_INFO] RKNN Driver version: 0.9.6
[NN_INFO] model input num: 1, output num: 6
[NN_INFO] input tensors:
[NN_INFO]   index=0, name=data, n_dims=4, dims=[1, 640, 640, 3], n_elems=1228800, size=1228800, fmt=NHWC, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003922
[NN_INFO] output tensors:
[NN_INFO]   index=0, name=reg1, n_dims=4, dims=[1, 1, 4, 6400], n_elems=25600, size=25600, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.048087
[NN_INFO]   index=1, name=cls1, n_dims=4, dims=[1, 80, 80, 80], n_elems=512000, size=512000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=115, scale=0.123865
[NN_INFO]   index=2, name=reg2, n_dims=4, dims=[1, 1, 4, 1600], n_elems=6400, size=6400, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.053342
[NN_INFO]   index=3, name=cls2, n_dims=4, dims=[1, 80, 40, 40], n_elems=128000, size=128000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=114, scale=0.178546
[NN_INFO]   index=4, name=reg3, n_dims=4, dims=[1, 1, 4, 400], n_elems=1600, size=1600, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.054902
[NN_INFO]   index=5, name=cls3, n_dims=4, dims=[1, 80, 20, 20], n_elems=32000, size=32000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=114, scale=0.206036
[NN_INFO] rknn_init success!
[NN_INFO] RKNN API version: 1.5.3b6 (181ec8d8b@2023-09-12T17:11:43)
[NN_INFO] RKNN Driver version: 0.9.6
[NN_INFO] model input num: 1, output num: 6
[NN_INFO] input tensors:
[NN_INFO]   index=0, name=data, n_dims=4, dims=[1, 640, 640, 3], n_elems=1228800, size=1228800, fmt=NHWC, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003922
[NN_INFO] output tensors:
[NN_INFO]   index=0, name=reg1, n_dims=4, dims=[1, 1, 4, 6400], n_elems=25600, size=25600, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.048087
[NN_INFO]   index=1, name=cls1, n_dims=4, dims=[1, 80, 80, 80], n_elems=512000, size=512000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=115, scale=0.123865
[NN_INFO]   index=2, name=reg2, n_dims=4, dims=[1, 1, 4, 1600], n_elems=6400, size=6400, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.053342
[NN_INFO]   index=3, name=cls2, n_dims=4, dims=[1, 80, 40, 40], n_elems=128000, size=128000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=114, scale=0.178546
[NN_INFO]   index=4, name=reg3, n_dims=4, dims=[1, 1, 4, 400], n_elems=1600, size=1600, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.054902
[NN_INFO]   index=5, name=cls3, n_dims=4, dims=[1, 80, 20, 20], n_elems=32000, size=32000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=114, scale=0.206036
[NN_INFO] rknn_init success!
[NN_INFO] RKNN API version: 1.5.3b6 (181ec8d8b@2023-09-12T17:11:43)
[NN_INFO] RKNN Driver version: 0.9.6
[NN_INFO] model input num: 1, output num: 6
[NN_INFO] input tensors:
[NN_INFO]   index=0, name=data, n_dims=4, dims=[1, 640, 640, 3], n_elems=1228800, size=1228800, fmt=NHWC, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003922
[NN_INFO] output tensors:
[NN_INFO]   index=0, name=reg1, n_dims=4, dims=[1, 1, 4, 6400], n_elems=25600, size=25600, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.048087
[NN_INFO]   index=1, name=cls1, n_dims=4, dims=[1, 80, 80, 80], n_elems=512000, size=512000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=115, scale=0.123865
[NN_INFO]   index=2, name=reg2, n_dims=4, dims=[1, 1, 4, 1600], n_elems=6400, size=6400, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.053342
[NN_INFO]   index=3, name=cls2, n_dims=4, dims=[1, 80, 40, 40], n_elems=128000, size=128000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=114, scale=0.178546
[NN_INFO]   index=4, name=reg3, n_dims=4, dims=[1, 1, 4, 400], n_elems=1600, size=1600, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.054902
[NN_INFO]   index=5, name=cls3, n_dims=4, dims=[1, 80, 20, 20], n_elems=32000, size=32000, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=114, scale=0.206036
=== yolov8 Meshgrid  Generate success! 
50.19 FPS
70.33 FPS
67.49 FPS
63.65 FPS
[NN_INFO] rknn context destroyed!
[NN_INFO] rknn context destroyed!
[NN_INFO] rknn context destroyed!
[NN_INFO] rknn context destroyed!
[NN_INFO] rknn context destroyed!
[NN_INFO] rknn context destroyed!
[NN_INFO] rknn context destroyed!
[NN_INFO] rknn context destroyed!
[NN_INFO] rknn context destroyed!
```


