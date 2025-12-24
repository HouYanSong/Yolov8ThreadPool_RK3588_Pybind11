#include <opencv2/opencv.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "task/yolov8_custom.h"
#include "utils/logging.h"
#include "draw/cv_draw.h"
#include <tuple>

#include "task/yolov8_thread_pool.h"

namespace py = pybind11;

cv::Mat numpy_to_mat(py::array_t<unsigned char>& input) {
    py::buffer_info buf_info = input.request();

    int height = buf_info.shape[0];
    int width = buf_info.shape[1];

    cv::Mat mat(height, width, CV_8UC3, (unsigned char*)buf_info.ptr);
    return mat.clone();
}

py::array_t<unsigned char> mat_to_numpy(cv::Mat& mat) {
    auto buffer_info = py::buffer_info(
        mat.data,
        sizeof(unsigned char),
        py::format_descriptor<unsigned char>::format(),
        3,
        { mat.rows, mat.cols, mat.channels() },
        { sizeof(unsigned char) * mat.cols * mat.channels(),
            sizeof(unsigned char) * mat.channels(),
            sizeof(unsigned char) }
    );
    return py::array_t<unsigned char>(buffer_info);
}

class Yolov8Wrapper_ThreadPool {
private:
    Yolov8ThreadPool *g_pool;
    int g_frame_start_id = 0;
    int g_frame_end_id = 0;
public:
    Yolov8Wrapper_ThreadPool(std::string& model_path, const int input_w, const int input_h, const int class_num, std::vector<std::vector<int>>& mapSizeVec, int num_threads) {
        g_pool = new Yolov8ThreadPool();
        g_pool->setUp(model_path, input_w, input_h, class_num, mapSizeVec, num_threads);
    }
    ~Yolov8Wrapper_ThreadPool() {
        g_pool->stopAll();
        delete g_pool;
    }

    void submitImg(py::array_t<unsigned char>& input_array, float objectThreshold, float nmsThreshold, std::string preprocess_mode) {
        cv::Mat img = numpy_to_mat(input_array);
        g_pool->submitTask(img.clone(), g_frame_start_id++, objectThreshold, nmsThreshold, preprocess_mode);
    }

    std::tuple<py::array_t<unsigned char>, std::vector<Detection>> getResult() {
        cv::Mat img;
        std::vector<Detection> objects;
        g_pool->getTargetResult(img, objects, g_frame_end_id++);
        py::array_t<unsigned char> numpy_img = mat_to_numpy(img);
        return std::make_tuple(numpy_img, objects);
    }
};


PYBIND11_MODULE(yolov8_threadPool_pybind11, m) {
    m.doc() = "Yolov8 ThreadPool module";

    py::class_<Detection>(m, "Detection")
        .def(py::init<>())
        .def_readwrite("class_id", &Detection::class_id)
        .def_readwrite("confidence", &Detection::confidence)
        .def_readwrite("box", &Detection::box);
        
    py::class_<cv::Rect>(m, "Rect")
        .def(py::init<>())
        .def_readwrite("x", &cv::Rect::x)
        .def_readwrite("y", &cv::Rect::y)
        .def_readwrite("width", &cv::Rect::width)
        .def_readwrite("height", &cv::Rect::height);

    py::class_<Yolov8Wrapper_ThreadPool>(m, "Yolov8ThreadPool")
        .def(py::init<std::string&, const int, const int, const int, std::vector<std::vector<int>>&, int>())
        .def("submitImg", &Yolov8Wrapper_ThreadPool::submitImg, "Submit image for detection")
        .def("getResult", &Yolov8Wrapper_ThreadPool::getResult, "Get detection result");
}


