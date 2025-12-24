#include <opencv2/opencv.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "task/yolov8_custom.h"
#include "utils/logging.h"
#include "draw/cv_draw.h"
#include "types/yolo_datatype.h"

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

class Yolov8Wrapper {
private:
    Yolov8Custom yolo;

public:
    Yolov8Wrapper(const std::string& model_path, const int input_w, const int input_h, const int class_num, const std::vector<std::vector<int>>& mapSizeVec) {
        yolo.LoadModel(model_path.c_str(), input_w, input_h, class_num, mapSizeVec);
    }

    std::vector<Detection> detect(py::array_t<unsigned char>& input_array, float objectThreshold, float nmsThreshold, std::string preprocess_mode) {
        cv::Mat img = numpy_to_mat(input_array);
        std::vector<Detection> objects;
        yolo.Run(img, objects, objectThreshold, nmsThreshold, preprocess_mode);
        return objects;
    }
};

PYBIND11_MODULE(yolov8_pybind11, m) {
    m.doc() = "Yolov8 detection module";
    
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
    
    py::class_<Yolov8Wrapper>(m, "Yolov8Detector")
        .def(py::init<const std::string&, const int, const int, const int, const std::vector<std::vector<int>>&>())
        .def("detect", &Yolov8Wrapper::detect, "Detect objects in image");
}