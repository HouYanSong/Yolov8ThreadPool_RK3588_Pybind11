#ifndef RK3588_DEMO_YOLOV8_CUSTOM_H
#define RK3588_DEMO_YOLOV8_CUSTOM_H

#include "engine/engine.h"

#include <memory>

#include <opencv2/opencv.hpp>
#include "process/preprocess.h"
#include "types/yolo_datatype.h"

class Yolov8Custom
{
public:
    Yolov8Custom();
    ~Yolov8Custom();

    nn_error_e LoadModel(const char *model_path, const int input_w, const int input_h, const int class_num, const std::vector<std::vector<int>>& mapSizeVec);

    nn_error_e Run(const cv::Mat &img, std::vector<Detection> &objects, float objectThreshold, float nmsThreshold, std::string preprocess_mode);

private:
    nn_error_e Preprocess(const cv::Mat &img, const std::string process_type, cv::Mat &image_letterbox);
    nn_error_e Inference();
    nn_error_e Postprocess(const cv::Mat &img, std::vector<Detection> &objects, float objectThreshold, float nmsThreshold);

    bool ready_;
    LetterBoxInfo letterbox_info_;
    tensor_data_s input_tensor_;
    std::vector<tensor_data_s> output_tensors_;
    bool want_float_;
    std::vector<int32_t> out_zps_;
    std::vector<float> out_scales_;
    std::shared_ptr<NNEngine> engine_;
    int input_w;
    int input_h;
    int class_num;
    int mapSize[3][2];
};

#endif // RK3588_DEMO_YOLOV8_CUSTOM_H
