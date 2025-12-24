

#ifndef RK3588_DEMO_Yolov8_THREAD_POOL_H
#define RK3588_DEMO_Yolov8_THREAD_POOL_H

#include "yolov8_custom.h"

#include <iostream>
#include <vector>
#include <tuple>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>

class Yolov8ThreadPool
{
private:
    std::queue<std::tuple<int, cv::Mat, float, float, std::string>> tasks;             // <id, img, objectThreshold, nmsThreshold, preprocess_mode>用来存放任务
    std::vector<std::shared_ptr<Yolov8Custom>> Yolov8_instances;       // 模型实例
    std::map<int, std::pair<cv::Mat, std::vector<Detection>>> results; // <id, img, objects>用来存放结果（检测框）
    // std::map<int, cv::Mat> img_results;                                // <id, img>用来存放结果（图片）
    std::vector<std::thread> threads;                                  // 线程池
    std::mutex mtx1;
    std::mutex mtx2;
    std::condition_variable cv_task;
    bool stop;

    void worker(int id);

public:
    Yolov8ThreadPool();  // 构造函数
    ~Yolov8ThreadPool(); // 析构函数

    nn_error_e setUp(std::string &model_path, int input_w, int input_h, int class_num, std::vector<std::vector<int>>& mapSizeVec, int num_threads = 12); // 初始化
    nn_error_e submitTask(const cv::Mat &img, int id, float objectThreshold, float nmsThreshold, std::string preprocess_mode); // 提交任务
    nn_error_e getTargetResult(cv::Mat &img, std::vector<Detection> &objects, int id); // 获取结果（检测框）
    // nn_error_e getTargetImgResult(cv::Mat &img, int id);                 // 获取结果（图片）
    void stopAll();                                                      // 停止所有线程
};

#endif // RK3588_DEMO_Yolov8_THREAD_POOL_H
