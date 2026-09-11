#ifndef YOLO_H
#define YOLO_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "itask.h"
#include "trtengine.h"

// 单个检测目标（YOLO 输出解析结果）
struct Detection
{
    cv::Rect box;
    float confidence;
    int classId;
};

// 目标检测任务：组合通用 TrtEngine，只实现 YOLO 专属的预处理/后处理/绘制。
// 模型输出格式 [1,N,6]（end2end 内置 NMS），输入 640x640 letterbox（填充 114）。
class YoloDetector : public ITask
{
public:
    bool init(const QString &enginePath) override;
    TaskResult run(const cv::Mat &frame) override;
    void release() override;
    bool isInitialized() const override;
    QString name() const override;

private:
    TrtEngine m_engine;

    cv::Mat preprocess(const cv::Mat &frame);
    std::vector<Detection> postprocess(const float *outputData, int numDetections,
                                        const cv::Size &originalSize,
                                        float confThreshold = 0.25f);
    cv::Mat draw(const cv::Mat &frame, const std::vector<Detection> &dets);
};

#endif // YOLO_H
