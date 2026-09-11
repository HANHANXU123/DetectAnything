#ifndef PPOCR_DET_H
#define PPOCR_DET_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "trtengine.h"
#include "ppocr_utils.h"   // BoxArray、detector_postprocess

// 文本检测（DB 算法）：组合通用 TrtEngine，只实现 DB 专属的预处理/后处理。
// 输入 960x960（非等比 resize，同原 demo），输出单通道概率图 →
// detector_postprocess 还原文本框，并映射回原图坐标系。
class TextDetector
{
public:
    bool init(const std::string &enginePath);
    void release();
    bool isInitialized() const;

    // 在原图 frame 上检测文本框；返回的框坐标已映射回原图坐标系
    BoxArray detect(const cv::Mat &frame);

private:
    TrtEngine m_engine;

    // DB 后处理阈值（沿用原 demo OcrParameter 默认值）
    float m_maskThresh    = 0.3f;
    float m_boxThresh     = 0.6f;
    float m_unclipRatio   = 1.5f;
    int   m_minSize       = 3;
    int   m_maxCandidates = 1000;

    // resize 到检测输入 + ImageNet 归一化 → NCHW float blob
    cv::Mat preprocess(const cv::Mat &frame);
};

#endif // PPOCR_DET_H
