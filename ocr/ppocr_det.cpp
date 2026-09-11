#include "ppocr_det.h"

bool TextDetector::init(const std::string &enginePath)
{
    return m_engine.load(enginePath);
}

void TextDetector::release()
{
    m_engine.release();
}

bool TextDetector::isInitialized() const
{
    return m_engine.isLoaded();
}

cv::Mat TextDetector::preprocess(const cv::Mat &frame)
{
    const int inputW = m_engine.inputW();   // 960
    const int inputH = m_engine.inputH();   // 960

    // 非等比 resize 到检测输入尺寸（同原 demo：resize(src, src, Size(960, 960))）
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(inputW, inputH));
    if (!resized.isContinuous())
        resized = resized.clone();

    const size_t area = (size_t)inputH * inputW;
    cv::Mat blob(1, (int)(area * 3), CV_32F);   // NCHW 连续平面缓冲
    float *base = blob.ptr<float>(0);

    // 沿用原 demo 的归一化系数与通道写入顺序（不擅自 swapRB，保证与已验证 demo 一致）：
    //   ch0 <- pixel[2]（mean[2]/std[2]），ch1 <- pixel[1]，ch2 <- pixel[0]（mean[0]/std[0]）
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float stdv[3] = {0.229f, 0.224f, 0.225f};
    float *ch0 = base + area * 0;   // demo: phostB
    float *ch1 = base + area * 1;   // demo: phostG
    float *ch2 = base + area * 2;   // demo: phostR

    const unsigned char *p = resized.data;
    for (size_t i = 0; i < area; ++i, p += 3) {
        *ch2++ = (p[0] / 255.0f - mean[0]) / stdv[0];
        *ch1++ = (p[1] / 255.0f - mean[1]) / stdv[1];
        *ch0++ = (p[2] / 255.0f - mean[2]) / stdv[2];
    }
    return blob;
}

BoxArray TextDetector::detect(const cv::Mat &frame)
{
    BoxArray boxes;
    if (!m_engine.isLoaded() || frame.empty())
        return boxes;

    cv::Mat blob = preprocess(frame);
    const float *out = m_engine.forward(blob);
    if (!out)
        return boxes;

    const int inputW = m_engine.inputW();
    const int inputH = m_engine.inputH();

    // 输出为单通道概率图 (inputH x inputW)，直接包裹锁页输出指针（不拷贝）
    cv::Mat pred_map(inputH, inputW, CV_32FC1, (void *)out);

    // src=原图、dst=检测输入，使 boxes 从概率图坐标系映射回原图坐标系
    detector_postprocess(pred_map, boxes, frame.rows, frame.cols, inputH, inputW,
                         m_maskThresh, m_boxThresh, m_unclipRatio, m_minSize, m_maxCandidates);
    return boxes;
}
