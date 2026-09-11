#ifndef PPOCR_REC_H
#define PPOCR_REC_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "trtengine.h"

// 文本识别（CTC）：组合通用 TrtEngine，只实现识别专属的预处理/CTC 解码。
// 输入 48x640（self_resize letterbox），输出 [1,maxChars,vocab] → CTC 贪心解码。
class TextRecognizer
{
public:
    bool init(const std::string &enginePath, const std::string &dictPath);
    void release();
    bool isInitialized() const;

    // 识别单行文本裁剪图（内部 self_resize 到模型输入尺寸），返回识别文本
    std::string recognize(const cv::Mat &crop);

private:
    TrtEngine m_engine;
    std::vector<std::string> m_labelList;   // ["#"] + dict + [" "]，大小须 == vocab

    bool readDict(const std::string &dictPath);
    cv::Mat preprocess(const cv::Mat &crop);   // self_resize + 归一化 → NCHW blob
    std::string ctcDecode(const float *out, int maxChars, int vocab) const;
};

#endif // PPOCR_REC_H
