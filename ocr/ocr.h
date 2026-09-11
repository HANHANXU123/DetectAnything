#ifndef OCR_H
#define OCR_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <QString>

#include "itask.h"
#include "ppocr_det.h"
#include "ppocr_rec.h"

// OCR 任务：耦合文本检测(TextDetector) + 文本识别(TextRecognizer)，实现 ITask。
// init 接收资源目录，内部拼接 det/rec engine 与字典文件名；
// run 先检测文本框，再逐框裁剪识别，最后绘制框并汇总文本。
class OCR : public ITask
{
public:
    bool init(const QString &baseDir) override;
    TaskResult run(const cv::Mat &frame) override;
    void release() override;
    bool isInitialized() const override;
    QString name() const override;

private:
    TextDetector   m_det;
    TextRecognizer m_rec;
    bool m_initialized = false;

    // 用 QImage+QPainter 绘制：只画文本框四边形 + 该框识别文本（中文正常渲染），不加序号
    cv::Mat draw(const cv::Mat &frame, const BoxArray &boxes,
                 const std::vector<std::string> &texts);
};

#endif // OCR_H
