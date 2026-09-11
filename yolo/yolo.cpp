#include "yolo.h"
#include "logger.h"

#include <algorithm>
#include <chrono>

// COCO 80 类别名称（目标检测专属，绘制标签用）
static const char *kCocoClasses[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
};

bool YoloDetector::init(const QString &enginePath)
{
    return m_engine.load(enginePath.toStdString());
}

void YoloDetector::release()
{
    m_engine.release();
}

bool YoloDetector::isInitialized() const
{
    return m_engine.isLoaded();
}

QString YoloDetector::name() const
{
    return QStringLiteral("目标检测");
}

cv::Mat YoloDetector::preprocess(const cv::Mat &frame)
{
    const int inputW = m_engine.inputW();
    const int inputH = m_engine.inputH();

    int w = frame.cols, h = frame.rows;
    float r = std::min((float)inputW / w, (float)inputH / h);
    int newW = (int)(w * r), newH = (int)(h * r);
    int padW = (inputW - newW) / 2;
    int padH = (inputH - newH) / 2;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(newW, newH));
    cv::Mat padded(inputH, inputW, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(padW, padH, newW, newH)));

    cv::Mat blob;
    cv::dnn::blobFromImage(padded, blob, 1.0 / 255.0,
                           cv::Size(inputW, inputH),
                           cv::Scalar(0, 0, 0), true, false);
    return blob;
}

std::vector<Detection> YoloDetector::postprocess(const float *outputData, int numDetections,
                                                  const cv::Size &originalSize,
                                                  float confThreshold)
{
    const int inputW = m_engine.inputW();
    const int inputH = m_engine.inputH();

    int w = originalSize.width, h = originalSize.height;
    float r = std::min((float)inputW / w, (float)inputH / h);
    int newW = (int)(w * r), newH = (int)(h * r);
    int padW = (inputW - newW) / 2;
    int padH = (inputH - newH) / 2;

    std::vector<Detection> result;
    for (int i = 0; i < numDetections; ++i) {
        const float *row = outputData + i * 6;
        float conf = row[4];
        if (conf < confThreshold)
            continue;

        float x1 = (row[0] - padW) / r;
        float y1 = (row[1] - padH) / r;
        float x2 = (row[2] - padW) / r;
        float y2 = (row[3] - padH) / r;

        x1 = std::max(0.0f, std::min(x1, (float)w - 1));
        y1 = std::max(0.0f, std::min(y1, (float)h - 1));
        x2 = std::max(0.0f, std::min(x2, (float)w - 1));
        y2 = std::max(0.0f, std::min(y2, (float)h - 1));

        Detection d;
        d.box = cv::Rect(cv::Point((int)x1, (int)y1), cv::Point((int)x2, (int)y2));
        d.confidence = conf;
        d.classId = (int)row[5];
        result.push_back(d);
    }
    return result;
}

TaskResult YoloDetector::run(const cv::Mat &frame)
{
    TaskResult res;
    if (!m_engine.isLoaded() || frame.empty())
        return res;

    cv::Size originalSize = frame.size();

    // 分段计时：定位耗时波动来源（预处理/后处理在此，H2D/推理由引擎计）
    auto t0 = std::chrono::high_resolution_clock::now();
    cv::Mat blob = preprocess(frame);
    auto t1 = std::chrono::high_resolution_clock::now();

    const float *out = m_engine.forward(blob);
    if (!out)
        return res;
    auto t2 = std::chrono::high_resolution_clock::now();

    std::vector<int> shape = m_engine.outputShape();
    int numDetections = (shape.size() >= 2) ? shape[1] : 0;
    std::vector<Detection> dets = postprocess(out, numDetections, originalSize);
    auto t3 = std::chrono::high_resolution_clock::now();

    double msPre  = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double msPost = std::chrono::duration<double, std::milli>(t3 - t2).count();
    Logger::instance().info(QString("耗时分解 预处理:%1 H2D:%2 推理:%3 后处理:%4 ms")
                                .arg(msPre, 0, 'f', 1)
                                .arg(m_engine.lastH2DMs(), 0, 'f', 1)
                                .arg(m_engine.lastInferMs(), 0, 'f', 1)
                                .arg(msPost, 0, 'f', 1));

    // inferMs 与改造前语义一致：预处理+GPU+后处理，不含绘制
    res.inferMs  = std::chrono::duration<double, std::milli>(t3 - t0).count();
    res.rendered = draw(frame, dets);
    res.count    = (int)dets.size();
    res.ok       = true;
    return res;
}

cv::Mat YoloDetector::draw(const cv::Mat &frame, const std::vector<Detection> &dets)
{
    cv::Mat output = frame.clone();

    // 标注尺寸随图片分辨率自适应（同 OCR）：以长边 1000px 为基准 s
    const double s = std::max(1.0, std::max(frame.cols, frame.rows) / 1000.0);
    const int lineW = std::max(2, (int)(2.5 * s));   // 框线宽

    for (const auto &d : dets) {
        cv::Scalar color(0, 255, 128);
        cv::rectangle(output, d.box, color, lineW);

        QString label;
        if (d.classId >= 0 && d.classId < 80) {
            label = QString("%1 %2%").arg(kCocoClasses[d.classId]).arg((int)(d.confidence * 100));
        } else {
            label = QString("cls_%1 %2%").arg(d.classId).arg((int)(d.confidence * 100));
        }

        int baseline = 0;
        double fontScale = 0.9 * s;                  // 字号随分辨率放大
        int thickness = std::max(1, (int)(1.5 * s)); // 字粗
        cv::Size textSize = cv::getTextSize(label.toStdString(), cv::FONT_HERSHEY_SIMPLEX,
                                             fontScale, thickness, &baseline);
        cv::Point textOrg(d.box.x, d.box.y - textSize.height - 2);
        cv::rectangle(output, cv::Rect(textOrg.x - 2, textOrg.y - 2,
                                        textSize.width + 4, textSize.height + 6),
                      color, cv::FILLED);
        cv::putText(output, label.toStdString(), cv::Point(textOrg.x, textOrg.y + textSize.height),
                    cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 0, 0), thickness);
    }

    return output;
}
