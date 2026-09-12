#include "segment.h"
#include "logger.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <QStringList>

// 按类别 id 生成稳定的调色板颜色（BGR）：同类别每次颜色一致。
// 用 HSV 均匀取色，可覆盖任意类别数（无需预置固定表）。
static cv::Scalar classColor(int classId)
{
    const int n = 20;                       // 调色板槽位数
    int idx = classId % n;
    if (idx < 0)
        idx += n;
    int hue = idx * 180 / n;                // OpenCV hue 取值 [0,180)
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 220, 255));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    cv::Vec3b c = bgr.at<cv::Vec3b>(0, 0);
    return cv::Scalar(c[0], c[1], c[2]);
}

bool SemanticSeg::init(const QString &enginePath)
{
    if (!m_engine.load(enginePath.toStdString()))
        return false;

    // 打印各输出形状与数据类型，便于核对模型 IO
    // （预期 output0=[1,960,960]，dtype=INT32 的类别索引图）
    for (int i = 0; i < m_engine.outputCount(); ++i) {
        std::vector<int> sh = m_engine.outputShape(i);
        QStringList parts;
        for (int d : sh)
            parts << QString::number(d);
        const char *dt = (m_engine.outputDataType(i) == nvinfer1::DataType::kINT32)
                             ? "INT32" : "FLOAT";
        Logger::instance().info(QString("语义分割 outputShape(%1)=[%2] dtype=%3")
                                    .arg(i).arg(parts.join(", ")).arg(dt));
    }
    return true;
}

void SemanticSeg::release()
{
    m_engine.release();
}

bool SemanticSeg::isInitialized() const
{
    return m_engine.isLoaded();
}

QString SemanticSeg::name() const
{
    return QStringLiteral("语义分割");
}

cv::Mat SemanticSeg::preprocess(const cv::Mat &frame)
{
    const int inputW = m_engine.inputW();   // 预期 960
    const int inputH = m_engine.inputH();   // 预期 960

    // 直接 resize 到网络输入尺寸（与参考脚本一致，不做 letterbox，允许宽高比形变）
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(inputW, inputH), 0, 0, cv::INTER_LINEAR);

    // 归一化：(x/255 - 0.5)/0.5 = (x - 127.5)/127.5；swapRB=true 完成 BGR→RGB。
    // blobFromImage 计算 scalefactor*(image-mean)，故 scalefactor=1/127.5、mean=127.5。
    cv::Mat blob;
    cv::dnn::blobFromImage(resized, blob, 1.0 / 127.5, cv::Size(inputW, inputH),
                           cv::Scalar(127.5, 127.5, 127.5), true, false);
    return blob;
}

cv::Mat SemanticSeg::decodeClassMap(const void *raw, nvinfer1::DataType dtype,
                                    const std::vector<int> &shape, int &outMaxClass)
{
    outMaxClass = 0;
    if (!raw)
        return cv::Mat();

    // 解析输出尺寸：本模型为 rank-3 [1,H,W]（已 argmax 的类别索引图），
    // 兼容 rank-4 [1,1,H,W]（keepdims）。
    int H = 0, W = 0;
    if (shape.size() == 3) {
        H = shape[1];
        W = shape[2];
    } else if (shape.size() == 4 && shape[1] == 1) {
        H = shape[2];
        W = shape[3];
    } else {
        return cv::Mat();
    }
    if (H <= 0 || W <= 0)
        return cv::Mat();

    // 按真实 dtype 读取类别索引：TensorRT 会把 int64 cast 成 int32，
    // 故整型输出只可能是 INT32（本模型）；另兼容少数把索引存成 float 的导出。
    const int32_t *pi32 = reinterpret_cast<const int32_t *>(raw);
    const float   *pf32 = reinterpret_cast<const float *>(raw);

    cv::Mat classMap(H, W, CV_32S);
    for (int y = 0; y < H; ++y) {
        int *dst = classMap.ptr<int>(y);
        for (int x = 0; x < W; ++x) {
            const size_t idx = (size_t)y * W + x;
            int c;
            if (dtype == nvinfer1::DataType::kINT32)
                c = pi32[idx];                     // 本模型：INT32 类别索引
            else
                c = (int)std::lround(pf32[idx]);   // 兼容：索引被存成 float
            if (c < 0)
                c = 0;
            dst[x] = c;
            if (c > outMaxClass)
                outMaxClass = c;
        }
    }
    return classMap;
}

cv::Mat SemanticSeg::draw(const cv::Mat &frame, const cv::Mat &classMapOrigSize, int maxClass)
{
    cv::Mat output = frame.clone();
    if (classMapOrigSize.empty())
        return output;

    // 半透明彩色掩膜：class 0 视作背景保持原图，其余类别按调色板上色后与原图融合
    cv::Mat overlay = frame.clone();
    for (int c = 1; c <= maxClass; ++c) {
        cv::Mat maskC = (classMapOrigSize == c);   // CV_8U 0/255
        if (cv::countNonZero(maskC) == 0)
            continue;
        overlay.setTo(classColor(c), maskC);
    }
    cv::addWeighted(overlay, 0.5, frame, 0.5, 0, output);
    return output;
}

TaskResult SemanticSeg::run(const cv::Mat &frame)
{
    TaskResult res;
    if (!m_engine.isLoaded() || frame.empty())
        return res;

    cv::Size originalSize = frame.size();

    // 分段计时：预处理/后处理在此，H2D/推理由引擎计
    auto t0 = std::chrono::high_resolution_clock::now();
    cv::Mat blob = preprocess(frame);
    auto t1 = std::chrono::high_resolution_clock::now();

    if (!m_engine.forward(blob))
        return res;
    auto t2 = std::chrono::high_resolution_clock::now();

    // 解析类别索引图（单输出模型：第 0 个即 output0=[1,960,960]，INT32 类别索引）
    const void *raw = m_engine.outputRaw(0);
    nvinfer1::DataType dtype = m_engine.outputDataType(0);
    std::vector<int> shape = m_engine.outputShape(0);
    int maxClass = 0;
    cv::Mat classMap = decodeClassMap(raw, dtype, shape, maxClass);
    if (classMap.empty()) {
        Logger::instance().error("语义分割输出为空或形状不支持，无法解析类别图");
        return res;
    }

    // 统计出现的前景类别数（排除背景 0），作为 count
    std::vector<char> present(maxClass + 1, 0);
    for (int y = 0; y < classMap.rows; ++y) {
        const int *p = classMap.ptr<int>(y);
        for (int x = 0; x < classMap.cols; ++x) {
            int c = p[x];
            if (c >= 0 && c <= maxClass)
                present[c] = 1;
        }
    }
    int distinctFg = 0;
    for (int c = 1; c <= maxClass; ++c)
        if (present[c])
            ++distinctFg;

    // 类别索引图上采样回原图尺寸：最近邻，避免插值产生非法类别 id
    cv::Mat classMapOrig;
    cv::resize(classMap, classMapOrig, originalSize, 0, 0, cv::INTER_NEAREST);
    auto t3 = std::chrono::high_resolution_clock::now();

    double msPre  = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double msPost = std::chrono::duration<double, std::milli>(t3 - t2).count();
    Logger::instance().info(QString("耗时分解 预处理:%1 H2D:%2 推理:%3 后处理:%4 ms")
                                .arg(msPre, 0, 'f', 1)
                                .arg(m_engine.lastH2DMs(), 0, 'f', 1)
                                .arg(m_engine.lastInferMs(), 0, 'f', 1)
                                .arg(msPost, 0, 'f', 1));

    // inferMs 与其它任务语义一致：预处理+GPU+后处理，不含绘制
    res.inferMs  = std::chrono::duration<double, std::milli>(t3 - t0).count();
    res.rendered = draw(frame, classMapOrig, maxClass);
    res.count    = distinctFg;
    res.ok       = true;
    Logger::instance().info(QString("语义分割 前景类别 %1 种（最大类 id %2），推理 %3 ms")
                                .arg(res.count).arg(maxClass).arg(res.inferMs, 0, 'f', 1));
    return res;
}
