#include "yoloseg.h"
#include "logger.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <QStringList>

// COCO 80 类别名称（实例分割标签用，与检测一致；独立子目录自包含）
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

// 按类别取稳定的调色板颜色（BGR），同类别每次颜色一致
static cv::Scalar classColor(int classId)
{
    static const cv::Scalar palette[] = {
        cv::Scalar(56, 56, 255),   cv::Scalar(151, 157, 255), cv::Scalar(31, 112, 255),
        cv::Scalar(29, 178, 255),  cv::Scalar(49, 210, 207),  cv::Scalar(10, 249, 72),
        cv::Scalar(23, 204, 146),  cv::Scalar(134, 219, 61),  cv::Scalar(52, 147, 26),
        cv::Scalar(187, 212, 0),   cv::Scalar(168, 153, 44),  cv::Scalar(255, 194, 0),
        cv::Scalar(147, 69, 52),   cv::Scalar(255, 115, 100), cv::Scalar(236, 24, 0),
        cv::Scalar(255, 56, 132),  cv::Scalar(133, 0, 82),    cv::Scalar(255, 56, 203),
        cv::Scalar(200, 149, 255), cv::Scalar(198, 55, 255)
    };
    const int n = (int)(sizeof(palette) / sizeof(palette[0]));
    int idx = classId % n;
    if (idx < 0)
        idx += n;
    return palette[idx];
}

// sigmoid：系数 x 原型得到的是 logits，必须过 sigmoid 再阈值，否则掩膜全白/全黑
static inline float sigmoidf(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

bool YoloSeg::init(const QString &enginePath)
{
    if (!m_engine.load(enginePath.toStdString()))
        return false;

    // 打印各输出形状，便于核对模型 IO（预期 det=[1,N,38]、proto=[1,32,160,160]）
    for (int i = 0; i < m_engine.outputCount(); ++i) {
        std::vector<int> sh = m_engine.outputShape(i);
        QStringList parts;
        for (int d : sh)
            parts << QString::number(d);
        Logger::instance().info(QString("实例分割 outputShape(%1)=[%2]")
                                    .arg(i).arg(parts.join(",")));
    }
    return true;
}

void YoloSeg::release()
{
    m_engine.release();
}

bool YoloSeg::isInitialized() const
{
    return m_engine.isLoaded();
}

QString YoloSeg::name() const
{
    return QStringLiteral("实例分割");
}

cv::Mat YoloSeg::preprocess(const cv::Mat &frame)
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

std::vector<SegInstance> YoloSeg::postprocess(const float *detOut, const float *protoOut,
                                              const std::vector<int> &detShape,
                                              const std::vector<int> &protoShape,
                                              const cv::Size &originalSize,
                                              float confThreshold, float maskThreshold)
{
    std::vector<SegInstance> result;
    if (!detOut || !protoOut)
        return result;

    // 解析形状：det=[1,N,fields]、proto=[1,C,mH,mW]
    const int N      = (detShape.size() >= 2) ? detShape[1] : 0;
    const int fields = (detShape.size() >= 3) ? detShape[2] : 0;
    const int C      = (protoShape.size() >= 2) ? protoShape[1] : 0;
    const int mH     = (protoShape.size() >= 3) ? protoShape[2] : 0;
    const int mW     = (protoShape.size() >= 4) ? protoShape[3] : 0;
    if (N <= 0 || fields < 6 || C <= 0 || mH <= 0 || mW <= 0)
        return result;

    const int nCoef = std::min(fields - 6, C);   // mask 系数个数（预期 32）
    if (nCoef <= 0)
        return result;
    const int protoPlane = mH * mW;              // 单个原型通道元素数

    // letterbox 参数（与 preprocess 对称）：把 640 空间坐标映射回原图
    const int inputW = m_engine.inputW();
    const int inputH = m_engine.inputH();
    const int w = originalSize.width, h = originalSize.height;
    const float r = std::min((float)inputW / w, (float)inputH / h);
    const int padW = (inputW - (int)(w * r)) / 2;
    const int padH = (inputH - (int)(h * r)) / 2;
    // 原型空间(160) 相对 640 letterbox 的缩放（预期 0.25）
    const float sx = (float)mW / (float)inputW;
    const float sy = (float)mH / (float)inputH;

    for (int i = 0; i < N; ++i) {
        const float *row = detOut + (size_t)i * fields;
        const float conf = row[4];
        if (conf < confThreshold)
            continue;

        // 检测框：640 letterbox 空间 → 原图（去 pad、除缩放、clamp）
        float x1 = (row[0] - padW) / r;
        float y1 = (row[1] - padH) / r;
        float x2 = (row[2] - padW) / r;
        float y2 = (row[3] - padH) / r;
        x1 = std::max(0.0f, std::min(x1, (float)w - 1));
        y1 = std::max(0.0f, std::min(y1, (float)h - 1));
        x2 = std::max(0.0f, std::min(x2, (float)w - 1));
        y2 = std::max(0.0f, std::min(y2, (float)h - 1));
        cv::Rect box(cv::Point((int)x1, (int)y1), cv::Point((int)x2, (int)y2));
        if (box.width <= 0 || box.height <= 0)
            continue;

        SegInstance inst;
        inst.box = box;
        inst.confidence = conf;
        inst.classId = (int)row[5];

        // 原型空间 ROI：把 640 空间框缩放到 160 空间并 clamp 到原型边界
        int px1 = std::max(0, std::min((int)std::floor(row[0] * sx), mW));
        int py1 = std::max(0, std::min((int)std::floor(row[1] * sy), mH));
        int px2 = std::max(0, std::min((int)std::ceil(row[2] * sx), mW));
        int py2 = std::max(0, std::min((int)std::ceil(row[3] * sy), mH));
        const int roiX = px1, roiY = py1, roiW = px2 - px1, roiH = py2 - py1;

        if (roiW > 0 && roiH > 0) {
            // 在 ROI 内组装：maskAcc(y,x) = sigmoid( sum_k coef[k] * proto[k][y][x] )
            const float *coef = row + 6;
            cv::Mat roiMask(roiH, roiW, CV_32F);
            for (int y = 0; y < roiH; ++y) {
                float *dst = roiMask.ptr<float>(y);
                const int rowBase = (roiY + y) * mW + roiX;
                for (int x = 0; x < roiW; ++x) {
                    float sum = 0.0f;
                    for (int k = 0; k < nCoef; ++k)
                        sum += coef[k] * protoOut[(size_t)k * protoPlane + rowBase + x];
                    dst[x] = sigmoidf(sum);
                }
            }
            // 概率图 resize 到原图框大小 → [0,255] → 阈值二值化（box 局部掩膜）
            cv::Mat boxMaskF;
            cv::resize(roiMask, boxMaskF, cv::Size(box.width, box.height));
            cv::Mat boxMaskU8;
            boxMaskF.convertTo(boxMaskU8, CV_8U, 255.0);
            cv::threshold(boxMaskU8, inst.mask, maskThreshold * 255.0, 255, cv::THRESH_BINARY);
        }

        result.push_back(inst);
    }
    return result;
}

TaskResult YoloSeg::run(const cv::Mat &frame)
{
    TaskResult res;
    if (!m_engine.isLoaded() || frame.empty())
        return res;

    cv::Size originalSize = frame.size();

    // 分段计时：预处理/后处理在此，H2D/推理由引擎计
    auto t0 = std::chrono::high_resolution_clock::now();
    cv::Mat blob = preprocess(frame);
    auto t1 = std::chrono::high_resolution_clock::now();

    const float *out0 = m_engine.forward(blob);
    if (!out0)
        return res;
    auto t2 = std::chrono::high_resolution_clock::now();

    // 双输出：按 shape 维度区分检测张量(3 维)与原型张量(4 维)，鲁棒于绑定顺序
    const float *detOut = nullptr, *protoOut = nullptr;
    std::vector<int> detShape, protoShape;
    for (int i = 0; i < m_engine.outputCount(); ++i) {
        std::vector<int> sh = m_engine.outputShape(i);
        if (sh.size() == 3) {
            detOut = m_engine.outputData(i);
            detShape = sh;
        } else if (sh.size() == 4) {
            protoOut = m_engine.outputData(i);
            protoShape = sh;
        }
    }
    if (!detOut || !protoOut)
        return res;

    std::vector<SegInstance> insts =
        postprocess(detOut, protoOut, detShape, protoShape, originalSize);
    auto t3 = std::chrono::high_resolution_clock::now();

    double msPre  = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double msPost = std::chrono::duration<double, std::milli>(t3 - t2).count();
    Logger::instance().info(QString("耗时分解 预处理:%1 H2D:%2 推理:%3 后处理:%4 ms")
                                .arg(msPre, 0, 'f', 1)
                                .arg(m_engine.lastH2DMs(), 0, 'f', 1)
                                .arg(m_engine.lastInferMs(), 0, 'f', 1)
                                .arg(msPost, 0, 'f', 1));

    // inferMs 与 detection 语义一致：预处理+GPU+后处理，不含绘制
    res.inferMs  = std::chrono::duration<double, std::milli>(t3 - t0).count();
    res.rendered = draw(frame, insts);
    res.count    = (int)insts.size();
    res.ok       = true;
    Logger::instance().info(QString("实例分割 检出 %1 个，推理 %2 ms")
                                .arg(res.count).arg(res.inferMs, 0, 'f', 1));
    return res;
}

cv::Mat YoloSeg::draw(const cv::Mat &frame, const std::vector<SegInstance> &insts)
{
    cv::Mat output = frame.clone();
    if (insts.empty())
        return output;

    // 标注尺寸随分辨率自适应（同 detection）：以长边 1000px 为基准 s
    const double s = std::max(1.0, std::max(frame.cols, frame.rows) / 1000.0);
    const int lineW = std::max(2, (int)(2.5 * s));

    // 半透明彩色掩膜：在 overlay 上按 box 局部 mask 上色，再与原图加权融合
    cv::Mat overlay = frame.clone();
    for (const auto &inst : insts) {
        if (inst.mask.empty())
            continue;
        if (inst.mask.size() != cv::Size(inst.box.width, inst.box.height))
            continue;
        cv::Scalar color = classColor(inst.classId);
        cv::Mat roi = overlay(inst.box);
        roi.setTo(color, inst.mask);
    }
    cv::addWeighted(overlay, 0.45, frame, 0.55, 0, output);

    // 检测框 + 标签（不透明，画在融合结果之上）
    for (const auto &inst : insts) {
        cv::Scalar color = classColor(inst.classId);
        cv::rectangle(output, inst.box, color, lineW);

        QString label;
        if (inst.classId >= 0 && inst.classId < 80)
            label = QString("%1 %2%").arg(kCocoClasses[inst.classId]).arg((int)(inst.confidence * 100));
        else
            label = QString("cls_%1 %2%").arg(inst.classId).arg((int)(inst.confidence * 100));

        int baseline = 0;
        double fontScale = 0.9 * s;                  // 字号随分辨率放大
        int thickness = std::max(1, (int)(1.5 * s)); // 字粗
        cv::Size textSize = cv::getTextSize(label.toStdString(), cv::FONT_HERSHEY_SIMPLEX,
                                             fontScale, thickness, &baseline);
        cv::Point textOrg(inst.box.x, inst.box.y - textSize.height - 2);
        cv::rectangle(output, cv::Rect(textOrg.x - 2, textOrg.y - 2,
                                        textSize.width + 4, textSize.height + 6),
                      color, cv::FILLED);
        cv::putText(output, label.toStdString(), cv::Point(textOrg.x, textOrg.y + textSize.height),
                    cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 0, 0), thickness);
    }

    return output;
}
