#include "anomalydetect.h"
#include "logger.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <QStringList>

bool AnomalyDetect::init(const QString &enginePath)
{
    if (!m_engine.load(enginePath.toStdString()))
        return false;

    // 打印各输出的名字/形状/dtype，便于核对模型 IO（预期 4 个输出）
    for (int i = 0; i < m_engine.outputCount(); ++i) {
        std::vector<int> sh = m_engine.outputShape(i);
        QStringList parts;
        for (int d : sh)
            parts << QString::number(d);
        nvinfer1::DataType dt = m_engine.outputDataType(i);
        const char *dts = (dt == nvinfer1::DataType::kFLOAT) ? "FLOAT"
                        : (dt == nvinfer1::DataType::kINT32) ? "INT32"
                        : (dt == nvinfer1::DataType::kBOOL)  ? "BOOL"  : "OTHER";
        Logger::instance().info(QString("异常检测 output[%1] name=%2 shape=[%3] dtype=%4")
                                    .arg(i)
                                    .arg(QString::fromStdString(m_engine.outputName(i)))
                                    .arg(parts.join(", "))
                                    .arg(dts));
    }

    resolveOutputs();
    if (m_scoreIdx < 0 || m_mapIdx < 0) {
        Logger::instance().error("异常检测：未在 engine 输出中定位到 pred_score / anomaly_map");
        m_engine.release();
        return false;
    }
    buildJetLut();
    Logger::instance().info(QString("异常检测：pred_score=out[%1] anomaly_map=out[%2]，输入 %3x%4")
                                .arg(m_scoreIdx).arg(m_mapIdx)
                                .arg(m_engine.inputW()).arg(m_engine.inputH()));
    return true;
}

void AnomalyDetect::resolveOutputs()
{
    m_scoreIdx = -1;
    m_mapIdx   = -1;
    int fbScore = -1, fbMap = -1;   // 兜底：按 dtype+shape 猜
    for (int i = 0; i < m_engine.outputCount(); ++i) {
        const std::string nm = m_engine.outputName(i);
        if (nm == "pred_score")  { m_scoreIdx = i; continue; }
        if (nm == "anomaly_map") { m_mapIdx   = i; continue; }

        // 名字对不上时的兜底：float 且元素数=1 → 分数；float 且 4 维 → 热力图
        if (m_engine.outputDataType(i) == nvinfer1::DataType::kFLOAT) {
            if (m_engine.outputSize(i) == 1 && fbScore < 0)
                fbScore = i;
            else if (m_engine.outputShape(i).size() == 4 && fbMap < 0)
                fbMap = i;
        }
    }
    if (m_scoreIdx < 0) m_scoreIdx = fbScore;
    if (m_mapIdx   < 0) m_mapIdx   = fbMap;
}

void AnomalyDetect::buildJetLut()
{
    // anomalib apply_colormap 的 9 个锚点（RGB），原实现用
    // F.interpolate(size=(256,3), mode="bilinear", align_corners=False) 插值到 256 级。
    // 这里用等价的 half-pixel 公式复刻：src = (i+0.5)*9/256 - 0.5。
    static const float anchors[9][3] = {
        {  0,   0, 143},   // dark blue
        {  0,   0, 255},   // blue
        {  0, 127, 255},   // light blue
        {  0, 255, 255},   // cyan
        {127, 255, 127},   // light green
        {255, 255,   0},   // yellow
        {255, 127,   0},   // orange
        {255,   0,   0},   // red
        {127,   0,   0},   // dark red
    };
    const int N = 9, OUT = 256;
    m_jetLut.create(1, OUT, CV_8UC3);
    cv::Vec3b *p = m_jetLut.ptr<cv::Vec3b>(0);
    for (int i = 0; i < OUT; ++i) {
        float src = (i + 0.5f) * (float)N / (float)OUT - 0.5f;
        src = std::max(0.f, std::min((float)(N - 1), src));
        int   i0 = (int)std::floor(src);
        int   i1 = std::min(i0 + 1, N - 1);
        float f  = src - i0;
        // anchors 为 RGB，OpenCV 用 BGR，故通道顺序翻转
        uchar b = cv::saturate_cast<uchar>(anchors[i0][2] * (1 - f) + anchors[i1][2] * f);
        uchar g = cv::saturate_cast<uchar>(anchors[i0][1] * (1 - f) + anchors[i1][1] * f);
        uchar r = cv::saturate_cast<uchar>(anchors[i0][0] * (1 - f) + anchors[i1][0] * f);
        p[i] = cv::Vec3b(b, g, r);
    }
}

void AnomalyDetect::release()
{
    m_engine.release();
    m_jetLut.release();
    m_scoreIdx = -1;
    m_mapIdx   = -1;
}

bool AnomalyDetect::isInitialized() const
{
    return m_engine.isLoaded();
}

QString AnomalyDetect::name() const
{
    return QStringLiteral("无监督异常检测");
}

cv::Mat AnomalyDetect::preprocess(const cv::Mat &frame)
{
    const int inputW = m_engine.inputW();   // 256
    const int inputH = m_engine.inputH();   // 256

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(inputW, inputH), 0, 0, cv::INTER_LINEAR);

    // /255 到[0,1] + BGR→RGB（swapRB=true）。ImageNet mean/std 已在 engine 内，勿再归一化。
    // blobFromImage 计算 scalefactor*(image-mean)，故 scalefactor=1/255、mean=0。
    cv::Mat blob;
    cv::dnn::blobFromImage(resized, blob, 1.0 / 255.0, cv::Size(inputW, inputH),
                           cv::Scalar(0, 0, 0), true, false);
    return blob;
}

cv::Mat AnomalyDetect::colorize(const cv::Mat &mapNorm)
{
    // mapNorm: CV_32F，值域[0,1]（anomalib 已归一化，0.5=阈值）。
    // 复刻 np_to_pil_image + apply_colormap：*255→uint8→jet LUT，不做 min-max 归一化。
    cv::Mat u8;
    // convertTo 带 alpha=255：u8 = saturate_cast<uchar>(mapNorm*255)，自动 clamp 到[0,255]
    mapNorm.convertTo(u8, CV_8UC1, 255.0);
    // 手动逐像素查表上色：绕开 cv::applyColorMap/LUT 对 userColor "total==256" 的断言
    // （OpenCV 4.8.0 的 applyColorMap 内部 cv::LUT 对自定义表形状校验过严会直接 terminate）。
    // m_jetLut 为 1x256 CV_8UC3，ptr(0) 即 256 项 BGR 连续表，按灰度值直接索引，零版本依赖。
    cv::Mat heat(u8.size(), CV_8UC3);
    const cv::Vec3b *lut = m_jetLut.ptr<cv::Vec3b>(0);
    for (int y = 0; y < u8.rows; ++y) {
        const uchar *s = u8.ptr<uchar>(y);
        cv::Vec3b *d = heat.ptr<cv::Vec3b>(y);
        for (int x = 0; x < u8.cols; ++x)
            d[x] = lut[s[x]];
    }
    return heat;
}

cv::Mat AnomalyDetect::draw(const cv::Mat &frame, const cv::Mat &heat,
                            const cv::Mat &mask, double score, int regions)
{
    const int W = frame.cols, H = frame.rows;
    const int gap = 6;      // 面板间隔
    const int bar = 46;     // 顶部标题栏高度

    // 叠加图 = 0.5*原图 + 0.5*热力图（复刻 anomalib overlay_image alpha=0.5）
    cv::Mat overlay;
    cv::addWeighted(frame, 0.5, heat, 0.5, 0, overlay);

    // 在叠加图上用红描边标出缺陷区域（mask = anomaly_map>0.5）
    if (!mask.empty() && regions > 0) {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(overlay, contours, -1, cv::Scalar(0, 0, 255), 2);
    }

    // 画布：3 面板横向拼接 + 顶部标题栏
    const int canvasW = W * 3 + gap * 4;
    const int canvasH = H + bar;
    cv::Mat canvas(canvasH, canvasW, CV_8UC3, cv::Scalar(255, 255, 255));

    const int xs[3] = { gap, gap * 2 + W, gap * 3 + W * 2 };
    const cv::Mat panels[3] = { frame, heat, overlay };
    for (int k = 0; k < 3; ++k) {
        cv::Mat roi = canvas(cv::Rect(xs[k], bar, W, H));
        panels[k].copyTo(roi);
    }

    // 判定：score>0.5 → NG（红），否则 OK（绿）
    const bool ng = score > 0.5;
    const cv::Scalar col = ng ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 150, 0);

    // 顶部横幅：判定 + 分数 + 缺陷区域数
    char banner[160];
    std::snprintf(banner, sizeof(banner), "%s    score = %.3f  (thr 0.5)    defect regions = %d",
                  ng ? "NG" : "OK", score, regions);
    cv::putText(canvas, banner, cv::Point(gap, 31), cv::FONT_HERSHEY_SIMPLEX,
                0.75, col, 2, cv::LINE_AA);

    // 各面板左上角小标题（白字黑边，保证任意底色可读）
    const char *titles[3] = { "Input", "Heatmap", "Overlay" };
    for (int k = 0; k < 3; ++k) {
        cv::Point org(xs[k] + 10, bar + 28);
        cv::putText(canvas, titles[k], org, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(canvas, titles[k], org, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
    return canvas;
}

TaskResult AnomalyDetect::run(const cv::Mat &frame)
{
    TaskResult res;
    if (!m_engine.isLoaded() || frame.empty())
        return res;

    const cv::Size orig = frame.size();

    auto t0 = std::chrono::high_resolution_clock::now();
    cv::Mat blob = preprocess(frame);
    auto t1 = std::chrono::high_resolution_clock::now();

    if (!m_engine.forward(blob))
        return res;
    auto t2 = std::chrono::high_resolution_clock::now();

    // 图像级分数（已归一化[0,1]，阈值 0.5）
    const float *scorePtr = m_engine.outputData(m_scoreIdx);
    if (!scorePtr) {
        Logger::instance().error("异常检测：pred_score 输出为空");
        return res;
    }
    double score = (double)scorePtr[0];

    // 像素级热力图 [1,1,mh,mw] → CV_32F(mh,mw)，并 clamp 到[0,1]
    std::vector<int> msh = m_engine.outputShape(m_mapIdx);
    const float *mp = m_engine.outputData(m_mapIdx);
    if (msh.size() < 2 || !mp) {
        Logger::instance().error("异常检测：anomaly_map 输出形状不支持或为空");
        return res;
    }
    const int mh = msh[msh.size() - 2];
    const int mw = msh[msh.size() - 1];
    cv::Mat mapNorm(mh, mw, CV_32F);
    for (int y = 0; y < mh; ++y) {
        float *dst = mapNorm.ptr<float>(y);
        for (int x = 0; x < mw; ++x) {
            float v = mp[(size_t)y * mw + x];
            dst[x] = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
        }
    }

    // 热力图上采样回原图（float 双线性更平滑），再套 jet LUT
    cv::Mat mapOrig;
    cv::resize(mapNorm, mapOrig, orig, 0, 0, cv::INTER_LINEAR);
    cv::Mat heat = colorize(mapOrig);

    // 缺陷区域：模型分辨率下 >0.5 的连通域个数；原图分辨率 mask 供绘制描边
    cv::Mat maskModel = (mapNorm > 0.5f);          // CV_8U 0/255
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(maskModel, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    int regions = (int)contours.size();
    cv::Mat maskOrig = (mapOrig > 0.5f);           // 原图分辨率

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
    res.rendered = draw(frame, heat, maskOrig, score, regions);
    res.count    = regions;
    res.summary  = QString("%1  score=%2")
                       .arg(score > 0.5 ? "NG" : "OK")
                       .arg(score, 0, 'f', 3);
    res.ok       = true;
    Logger::instance().info(QString("异常检测 %1，score=%2，缺陷区域 %3 个，推理 %4 ms")
                                .arg(score > 0.5 ? "NG" : "OK")
                                .arg(score, 0, 'f', 3)
                                .arg(regions)
                                .arg(res.inferMs, 0, 'f', 1));
    return res;
}
