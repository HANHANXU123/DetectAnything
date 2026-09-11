#include "ocr.h"
#include "ppocr_utils.h"   // get_rotate_crop_image
#include "logger.h"

#include <QImage>
#include <QPainter>
#include <QPainterPath>
#include <QPolygonF>
#include <QFont>
#include <QColor>
#include <QPen>

#include <chrono>
#include <algorithm>
#include <cstring>
#include <climits>

bool OCR::init(const QString &baseDir)
{
    // baseDir 为资源目录（如 .../models/ocr），内部拼接 det/rec/字典文件名
    QString dir = baseDir;
    while (dir.endsWith('/') || dir.endsWith('\\'))
        dir.chop(1);

    const QString detPath  = dir + "/det.trtmodel";
    const QString recPath  = dir + "/rec.trtmodel";
    const QString dictPath = dir + "/ppocrv6_dict.txt";

    if (!m_det.init(detPath.toStdString())) {
        Logger::instance().error(QString("OCR 文本检测初始化失败: %1").arg(detPath));
        return false;
    }
    if (!m_rec.init(recPath.toStdString(), dictPath.toStdString())) {
        Logger::instance().error(QString("OCR 文本识别初始化失败: %1").arg(recPath));
        m_det.release();   // 回滚已加载的检测引擎
        return false;
    }

    m_initialized = true;
    return true;
}

void OCR::release()
{
    m_det.release();
    m_rec.release();
    m_initialized = false;
}

bool OCR::isInitialized() const
{
    return m_initialized;
}

QString OCR::name() const
{
    return QStringLiteral("OCR");
}

TaskResult OCR::run(const cv::Mat &frame)
{
    TaskResult res;
    if (!m_initialized || frame.empty())
        return res;

    auto t0 = std::chrono::high_resolution_clock::now();

    // 1) 文本检测（框坐标已映射回原图坐标系）
    BoxArray boxes = m_det.detect(frame);

    // 2) 逐框裁剪 + 识别
    std::vector<std::string> texts;
    texts.reserve(boxes.size());
    for (const auto &box : boxes) {
        cv::Mat crop = get_rotate_crop_image(frame, box);
        if (crop.empty()) {
            texts.push_back(std::string());
            continue;
        }
        texts.push_back(m_rec.recognize(crop));
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    // inferMs = 检测 + 所有识别总耗时（不含绘制），与 YoloDetector 语义一致
    res.inferMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // summary：拼接识别文本（过长截断）；中文可正常显示在 UI 摘要/日志
    QString summary;
    for (const auto &t : texts) {
        if (t.empty())
            continue;
        if (!summary.isEmpty())
            summary += QStringLiteral(" | ");
        summary += QString::fromStdString(t);
    }
    const int kMaxSummary = 200;
    if (summary.length() > kMaxSummary)
        summary = summary.left(kMaxSummary) + QStringLiteral("...");

    res.rendered = draw(frame, boxes, texts);
    res.count    = (int)boxes.size();
    res.summary  = summary;
    res.ok       = true;

    Logger::instance().info(QString("OCR 检出 %1 个文本框，推理 %2 ms")
                                .arg(boxes.size()).arg(res.inferMs, 0, 'f', 1));
    if (!summary.isEmpty())
        Logger::instance().info(QString("OCR 识别文本: %1").arg(summary));
    return res;
}

cv::Mat OCR::draw(const cv::Mat &frame, const BoxArray &boxes,
                  const std::vector<std::string> &texts)
{
    // 用 QImage + QPainter 绘制：cv::putText 无法渲染中文，QPainter 配合微软雅黑
    // 可正确显示中英文。按需求只画「线框 + 该框识别文本」，不加序号等多余元素。
    // run 在 worker 子线程执行，QPainter 绘制到 QImage 在非 GUI 线程安全（不可用
    // QPixmap，它依赖 GUI 线程）。

    // 1) cv::Mat(BGR) -> QImage：转为 ARGB32_Premultiplied 作画布（QPainter 最佳格式，
    //    convertToFormat 会深拷贝，脱离 rgb 内存独立持有）
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    QImage wrapped(rgb.data, rgb.cols, rgb.rows, (int)rgb.step, QImage::Format_RGB888);
    QImage canvas = wrapped.convertToFormat(QImage::Format_ARGB32_Premultiplied);

    // 2) 标注尺寸随分辨率自适应：以长边 1000px 为基准 s，避免大图上标注过小
    const double s       = std::max(1.0, std::max(frame.cols, frame.rows) / 1000.0);
    const int    lineW   = std::max(2, (int)(2.5 * s));   // 线框宽
    const int    fontSize = std::max(14, (int)(18 * s));  // 字号（像素）
    const int    strokeW  = std::max(2, (int)(3 * s));    // 文字描边宽

    QFont font(QStringLiteral("Microsoft YaHei"));
    font.setPixelSize(fontSize);
    font.setBold(true);

    QPainter painter(&canvas);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setRenderHint(QPainter::TextAntialiasing, true);

    for (size_t i = 0; i < boxes.size(); ++i) {
        const auto &box = boxes[i];
        if (box.size() < 4)
            continue;

        // 四边形顶点，同时记录最上-最左角作为文本锚点
        QPolygonF poly;
        int minX = INT_MAX, minY = INT_MAX;
        for (int k = 0; k < 4; ++k) {
            const int x = box[k][0], y = box[k][1];
            poly << QPointF(x, y);
            minX = std::min(minX, x);
            minY = std::min(minY, y);
        }

        // 颜色按序号轮换（RGB），便于区分相邻文本框
        QColor color(128, 255, 0);
        if (i % 3 == 1)      color = QColor(0, 128, 255);
        else if (i % 3 == 2) color = QColor(255, 128, 0);

        // 线框
        painter.setPen(QPen(color, lineW, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
        painter.setBrush(Qt::NoBrush);
        painter.drawPolygon(poly);

        // 该框识别文本：黑边 + 绿色填充描边，保证任意背景下清晰
        if (i < texts.size() && !texts[i].empty()) {
            const QString text = QString::fromStdString(texts[i]);
            // 基线默认放在框顶上方；顶部空间不足时改放到框内顶部下方
            qreal tx = minX;
            qreal ty = minY - lineW - 2 * s;
            if (ty < fontSize)
                ty = minY + fontSize + lineW;

            QPainterPath path;
            path.addText(tx, ty, font, text);
            // 分两步：先描黑边(轮廓)、再填绿色。单次 drawPath 是“先填充后描边”，
            // 粗黑边会向内盖住绿色笔画，看起来就成了黑字；两步绘制让绿色在最上层。
            painter.setBrush(Qt::NoBrush);
            painter.setPen(QPen(Qt::black, strokeW, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
            painter.drawPath(path);
            painter.setPen(Qt::NoPen);
            painter.setBrush(QColor(0, 255, 0));   // 文字统一绿色
            painter.drawPath(path);
        }
    }
    painter.end();

    // 3) QImage(ARGB32) -> RGB888 -> cv::Mat(BGR)：逐行拷贝（处理 QImage 行对齐）后转回 BGR
    QImage out = canvas.convertToFormat(QImage::Format_RGB888);
    cv::Mat result(out.height(), out.width(), CV_8UC3);
    for (int y = 0; y < out.height(); ++y)
        std::memcpy(result.ptr<uchar>(y), out.constScanLine(y), (size_t)out.width() * 3);
    cv::cvtColor(result, result, cv::COLOR_RGB2BGR);
    return result;
}
