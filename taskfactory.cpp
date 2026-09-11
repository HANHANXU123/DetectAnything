#include "taskfactory.h"
#include "yolo.h"   // YoloDetector
#include "ocr.h"    // OCR（位于 ocr/ 子目录，INCLUDEPATH 已含）
#include "yoloseg.h" // YoloSeg（位于 yolo-seg/ 子目录，INCLUDEPATH 已含）

std::unique_ptr<ITask> TaskFactory::create(TaskType type)
{
    switch (type) {
    case TaskType::Detection:
        return std::make_unique<YoloDetector>();
    case TaskType::OCR:
        return std::make_unique<OCR>();
    case TaskType::InstanceSeg:
        return std::make_unique<YoloSeg>();
    // 以下任务尚未实现，返回 nullptr，由 UI 侧记「未实现」占位
    case TaskType::SemanticSeg:
    case TaskType::Anomaly:
        return nullptr;
    }
    return nullptr;
}

QString TaskFactory::enginePathFor(TaskType type)
{
    switch (type) {
    case TaskType::Detection:   return QStringLiteral("models/yolo/yolo26m.engine");
    case TaskType::SemanticSeg: return QStringLiteral("models/segment/");
    case TaskType::InstanceSeg: return QStringLiteral("models/yolo-seg/yolo26m-seg.engine");
    case TaskType::OCR:         return QStringLiteral("models/ocr/");
    case TaskType::Anomaly:     return QStringLiteral("models/anomalyDetect/");
    }
    return QString();
}
