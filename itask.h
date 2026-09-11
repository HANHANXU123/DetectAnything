#ifndef ITASK_H
#define ITASK_H

#include <opencv2/opencv.hpp>
#include <QString>

// 任务类型：与 UI 任务类型栏一一对应
enum class TaskType { Detection, SemanticSeg, InstanceSeg, OCR, Anomaly };

// 统一推理结果：rendered 为任务自绘好的图，worker/UI 无需了解具体任务
struct TaskResult
{
    cv::Mat rendered;   // 已绘制框/掩膜/文字/热力图的输出图，直接显示
    int     count = 0;  // 检出数量（目标数/字符数/异常区域数）
    QString summary;    // 预留：文字摘要（OCR 文本、异常分数等）
    double  inferMs = 0; // 单帧推理耗时（预处理+GPU+后处理，不含绘制）
    bool    ok = false;
};

// 任务抽象接口：每个具体任务（检测/分割/OCR/异常检测）实现之
class ITask
{
public:
    virtual ~ITask() = default;
    virtual bool init(const QString &enginePath) = 0;
    virtual TaskResult run(const cv::Mat &frame) = 0;  // 预处理→推理→后处理→绘制
    virtual void release() = 0;
    virtual bool isInitialized() const = 0;
    virtual QString name() const = 0;
};

#endif // ITASK_H
