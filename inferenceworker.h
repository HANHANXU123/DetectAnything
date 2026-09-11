#ifndef INFERENCEWORKER_H
#define INFERENCEWORKER_H

#include <QObject>
#include <QMutex>
#include <QWaitCondition>
#include <queue>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#include "itask.h"

// 消费者：在子线程中读取图像，交给当前任务(ITask)推理并绘制，再通过信号回传
class InferenceWorker : public QObject
{
    Q_OBJECT
public:
    explicit InferenceWorker(QObject *parent = nullptr);
    ~InferenceWorker() override;

    // 生产者（主线程）调用：将文件路径入队，由 worker 自己读取图像
    void enqueue(const QString &filePath, int index);
    // 清空待处理队列
    void clear();
    // 消费者是否空闲（队列空且未在推理）
    bool isIdle() const;

    // 由主线程注入当前任务（受 m_mutex 保护）；传 nullptr 摘除
    void setTask(std::shared_ptr<ITask> task);

public slots:
    void start();
    void stop();

signals:
    // 推理+绘制完成后发出，由主线程接收显示；inferMs 为推理耗时
    void resultReady(const cv::Mat &frame, int index, int detectionCount, double inferMs);
    // 开始推理一帧 / 一帧处理完且队列空
    void busy();
    void idle();
    void finished();

private:
    struct Task {
        QString filePath;
        int index;
    };

    std::queue<Task> m_queue;
    mutable QMutex m_mutex;
    QWaitCondition m_cond;
    bool m_running;
    bool m_busy;

    std::shared_ptr<ITask> m_task;   // 当前任务（主线程注入，worker 线程使用）
};

#endif // INFERENCEWORKER_H
