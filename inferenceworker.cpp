#include "inferenceworker.h"
#include "logger.h"
#include <QFile>
#include <QByteArray>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

InferenceWorker::InferenceWorker(QObject *parent)
    : QObject(parent)
    , m_running(false)
    , m_busy(false)
{
}

InferenceWorker::~InferenceWorker()
{
    stop();
}

void InferenceWorker::enqueue(const QString &filePath, int index)
{
    {
        QMutexLocker locker(&m_mutex);
        if (m_queue.size() >= 8)
            m_queue.pop();
        m_queue.push({filePath, index});
    }
    m_cond.wakeOne();
}

void InferenceWorker::clear()
{
    QMutexLocker locker(&m_mutex);
    std::queue<Task> empty;
    m_queue.swap(empty);
}

bool InferenceWorker::isIdle() const
{
    QMutexLocker locker(&m_mutex);
    return m_queue.empty() && !m_busy;
}

void InferenceWorker::setTask(std::shared_ptr<ITask> task)
{
    QMutexLocker locker(&m_mutex);
    m_task = std::move(task);
}

void InferenceWorker::start()
{
    m_running = true;
    while (m_running) {
        Task task;
        {
            QMutexLocker locker(&m_mutex);
            while (m_queue.empty() && m_running)
                m_cond.wait(&m_mutex);
            if (!m_running)
                break;
            task = m_queue.front();
            m_queue.pop();
            m_busy = true;
        }
        emit busy();

        // 子线程读图（避免主线程卡顿）
        // 用 QFile 读字节 + cv::imdecode 解码：cv::imread 内部走 fopen，在 Windows
        // 下按本地 ANSI 码页解析路径，遇到中文（非 ASCII）路径会失败；QFile 用宽字符
        // API 读取、imdecode 在内存中解码，彻底规避该问题且不依赖系统区域设置。
        cv::Mat frame;
        {
            QFile file(task.filePath);
            if (file.open(QIODevice::ReadOnly)) {
                QByteArray data = file.readAll();
                file.close();
                if (!data.isEmpty()) {
                    std::vector<uchar> buf(data.begin(), data.end());
                    frame = cv::imdecode(buf, cv::IMREAD_COLOR);
                }
            }
        }
        if (frame.empty()) {
            Logger::instance().error(QString("无法读取图片: %1").arg(task.filePath));
            bool becameIdle = false;
            {
                QMutexLocker locker(&m_mutex);
                if (m_queue.empty()) {
                    m_busy = false;
                    becameIdle = true;
                }
            }
            if (becameIdle)
                emit idle();
            continue;
        }

        // 取当前任务（主线程注入）：任务内部完成 预处理→推理→后处理→绘制
        std::shared_ptr<ITask> activeTask;
        {
            QMutexLocker locker(&m_mutex);
            activeTask = m_task;
        }
        if (activeTask && activeTask->isInitialized()) {
            TaskResult r = activeTask->run(frame);
            Logger::instance().info(QString("%1 [%2] 检出 %3 个，耗时 %4 ms")
                                        .arg(activeTask->name()).arg(task.index + 1)
                                        .arg(r.count).arg(r.inferMs, 0, 'f', 1));
            // 回传结果到主线程显示（rendered 已是绘制好的图）
            emit resultReady(r.rendered, task.index, r.count, r.inferMs);
        } else {
            // 无可用任务：仍回传原图以便预览（与改造前一致）
            emit resultReady(frame, task.index, 0, 0.0);
        }

        // 处理完一帧，若队列已空则通知主线程空闲
        bool becameIdle = false;
        {
            QMutexLocker locker(&m_mutex);
            if (m_queue.empty()) {
                m_busy = false;
                becameIdle = true;
            }
        }
        if (becameIdle)
            emit idle();
    }
    emit finished();
}

void InferenceWorker::stop()
{
    {
        QMutexLocker locker(&m_mutex);
        m_running = false;
    }
    m_cond.wakeAll();
}
