#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QCheckBox>
#include <QMouseEvent>
#include <QHBoxLayout>
#include <QWidget>
#include <QStringList>
#include <QTimer>
#include <QElapsedTimer>
#include <QThread>
#include <memory>
#include "logger.h"
#include "itask.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class InferenceWorker;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

protected:
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;

private slots:
    void onMinimize();
    void onClose();
    void onTaskTypeChanged(QCheckBox *box, const QString &name);
    void onSelectImages();
    void updateTime();
    void onStartStop();
    void onInferenceResult(const cv::Mat &frame, int index, int detectionCount, double inferMs);
    void onWorkerBusy();
    void onWorkerIdle();

private:
    void setupTitleBar();
    void displayImage(const cv::Mat &frame);
    void requestInference(int index);

    Ui::MainWindow *ui;

    QWidget *m_titleBar;
    QLabel *m_titleLabel;
    QPushButton *m_minBtn;
    QPushButton *m_closeBtn;

    QLabel *m_imageLabel;
    QLabel *m_timeLabel;
    QLabel *m_inferLabel;
    QLabel *m_fpsLabel;
    QTimer *m_timeTimer;

    QElapsedTimer m_frameTimer;  // 真实帧间隔计时

    bool m_isRunning;

    QThread *m_inferThread;
    InferenceWorker *m_worker;

    bool m_isDragging;
    QPoint m_dragPos;

    QStringList m_imageList;
    QString m_imageDir;
    int m_currentImageIndex;

    std::shared_ptr<ITask> m_task;              // 当前任务（工厂创建，注入 worker）
    TaskType m_taskType = TaskType::Detection;

    void setupTask(TaskType type);
    QString resolveEnginePath(const QString &relativePath) const;
};
#endif // MAINWINDOW_H
