#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "inferenceworker.h"
#include "taskfactory.h"

#include <QApplication>
#include <QCoreApplication>
#include <QTextCursor>
#include <QFileDialog>
#include <QDir>
#include <QPixmap>
#include <QImage>
#include <QKeyEvent>
#include <QFileInfo>
#include <QTime>
#include <QDateTime>
#include <QSettings>
#include <QStatusBar>
#include <QButtonGroup>
#include <opencv2/imgproc.hpp>

namespace {
// TaskType → 中文任务名（用于日志与任务切换提示）
QString taskTypeName(TaskType t)
{
    switch (t) {
    case TaskType::Detection:   return QStringLiteral("目标检测");
    case TaskType::SemanticSeg: return QStringLiteral("语义分割");
    case TaskType::InstanceSeg: return QStringLiteral("实例分割");
    case TaskType::OCR:         return QStringLiteral("OCR");
    case TaskType::Anomaly:     return QStringLiteral("无监督异常检测");
    }
    return QString();
}
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , m_isDragging(false)
    , m_currentImageIndex(-1)
    , m_isRunning(false)
    , m_inferThread(nullptr)
    , m_worker(nullptr)
{
    ui->setupUi(this);
    setWindowFlags(Qt::FramelessWindowHint | windowFlags());
    setAttribute(Qt::WA_TranslucentBackground, true);

    setStyleSheet(
        "QPushButton {"
        "  background-color: rgb(85, 85, 85);"
        "  color: rgb(220, 220, 220);"
        "  border: 2px solid rgb(100, 100, 100);"
        "  border-radius: 4px;"
        "  padding: 6px 16px;"
        "  min-height: 24px;"
        "  font-family: 'SimSun', '宋体';"
        "  font-weight: bold;"
        "  font-size: 13px;"
        "}"
        "QPushButton:hover {"
        "  background-color: rgb(100, 115, 130);"
        "  color: rgb(255, 255, 255);"
        "  border: 2px solid rgb(120, 140, 160);"
        "}"
        "QPushButton:pressed {"
        "  background-color: rgb(60, 60, 60);"
        "  color: rgb(180, 180, 180);"
        "  border: 2px solid rgb(80, 80, 80);"
        "}"
        "QCheckBox {"
        "  color: rgb(220, 220, 220);"
        "  font-family: 'SimSun', '宋体';"
        "  font-weight: bold;"
        "  font-size: 13px;"
        "  spacing: 8px;"
        "  padding: 4px;"
        "}"
        "QCheckBox::indicator {"
        "  width: 16px;"
        "  height: 16px;"
        "}"
        "QLabel {"
        "  color: rgb(220, 220, 220);"
        "}"
    );

    setupTitleBar();

    QVBoxLayout *rootLayout = new QVBoxLayout();
    rootLayout->setContentsMargins(0, 0, 0, 0);
    rootLayout->setSpacing(0);
    rootLayout->addWidget(m_titleBar);
    rootLayout->addWidget(ui->centralwidget);

    QWidget *rootWidget = new QWidget(this);
    rootWidget->setObjectName("rootWidget");
    rootWidget->setStyleSheet("#rootWidget { background-color: rgb(45, 45, 45); border-top-left-radius: 12px; border-top-right-radius: 12px; }");
    rootWidget->setLayout(rootLayout);
    setCentralWidget(rootWidget);

    // centralwidget 保持直角
    ui->centralwidget->setStyleSheet("#centralwidget { background-color: rgb(45, 45, 45); border-radius: 0px; }");

    adjustSize();
    setFixedSize(sizeHint());

    // 初始化日志系统
    Logger::instance().init(ui->log_widget);
    Logger::instance().info("应用程序启动");

    // 在 show_widget 中添加图片显示标签
    m_imageLabel = new QLabel(ui->show_widget);
    m_imageLabel->setAlignment(Qt::AlignCenter);
    m_imageLabel->setStyleSheet("background-color: rgb(50, 50, 50);");
    QVBoxLayout *showLayout = new QVBoxLayout(ui->show_widget);
    showLayout->setContentsMargins(0, 0, 0, 0);
    showLayout->addWidget(m_imageLabel);

    // 连接选择图片按钮
    connect(ui->btn_selectImages, &QPushButton::clicked, this, &MainWindow::onSelectImages);

    // 任务类型 CheckBox 互斥：同一时刻只能选中一个
    QButtonGroup *taskGroup = new QButtonGroup(this);
    taskGroup->setExclusive(true);
    taskGroup->addButton(ui->detectObject);
    taskGroup->addButton(ui->segment);
    taskGroup->addButton(ui->instanceSegment);
    taskGroup->addButton(ui->OCR);
    taskGroup->addButton(ui->anomalyDetect);

    // 连接任务类型 CheckBox 信号：选中即通过工厂切换当前任务
    connect(ui->detectObject, &QCheckBox::toggled, this, [this](bool checked) {
        if (checked) setupTask(TaskType::Detection);
    });
    connect(ui->segment, &QCheckBox::toggled, this, [this](bool checked) {
        if (checked) setupTask(TaskType::SemanticSeg);
    });
    connect(ui->instanceSegment, &QCheckBox::toggled, this, [this](bool checked) {
        if (checked) setupTask(TaskType::InstanceSeg);
    });
    connect(ui->OCR, &QCheckBox::toggled, this, [this](bool checked) {
        if (checked) setupTask(TaskType::OCR);
    });
    connect(ui->anomalyDetect, &QCheckBox::toggled, this, [this](bool checked) {
        if (checked) setupTask(TaskType::Anomaly);
    });

    // 开始/停止推理按钮
    connect(ui->btn_start, &QPushButton::clicked, this, &MainWindow::onStartStop);

    // 生产者-消费者：消费者（推理）运行在子线程
    m_inferThread = new QThread(this);
    m_worker = new InferenceWorker();
    m_worker->moveToThread(m_inferThread);
    // 推理结果回传到主线程显示
    connect(m_worker, &InferenceWorker::resultReady, this, &MainWindow::onInferenceResult);
    // 推理进行中/空闲时控制任务类型是否可选
    connect(m_worker, &InferenceWorker::busy, this, &MainWindow::onWorkerBusy);
    connect(m_worker, &InferenceWorker::idle, this, &MainWindow::onWorkerIdle);
    // 线程结束后自动清理 worker
    connect(m_inferThread, &QThread::finished, m_worker, &QObject::deleteLater);
    connect(m_worker, &InferenceWorker::finished, m_inferThread, &QThread::quit);
    m_inferThread->start();
    // 触发消费者循环
    QMetaObject::invokeMethod(m_worker, &InferenceWorker::start, Qt::QueuedConnection);

    // 读取上次保存的图片路径
    m_imageDir = QSettings("AIVision", "DetectAnything2").value("imageDir").toString();

    // 状态栏：推理耗时 / FPS / 系统时间
    m_inferLabel = new QLabel(this);
    m_inferLabel->setStyleSheet(
        "color: rgb(120, 200, 120);"
        "font-family: 'Consolas', 'SimSun';"
        "font-size: 12px;"
        "padding: 0 8px;"
    );
    m_fpsLabel = new QLabel(this);
    m_fpsLabel->setStyleSheet(
        "color: rgb(120, 180, 230);"
        "font-family: 'Consolas', 'SimSun';"
        "font-size: 12px;"
        "padding: 0 8px;"
    );
    m_timeLabel = new QLabel(this);
    m_timeLabel->setStyleSheet(
        "color: rgb(220, 220, 220);"
        "font-family: 'Consolas', 'SimSun';"
        "font-size: 12px;"
        "padding: 0 8px;"
    );
    statusBar()->setStyleSheet("QStatusBar { background-color: rgb(40, 40, 40); border-bottom-left-radius: 12px; border-bottom-right-radius: 12px; }");
    statusBar()->addPermanentWidget(m_inferLabel);
    statusBar()->addPermanentWidget(m_fpsLabel);
    statusBar()->addPermanentWidget(m_timeLabel);

    m_timeTimer = new QTimer(this);
    connect(m_timeTimer, &QTimer::timeout, this, &MainWindow::updateTime);
    m_timeTimer->start(1000);
    updateTime();
}

MainWindow::~MainWindow()
{
    if (m_worker)
        m_worker->stop();
    if (m_inferThread) {
        m_inferThread->quit();
        m_inferThread->wait();
    }
    delete ui;
}

void MainWindow::setupTitleBar()
{
    m_titleBar = new QWidget(this);
    m_titleBar->setFixedHeight(36);
    m_titleBar->setObjectName("titleBar");
    m_titleBar->setStyleSheet(
        "#titleBar { background-color: rgb(40, 40, 40); border-bottom: 1px solid rgb(65, 65, 65); border-top-left-radius: 12px; border-top-right-radius: 12px; }"
    );

    QHBoxLayout *layout = new QHBoxLayout(m_titleBar);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    layout->addStretch();
    m_titleLabel = new QLabel("AIVision", m_titleBar);
    m_titleLabel->setAlignment(Qt::AlignCenter);
    m_titleLabel->setStyleSheet("color: rgb(220, 220, 220); font-size: 14px; font-weight: bold;");
    layout->addWidget(m_titleLabel);
    layout->addStretch();

    m_minBtn = new QPushButton("—", m_titleBar);
    m_minBtn->setFixedSize(46, 36);
    m_minBtn->setCursor(Qt::PointingHandCursor);
    m_minBtn->setStyleSheet(
        "QPushButton {"
        "  background-color: transparent;"
        "  color: rgb(200, 200, 200);"
        "  border: none;"
        "  font-size: 14px;"
        "}"
        "QPushButton:hover {"
        "  background-color: rgb(60, 60, 60);"
        "}"
    );
    connect(m_minBtn, &QPushButton::clicked, this, &MainWindow::onMinimize);
    layout->addWidget(m_minBtn);

    m_closeBtn = new QPushButton("✕", m_titleBar);
    m_closeBtn->setFixedSize(46, 36);
    m_closeBtn->setCursor(Qt::PointingHandCursor);
    m_closeBtn->setStyleSheet(
        "QPushButton {"
        "  background-color: transparent;"
        "  color: rgb(200, 200, 200);"
        "  border: none;"
        "  font-size: 12px;"
        "}"
        "QPushButton:hover {"
        "  background-color: rgb(232, 17, 35);"
        "  color: white;"
        "}"
    );
    connect(m_closeBtn, &QPushButton::clicked, this, &MainWindow::onClose);
    layout->addWidget(m_closeBtn);
}

void MainWindow::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton) {
        QWidget *child = childAt(event->pos());
        if (m_titleBar && (child == m_titleBar || m_titleBar->isAncestorOf(child))) {
            m_isDragging = true;
            m_dragPos = event->globalPosition().toPoint() - frameGeometry().topLeft();
            event->accept();
            return;
        }
    }
    QMainWindow::mousePressEvent(event);
}

void MainWindow::mouseMoveEvent(QMouseEvent *event)
{
    if (m_isDragging) {
        move(event->globalPosition().toPoint() - m_dragPos);
        event->accept();
        return;
    }
    QMainWindow::mouseMoveEvent(event);
}

void MainWindow::mouseReleaseEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton) {
        m_isDragging = false;
    }
    QMainWindow::mouseReleaseEvent(event);
}

void MainWindow::onMinimize()
{
    showMinimized();
}

void MainWindow::onClose()
{
    close();
}

void MainWindow::updateTime()
{
    m_timeLabel->setText(QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss"));
}

void MainWindow::onStartStop()
{
    if (!m_isRunning) {
        // 开始推理前校验
        if (!(m_task && m_task->isInitialized())) {
            Logger::instance().warning("推理引擎未初始化，请先选择任务类型");
            return;
        }
        if (m_imageList.isEmpty()) {
            Logger::instance().warning("未加载图片，请先选择图片文件夹");
            return;
        }

        m_isRunning = true;
        ui->widget_3->setEnabled(false);   // 运行期间禁止切换任务类型
        ui->btn_start->setText("停止");
        m_frameTimer.start();               // 启动帧间隔计时
        // 事件驱动：直接入队当前帧，推理完成后自动续推下一帧
        requestInference(m_currentImageIndex);
        Logger::instance().info("推理已开始");
    } else {
        // 停止：停止生产、清空待处理队列
        m_isRunning = false;
        ui->btn_start->setText("开始");
        if (m_worker)
            m_worker->clear();
        // 若消费者已空闲则立即恢复任务类型可选；否则等 idle 信号
        if (m_worker && m_worker->isIdle())
            ui->widget_3->setEnabled(true);
        Logger::instance().info("推理已停止");
    }
}

void MainWindow::onWorkerBusy()
{
    // 推理进行中：禁止切换任务类型
    ui->widget_3->setEnabled(false);
}

void MainWindow::onWorkerIdle()
{
    // 推理空闲：非自动运行时恢复任务类型可选
    if (!m_isRunning)
        ui->widget_3->setEnabled(true);
}

void MainWindow::onTaskTypeChanged(QCheckBox *box, const QString &name)
{
    Q_UNUSED(box);
    Logger::instance().info(QString("已选择任务类型: %1").arg(name));
}

QString MainWindow::resolveEnginePath(const QString &relativePath) const
{
    // 智能查找 engine 文件：依次尝试当前路径、可执行文件目录、项目源码目录
    QString path = relativePath;
    if (!QFileInfo(path).exists()) {
        // 相对于可执行文件目录（build/.../debug/ -> 向上找项目根目录）
        path = QCoreApplication::applicationDirPath() + "/../../../" + relativePath;
        if (!QFileInfo(path).exists()) {
            // 再尝试少一层
            path = QCoreApplication::applicationDirPath() + "/../../" + relativePath;
            if (!QFileInfo(path).exists()) {
                // 都找不到，用原始路径（让 init 输出错误信息）
                path = relativePath;
            }
        }
    }
    return path;
}

void MainWindow::setupTask(TaskType type)
{
    m_taskType = type;

    // 摘除并释放旧任务（仅在 worker 空闲时切换，沿用 busy/idle 门控）
    if (m_worker)
        m_worker->setTask(nullptr);
    if (m_task) {
        m_task->release();
        m_task.reset();
    }

    // 由工厂创建具体任务；未实现的类型返回 nullptr
    std::unique_ptr<ITask> t = TaskFactory::create(type);
    if (!t) {
        Logger::instance().info(QString("%1 任务未实现（占位）").arg(taskTypeName(type)));
        onTaskTypeChanged(nullptr, taskTypeName(type));
        return;
    }

    QString path = resolveEnginePath(TaskFactory::enginePathFor(type));
    if (t->init(path)) {
        m_task = std::move(t);
        if (m_worker)
            m_worker->setTask(m_task);
        Logger::instance().info(QString("%1推理初始化成功: %2").arg(m_task->name()).arg(path));
    } else {
        Logger::instance().error(QString("%1推理初始化失败: %2").arg(t->name()).arg(path));
    }
    onTaskTypeChanged(nullptr, taskTypeName(type));
}

void MainWindow::onSelectImages()
{
    // 默认路径优先使用上次保存的，其次按当前任务类型选择项目自带 images 子目录
    QString defaultDir = m_imageDir;
    if (defaultDir.isEmpty()) {
        const QString base = "E:/work/AIversion/DetectAnything2/images/";
        // TaskType → images 子目录，便于各任务默认打开各自的测试图片
        QString sub;
        switch (m_taskType) {
        case TaskType::Detection:   sub = QStringLiteral("目标检测"); break;
        case TaskType::SemanticSeg: sub = QStringLiteral("语义分割"); break;
        case TaskType::InstanceSeg: sub = QStringLiteral("实例分割"); break;
        case TaskType::OCR:         sub = QStringLiteral("字符检测"); break;
        case TaskType::Anomaly:     sub = QStringLiteral("无监督异常检测"); break;
        }
        defaultDir = base + sub;
        if (!QFileInfo(defaultDir).exists())
            defaultDir = base + QStringLiteral("目标检测");   // 任务专属目录不存在则回退
    }

    QString dir = QFileDialog::getExistingDirectory(
        this, "选择图片文件夹", defaultDir,
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);

    if (dir.isEmpty())
        return;

    // 保存当前选择的路径，下次默认打开它
    m_imageDir = dir;
    QSettings("AIVision", "DetectAnything2").setValue("imageDir", dir);

    // 扫描文件夹中的图片文件
    QStringList filters;
    filters << "*.jpg" << "*.jpeg" << "*.png" << "*.bmp" << "*.tif" << "*.tiff";
    QDir imageDir(dir);
    imageDir.setNameFilters(filters);
    imageDir.setFilter(QDir::Files | QDir::NoDotAndDotDot);
    QStringList files = imageDir.entryList();

    if (files.isEmpty()) {
        Logger::instance().warning("所选文件夹中没有图片文件");
        return;
    }

    // 构建完整路径列表
    m_imageList.clear();
    for (const QString &f : files)
        m_imageList.append(imageDir.absoluteFilePath(f));

    m_currentImageIndex = 0;
    Logger::instance().info(QString("已加载 %1 张图片，来自: %2").arg(m_imageList.size()).arg(dir));

    // 清空旧的待处理任务，请求推理第一张
    if (!(m_task && m_task->isInitialized())) {
        Logger::instance().warning("推理引擎未初始化，请先选择任务类型");
    }
    if (m_worker)
        m_worker->clear();
    requestInference(m_currentImageIndex);
}

// 生产者：将文件路径交给消费者，由 worker 线程自己读取图像
void MainWindow::requestInference(int index)
{
    if (index < 0 || index >= m_imageList.size() || !m_worker)
        return;

    QString path = m_imageList[index];
    Logger::instance().info(QString("推理中 [%1/%2]: %3")
                                .arg(index + 1)
                                .arg(m_imageList.size())
                                .arg(QFileInfo(path).fileName()));

    m_worker->enqueue(path, index);
}

// 消费者回传结果：显示已绘制检测框的图像，并刷新推理耗时/FPS
void MainWindow::onInferenceResult(const cv::Mat &frame, int index, int detectionCount, double inferMs)
{
    Q_UNUSED(index);
    Q_UNUSED(detectionCount);
    displayImage(frame);

    // 刷新状态栏：单帧推理耗时 + 真实帧率
    m_inferLabel->setText(QString("推理: %1 ms").arg(inferMs, 0, 'f', 1));
    double frameMs = m_frameTimer.nsecsElapsed() / 1e6;
    m_frameTimer.restart();
    double fps = frameMs > 0.0 ? 1000.0 / frameMs : 0.0;
    m_fpsLabel->setText(QString("FPS: %1").arg(fps, 0, 'f', 1));

    // 事件驱动续推：仍在运行时，立即生产下一帧
    // 注意：m_isRunning 可能已在点击停止时被置为 false，此时间隙中结果仍可能到达
    //       此时不应再入队下一帧
    if (m_isRunning && !m_imageList.isEmpty()) {
        m_currentImageIndex = (m_currentImageIndex + 1) % m_imageList.size();
        requestInference(m_currentImageIndex);
    }
}

void MainWindow::displayImage(const cv::Mat &frame)
{
    if (frame.empty())
        return;

    // 转换为 QImage 并显示
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    QImage img(rgb.data, rgb.cols, rgb.rows, (int)rgb.step, QImage::Format_RGB888);
    m_imageLabel->setPixmap(QPixmap::fromImage(img).scaled(
        m_imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void MainWindow::keyPressEvent(QKeyEvent *event)
{
    if (m_imageList.isEmpty()) {
        QMainWindow::keyPressEvent(event);
        return;
    }

    if (event->key() == Qt::Key_Right || event->key() == Qt::Key_D) {
        if (m_currentImageIndex < m_imageList.size() - 1) {
            m_currentImageIndex++;
            // 手动切换时丢弃旧的待处理任务
            if (m_worker) m_worker->clear();
            requestInference(m_currentImageIndex);
        }
    } else if (event->key() == Qt::Key_Left || event->key() == Qt::Key_A) {
        if (m_currentImageIndex > 0) {
            m_currentImageIndex--;
            if (m_worker) m_worker->clear();
            requestInference(m_currentImageIndex);
        }
    } else {
        QMainWindow::keyPressEvent(event);
    }
}
