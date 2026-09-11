#include "mainwindow.h"

#include <QApplication>
#include <QLocale>
#include <QTranslator>
#include <QMetaType>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    // 抑制 OpenCV INFO 级别日志（如并行后端 DLL 加载失败等噪音）
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

    // 注册 cv::Mat 元类型，用于跨线程信号槽（队列连接）传递
    qRegisterMetaType<cv::Mat>("cv::Mat");

    QTranslator translator;
    const QStringList uiLanguages = QLocale::system().uiLanguages();
    for (const QString &locale : uiLanguages) {
        const QString baseName = "DetectAnything_" + QLocale(locale).name();
        if (translator.load(":/i18n/" + baseName)) {
            a.installTranslator(&translator);
            break;
        }
    }
    MainWindow w;
    w.show();
    return QApplication::exec();
}
