#ifndef LOGGER_H
#define LOGGER_H

#include <QObject>
#include <QPlainTextEdit>
#include <QFile>
#include <QTextStream>
#include <QDateTime>
#include <QMutex>

class Logger : public QObject
{
    Q_OBJECT

public:
    enum Level {
        INFO,
        WARNING,
        ERROR
    };

    static Logger &instance();
    void init(QPlainTextEdit *logWidget, const QString &filePath = "log.txt");

    // 线程安全日志入口：可在任意线程调用
    void log(Level level, const QString &message);
    void info(const QString &message);
    void warning(const QString &message);
    void error(const QString &message);

signals:
    // 跨线程信号：由主线程接收并更新 UI
    void messageLogged(int level, const QString &timestamp, const QString &levelStr, const QString &message);

private slots:
    // 主线程槽：真正操作 QPlainTextEdit
    void onMessageLogged(int level, const QString &timestamp, const QString &levelStr, const QString &message);

private:
    explicit Logger(QObject *parent = nullptr);
    ~Logger();

    QPlainTextEdit *m_logWidget;
    QFile *m_logFile;
    QTextStream *m_logStream;
    QMutex m_mutex;

    QString levelToString(Level level);
};

#endif // LOGGER_H
