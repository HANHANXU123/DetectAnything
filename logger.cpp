#include "logger.h"

Logger &Logger::instance()
{
    static Logger instance;
    return instance;
}

Logger::Logger(QObject *parent)
    : QObject(parent)
    , m_logWidget(nullptr)
    , m_logFile(nullptr)
    , m_logStream(nullptr)
{
}

Logger::~Logger()
{
    if (m_logStream) {
        m_logStream->flush();
        delete m_logStream;
    }
    if (m_logFile) {
        if (m_logFile->isOpen())
            m_logFile->close();
        delete m_logFile;
    }
}

void Logger::init(QPlainTextEdit *logWidget, const QString &filePath)
{
    m_logWidget = logWidget;
    // 跨线程信号→主线程槽，确保 UI 操作在主线程
    connect(this, &Logger::messageLogged, this, &Logger::onMessageLogged, Qt::QueuedConnection);

    m_logFile = new QFile(filePath, this);
    if (m_logFile->open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text)) {
        m_logStream = new QTextStream(m_logFile);
    }
}

void Logger::log(Level level, const QString &message)
{
    QString timestamp = QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss");
    QString levelStr = levelToString(level);

    // 写日志文件（加锁保护多线程写入）
    {
        QMutexLocker locker(&m_mutex);
        if (m_logStream) {
            *m_logStream << QString("[%1] [%2] %3\n").arg(timestamp, levelStr, message);
            m_logStream->flush();
        }
    }

    // 通过信号投递到主线程更新 UI
    emit messageLogged((int)level, timestamp, levelStr, message);
}

void Logger::info(const QString &message)
{
    log(INFO, message);
}

void Logger::warning(const QString &message)
{
    log(WARNING, message);
}

void Logger::error(const QString &message)
{
    log(ERROR, message);
}

// 主线程槽：真正操作 QPlainTextEdit
void Logger::onMessageLogged(int level, const QString &timestamp, const QString &levelStr, const QString &message)
{
    if (!m_logWidget)
        return;

    // 等级标签着色
    QString color;
    switch ((Level)level) {
    case INFO:    color = "green"; break;
    case WARNING: color = "orange"; break;
    case ERROR:   color = "red"; break;
    default:      color = "black"; break;
    }

    QString htmlLine = QString("[%1] <span style=\"color:%2;\">[%3]</span> %4")
                           .arg(timestamp.toHtmlEscaped(),
                                color,
                                levelStr,
                                message.toHtmlEscaped());
    m_logWidget->appendHtml(htmlLine);
    QTextCursor cursor = m_logWidget->textCursor();
    cursor.movePosition(QTextCursor::End);
    m_logWidget->setTextCursor(cursor);
}

QString Logger::levelToString(Level level)
{
    switch (level) {
    case INFO:    return "INFO";
    case WARNING: return "WARNING";
    case ERROR:   return "ERROR";
    }
    return "UNKNOWN";
}
