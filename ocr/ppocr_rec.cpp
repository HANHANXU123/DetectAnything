#include "ppocr_rec.h"
#include "ppocr_utils.h"   // self_resize
#include "logger.h"

#include <fstream>
#include <algorithm>

bool TextRecognizer::readDict(const std::string &dictPath)
{
    std::ifstream in(dictPath);
    if (!in.is_open())
        return false;

    m_labelList.clear();
    m_labelList.push_back("#");   // blank，索引 0（CTC 解码跳过）
    std::string line;
    while (std::getline(in, line))
        m_labelList.push_back(line);
    m_labelList.push_back(" ");   // 末位补空格
    // 有效字符数须 > 0（即除 "#" 与 " " 外至少 1 个字典项）
    return m_labelList.size() > 2;
}

bool TextRecognizer::init(const std::string &enginePath, const std::string &dictPath)
{
    if (!readDict(dictPath)) {
        Logger::instance().error(QString("无法读取 OCR 字典: %1").arg(dictPath.c_str()));
        return false;
    }
    return m_engine.load(enginePath);
}

void TextRecognizer::release()
{
    m_engine.release();
}

bool TextRecognizer::isInitialized() const
{
    return m_engine.isLoaded();
}

cv::Mat TextRecognizer::preprocess(const cv::Mat &crop)
{
    const int inputW = m_engine.inputW();   // 640
    const int inputH = m_engine.inputH();   // 48

    // 等比缩放到高 inputH、letterbox 填充到 inputH x inputW（同原 demo self_resize）
    cv::Mat resized = self_resize(crop, inputW, inputH);
    if (!resized.isContinuous())
        resized = resized.clone();

    const size_t area = (size_t)inputH * inputW;
    cv::Mat blob(1, (int)(area * 3), CV_32F);
    float *base = blob.ptr<float>(0);

    // 沿用原 demo 归一化 (x/255-0.5)/0.5 与通道写入顺序：
    //   ch0 <- pixel[2]，ch1 <- pixel[1]，ch2 <- pixel[0]
    const float mean = 0.5f, stdv = 0.5f;
    float *ch0 = base + area * 0;   // demo: phostB
    float *ch1 = base + area * 1;   // demo: phostG
    float *ch2 = base + area * 2;   // demo: phostR

    const unsigned char *p = resized.data;
    for (size_t i = 0; i < area; ++i, p += 3) {
        *ch2++ = (p[0] / 255.0f - mean) / stdv;
        *ch1++ = (p[1] / 255.0f - mean) / stdv;
        *ch0++ = (p[2] / 255.0f - mean) / stdv;
    }
    return blob;
}

std::string TextRecognizer::ctcDecode(const float *out, int maxChars, int vocab) const
{
    std::string s;
    int lastIdx = -1;   // 上一时间步 argmax，用于 CTC 合并连续重复字符
    for (int i = 0; i < maxChars; ++i) {
        const float *begin = out + (size_t)i * vocab;
        const float *end   = begin + vocab;
        int idx = (int)(std::max_element(begin, end) - begin);
        // 跳过 blank(索引 0)、合并连续重复，并做字典越界保护
        if (idx != 0 && idx != lastIdx && idx < (int)m_labelList.size())
            s += m_labelList[idx];
        lastIdx = idx;
    }
    return s;
}

std::string TextRecognizer::recognize(const cv::Mat &crop)
{
    if (!m_engine.isLoaded() || crop.empty())
        return std::string();

    cv::Mat blob = preprocess(crop);
    const float *out = m_engine.forward(blob);
    if (!out)
        return std::string();

    std::vector<int> shape = m_engine.outputShape();   // {1, maxChars, vocab}
    int maxChars = (shape.size() >= 2) ? shape[1] : 0;
    int vocab    = (shape.size() >= 3) ? shape[2] : 0;
    return ctcDecode(out, maxChars, vocab);
}
