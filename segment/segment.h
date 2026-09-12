#ifndef SEMANTICSEG_H
#define SEMANTICSEG_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "itask.h"
#include "trtengine.h"

// 语义分割任务：组合通用 TrtEngine，只实现本模型专属的预处理/后处理/绘制。
//
// 模型由 PaddleSeg 导出并简化（model_sim.onnx），图内已做 argmax，IO 约定：
//   输入 images  FLOAT [1,3,960,960]：BGR2RGB → resize(960,960) → (x/255-0.5)/0.5
//                                       （即 (x-127.5)/127.5，直接 resize，不做 letterbox）
//   输出 output0 INT32  [1,960,960]  ：每像素的类别索引（已 argmax，非 logits）
//
// 注意：输出是 INT32 而非 float，必须用 TrtEngine::outputRaw + outputDataType 读取，
//       不能用 outputData(float*)，否则整型位模式被当 float 解释会得到全 0 类别图。
//
// 后处理：把类别索引图用最近邻上采样回原图尺寸（保持类别不插值），按调色板着色
//         并与原图半透明融合；背景类（id=0）保持原图不上色。
class SemanticSeg : public ITask
{
public:
    bool init(const QString &enginePath) override;
    TaskResult run(const cv::Mat &frame) override;
    void release() override;
    bool isInitialized() const override;
    QString name() const override;

private:
    TrtEngine m_engine;

    // 直接 resize + (x-127.5)/127.5 + BGR2RGB，输出 NCHW float blob
    cv::Mat preprocess(const cv::Mat &frame);

    // 把引擎输出解析为「模型分辨率下的类别索引图」CV_32S(H x W)。
    // raw 按 dtype 解释（本模型 INT32；亦兼容 float 索引图）；shape 支持 rank-3
    // [1,H,W] 与 rank-4 [1,1,H,W]。outMaxClass 回传出现的最大类别 id。
    // 返回空 Mat 表示输出为空或形状不支持。
    cv::Mat decodeClassMap(const void *raw, nvinfer1::DataType dtype,
                           const std::vector<int> &shape, int &outMaxClass);

    // 按调色板着色 + 与原图半透明融合（背景类 id=0 不上色）
    cv::Mat draw(const cv::Mat &frame, const cv::Mat &classMapOrigSize, int maxClass);
};

#endif // SEMANTICSEG_H
