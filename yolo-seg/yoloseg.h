#ifndef YOLOSEG_H
#define YOLOSEG_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "itask.h"
#include "trtengine.h"

// 单个分割实例（YOLO-seg 输出解析结果）
struct SegInstance
{
    cv::Rect box;              // 检测框（原图坐标）
    float confidence = 0.f;    // 置信度 [0,1]
    int classId = 0;           // 类别 id（COCO 0..79）
    cv::Mat mask;              // box 局部二值掩膜 CV_8U(0/255)，尺寸同 box
};

// 实例分割任务：组合通用 TrtEngine，只实现 YOLO-seg 专属的预处理/后处理/绘制。
// 模型为 end2end（内置 NMS），双输出：
//   output0 [1,300,38]：300 个检测 x [x1,y1,x2,y2, conf, class, 32 个 mask 系数]
//   output1 [1,32,160,160]：mask 原型（prototype）
// 输入 640x640 letterbox（填充 114），与 YoloDetector 一致。
class YoloSeg : public ITask
{
public:
    bool init(const QString &enginePath) override;
    TaskResult run(const cv::Mat &frame) override;
    void release() override;
    bool isInitialized() const override;
    QString name() const override;

private:
    TrtEngine m_engine;

    cv::Mat preprocess(const cv::Mat &frame);

    // 解析双输出：detOut=[1,N,38] 检测张量、protoOut=[1,32,mH,mW] 原型张量。
    // 按各自 shape 解析（detShape 为 3 维、protoShape 为 4 维），坐标映射回原图，
    // 掩膜仅在 box 对应 ROI 内组装（系数 x 原型 → sigmoid → 阈值），返回原图坐标实例。
    std::vector<SegInstance> postprocess(const float *detOut, const float *protoOut,
                                         const std::vector<int> &detShape,
                                         const std::vector<int> &protoShape,
                                         const cv::Size &originalSize,
                                         float confThreshold = 0.25f,
                                         float maskThreshold = 0.5f);

    // 半透明彩色掩膜 + 检测框 + “类别 置信度”标签（尺寸随分辨率自适应）
    cv::Mat draw(const cv::Mat &frame, const std::vector<SegInstance> &insts);
};

#endif // YOLOSEG_H
