#ifndef ANOMALYDETECT_H
#define ANOMALYDETECT_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "itask.h"
#include "trtengine.h"

// 无监督异常检测任务：加载 anomalib(PatchCore) 导出、再经 trtexec 构建的 engine。
//
// 模型 IO 约定（归一化与阈值均已烤进 engine，C++ 侧不可重复处理）：
//   输入 input  FLOAT [1,3,256,256]：BGR→RGB → resize(256,256) → x/255（[0,1]）。
//                                     ImageNet mean/std 已在图内，切勿再做归一化。
//   输出（4 个，绑定顺序不保证，一律按名字取用）：
//     pred_score  FLOAT [1]            图像级异常分，已归一化到[0,1]，>0.5 判异常
//     anomaly_map FLOAT [1,1,256,256]  像素级热力图，已归一化到[0,1]，0.5=阈值
//     pred_label  BOOL  [1]            = pred_score>0.5（本任务用分数重算，规避 bool 输出）
//     pred_mask   BOOL  [1,1,256,256]  = anomaly_map>0.5（本任务用热力图重算）
//
// 可视化 1:1 复刻 anomalib：自定义 9 锚点 "jet" 配色（双线性插值到 256 级），
// 不做 min-max 归一化（map 已在[0,1]），热力图与原图 alpha=0.5 叠加，缺陷区红描边。
class AnomalyDetect : public ITask
{
public:
    bool init(const QString &enginePath) override;
    TaskResult run(const cv::Mat &frame) override;
    void release() override;
    bool isInitialized() const override;
    QString name() const override;

private:
    TrtEngine m_engine;

    int     m_scoreIdx = -1;   // pred_score 的输出索引
    int     m_mapIdx   = -1;   // anomaly_map 的输出索引
    cv::Mat m_jetLut;          // 256x1 BGR 查找表（复刻 anomalib 的 jet 配色）

    // 定位输出索引：优先按名字（pred_score/anomaly_map），失败则按 dtype+shape 兜底
    void resolveOutputs();
    // 构建 anomalib apply_colormap 的 9 锚点 jet LUT（256 级，BGR）
    void buildJetLut();
    // 预处理：resize + BGR2RGB + /255，输出 NCHW float blob（不做 ImageNet 归一化）
    cv::Mat preprocess(const cv::Mat &frame);
    // 把[0,1]的 anomaly_map 套 jet LUT 转成 BGR 热力图（尺寸同输入 map）
    cv::Mat colorize(const cv::Mat &mapNorm);
    // 组合可视化画布：原图 | 热力图 | 叠加图 + 判定/分数文字
    cv::Mat draw(const cv::Mat &frame, const cv::Mat &heat,
                 const cv::Mat &mask, double score, int regions);
};

#endif // ANOMALYDETECT_H
