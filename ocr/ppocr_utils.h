#ifndef PPOCR_UTILS_H
#define PPOCR_UTILS_H

#include <string>
#include <vector>
#include <array>
#include <opencv2/opencv.hpp>

// 文本检测框数组：每个框为 4 个点，每个点为 {x, y}
using BoxArray = std::vector<std::vector<std::vector<int>>>;

// 从原图按四点多边形裁剪出（可能旋转的）文本行图像
cv::Mat get_rotate_crop_image(const cv::Mat &src_image, const std::vector<std::vector<int>> &box);

// 图像预处理：等比缩放到高 destH，填充到 destH x destW 画布(letterbox)
// destW 取自识别模型输入宽度，destH 默认 48（识别模型固定高度）
cv::Mat self_resize(cv::Mat src, int destW, int destH = 48);

// 去除字符串首尾空白
std::string trim(const std::string &s);

// DB 文本检测后处理：从概率图 pred_map 还原文本框，并映射回原图坐标系。
// src_h/src_w = 原图尺寸，dst_h/dst_w = 检测输入(概率图)尺寸。
void detector_postprocess(
    const cv::Mat &pred_map, BoxArray &boxes, int src_h, int src_w, int dst_h, int dst_w,
    float mask_thresh, float box_thresh, float unclip_ratio, int min_size, int max_candidates);

#endif // PPOCR_UTILS_H
