#include "ppocr_utils.h"

using namespace cv;
using namespace std;

// 图像预处理: 等比缩放到高 destH, 填充到 destH x destW 画布(letterbox)
cv::Mat self_resize(cv::Mat src, int destW, int destH)
{
    int ratio = src.cols / src.rows;
    int resizeW = ratio * destH > destW ? destW : ratio * destH;
    cv::resize(src, src, cv::Size(resizeW, destH));
    cv::Mat dst = cv::Mat::zeros(destH, destW, CV_8UC3);
    cv::Rect roi(0, 0, resizeW, destH);
    src(roi).copyTo(dst(roi));
    return dst;
}

// 去除字符串首尾空白
std::string trim(const std::string &s)
{
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

cv::Mat get_rotate_crop_image(const cv::Mat &src_image, const vector<vector<int>> &box)
{
    std::array<cv::Point, 4> points;
    for (int i = 0; i < 4; ++i) {
        points[i] = cv::Point(box[i][0], box[i][1]);
    }

    auto [minPoint, maxPoint] = std::minmax_element(points.begin(), points.end(), [](const cv::Point &a, const cv::Point &b) {
        return a.x < b.x;
    });
    int left  = minPoint->x;
    int right = maxPoint->x;
    minPoint = std::min_element(points.begin(), points.end(), [](const cv::Point &a, const cv::Point &b) {
        return a.y < b.y;
    });
    maxPoint = std::max_element(points.begin(), points.end(), [](const cv::Point &a, const cv::Point &b) {
        return a.y < b.y;
    });
    int top    = minPoint->y;
    int bottom = maxPoint->y;

    cv::Rect roi(left, top, right - left, bottom - top);
    cv::Mat img_crop = src_image(roi);
    int width  = cv::norm(points[0] - points[1]);
    int height = cv::norm(points[0] - points[3]);

    std::array<cv::Point2f, 4> srcPoints = {
        cv::Point2f(points[0].x - left, points[0].y - top),
        cv::Point2f(points[1].x - left, points[1].y - top),
        cv::Point2f(points[2].x - left, points[2].y - top),
        cv::Point2f(points[3].x - left, points[3].y - top)
    };
    std::array<cv::Point2f, 4> dstPoints = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(width, 0.0f),
        cv::Point2f(width, height),
        cv::Point2f(0.0f, height)
    };

    cv::Mat M = cv::getPerspectiveTransform(srcPoints.data(), dstPoints.data());
    cv::Mat dst_img;
    cv::warpPerspective(img_crop, dst_img, M, cv::Size(width, height), cv::BORDER_REPLICATE);

    if (float(dst_img.rows) >= float(dst_img.cols) * 1.5) {
        cv::Mat rotated;
        cv::transpose(dst_img, rotated);
        cv::flip(rotated, rotated, 0);
        return rotated;
    } else {
        return dst_img;
    }
}
