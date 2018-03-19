#pragma once
#include <opencv2/opencv.hpp>

namespace features {
  static void draw_points(cv::Mat frame,
                          std::vector<cv::KeyPoint> kps,
                          cv::Scalar color = cv::Scalar(0,255,0)) {
    for(auto &kp: kps) {
      cv::Point pt(kp.pt.x, kp.pt.y);
      cv::drawMarker(frame, pt, color);
    }
  };

  static void draw_history(cv::Mat frame,
                           const std::vector<cv::KeyPoint>& kps_src,
                           const std::vector<cv::KeyPoint>& kps_dst) {
    for(size_t k = 0; k < kps_src.size(); k++) {
      cv::Point pt1(kps_src[k].pt.x, kps_src[k].pt.y);
      cv::Point pt2(kps_dst[k].pt.x, kps_dst[k].pt.y);
      cv::arrowedLine(frame, pt1, pt2, cv::Scalar(0,255,0));
    }
  };
}
