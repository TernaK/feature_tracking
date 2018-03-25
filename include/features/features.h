#pragma once
#include <opencv2/opencv.hpp>

namespace features {

  // global
  //--------------------------------------------------
  static inline void draw_points(cv::Mat frame,
                                 std::vector<cv::KeyPoint> kps,
                                 cv::Scalar color = cv::Scalar(0,255,0)) {
    for(auto &kp: kps) {
      cv::Point pt(kp.pt.x, kp.pt.y);
      cv::drawMarker(frame, pt, color, cv::MARKER_SQUARE, 10, 1);
    }
  };

  static inline void draw_history(cv::Mat frame,
                                  const std::vector<cv::KeyPoint>& kps_src,
                                  const std::vector<cv::KeyPoint>& kps_dst,
                                  std::vector<bool> mask = {}) {
    for(size_t k = 0; k < kps_src.size(); k++) {
      if(!mask.empty() && !mask[k]) continue;
      cv::Point pt1(kps_src[k].pt.x, kps_src[k].pt.y);
      cv::Point pt2(kps_dst[k].pt.x, kps_dst[k].pt.y);
      cv::line(frame, pt1, pt2, cv::Scalar(0,255,0));
    }
  };

  static inline std::vector<cv::Point2f>
  keypoints_to_points(std::vector<cv::KeyPoint> kps,
                      std::vector<bool> mask = {}) {
    std::vector<cv::Point2f> pts;
    for(int k = 0; k < kps.size(); k++) {
      if(!mask.empty() && !mask[k]) continue;
      pts.push_back(kps[k].pt);
    }
    return pts;
  }

  /// Camera
  //--------------------------------------------------
  struct Camera {
    cv::Mat camera_matrix;
    cv::Mat distortion;

    Camera() = default;

    Camera(std::string calib_path, cv::Size size = cv::Size(0,0));

    Camera(cv::Mat camera_matrix, cv::Mat distortion = cv::Mat());

    void rescale_camera_matrix(cv::Size size);

    static cv::Mat rescale_camera_matrix(cv::Mat cam_mat, cv::Size size);
  };
}
