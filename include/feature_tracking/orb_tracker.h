#include <opencv2/opencv.hpp>
#include <iostream>

namespace feature_tracking {
  class OrbTracker {
    cv::Ptr<cv::ORB> orb_detector;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    cv::Mat old_descriptors;
    std::vector<cv::KeyPoint> old_keypoints;

    struct tracker_results_t {
      std::vector<cv::KeyPoint> matched;
      std::vector<cv::KeyPoint> matched_src;
      std::vector<cv::KeyPoint> outliers;
    };

  public:
    OrbTracker();

    tracker_results_t update(const cv::Mat frame);
  };
}
