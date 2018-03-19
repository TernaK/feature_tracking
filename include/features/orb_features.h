#include <opencv2/opencv.hpp>
#include <iostream>

namespace features {
  class OrbFeatures {
  private:
    cv::Ptr<cv::ORB> orb_detector;
    cv::Ptr<cv::DescriptorMatcher> matcher;

  public:
    struct Match {
      std::vector<cv::KeyPoint> matched;
      std::vector<cv::KeyPoint> matched_src;
      std::vector<cv::KeyPoint> outliers;
    };

    struct Detection {
      cv::Mat descriptors;
      std::vector<cv::KeyPoint> keypoints;
    };

    OrbFeatures();

    Detection detect(const cv::Mat frame);

    Match match(const Detection& prev_det,
                const Detection& curr_det);
  };
}
