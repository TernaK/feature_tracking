#include <feature_tracking/orb_tracker.h>
using namespace cv;
using namespace std;
using namespace feature_tracking;

OrbTracker::OrbTracker() {
  orb_detector = ORB::create();
  matcher = DescriptorMatcher::create("BruteForce-Hamming");
}

OrbTracker::tracker_results_t OrbTracker::update(const cv::Mat frame) {
  static bool _first = true;
  Mat gray;
  cvtColor(frame, gray, CV_BGR2GRAY);
  cv::GaussianBlur(gray, gray, Size(3,3), 1.0);
  vector<KeyPoint> keypoints;
  Mat descriptors;

  orb_detector->detectAndCompute(gray, noArray(), keypoints, descriptors);

  //first frame
  if(_first) {
    old_descriptors = descriptors;
    old_keypoints = keypoints;
    _first = false;
  }

  //match
  std::vector<std::vector<DMatch>> matches;
  matcher->knnMatch(descriptors, old_descriptors, matches, 2);

  //get best matches
  tracker_results_t results;
  for(auto &match: matches) {
    if(match[0].distance < match[1].distance * 0.8) {
      results.matched.push_back(keypoints[match[0].queryIdx]);
      results.matched_src.push_back(old_keypoints[match[0].trainIdx]);
    }
    else {
      results.outliers.push_back(keypoints[match[0].queryIdx]);
    }
  }

  //track
  old_descriptors = descriptors;
  old_keypoints = keypoints;

  return results;
}
