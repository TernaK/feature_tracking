#include <features/orb_features.h>
using namespace cv;
using namespace std;
using namespace features;

OrbFeatures::OrbFeatures() {
  orb_detector = ORB::create();
  matcher = DescriptorMatcher::create("BruteForce-Hamming");
}

OrbFeatures::Detection OrbFeatures::detect(const cv::Mat frame) {
  Mat gray;
  if(frame.channels() == 3) {
    cvtColor(frame, gray, CV_BGR2GRAY);
    cv::GaussianBlur(gray, gray, Size(3,3), 1.0);
  } else {
    gray = frame;
  }
  
  Detection results;
  orb_detector->detectAndCompute(gray, noArray(), results.keypoints, results.descriptors);
  return results;
}

OrbFeatures::Match OrbFeatures::match(const Detection& prev_det,
                                      const Detection& curr_det) {
  //match
  std::vector<std::vector<DMatch>> matches;
  matcher->knnMatch(curr_det.descriptors, prev_det.descriptors, matches, 2);

  //get best matches
  Match results;
  for(auto &match: matches) {
    if(match[0].distance < match[1].distance * 0.8) {
      results.matched.push_back(curr_det.keypoints[match[0].queryIdx]);
      results.matched_src.push_back(prev_det.keypoints[match[0].trainIdx]);
    }
    else {
      results.outliers.push_back(curr_det.keypoints[match[0].queryIdx]);
    }
  }
  return results;
}

std::vector<bool>
OrbFeatures::median_filter_matches(const OrbFeatures::Match& match_results) {
  if(match_results.matched.size() == 0) return {};
  vector<float> diffs;
  for(int i = 0; i < match_results.matched.size(); i++) {
    float px_diff = cv::norm(match_results.matched[i].pt - match_results.matched_src[i].pt);
    diffs.push_back(px_diff);
  }
  
  vector<float> diffs_sorted = diffs;
  std::sort(diffs_sorted.begin(), diffs_sorted.end());
  float median = diffs[diffs_sorted.size()/2];
  
  vector<bool> mask(match_results.matched.size(), false);
  for(int i = 0; i < diffs.size(); i++)
    mask[i] = (fabs(diffs[i] - median) / median) < 1;
  
  return mask;
}
