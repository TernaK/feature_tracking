#include <features/features.h>
#include <features/orb_features.h>
#include <iostream>

using namespace std;
using namespace features;
using namespace cv;

std::string IMAGES_DIR = TEST_IMAGES_DIR;

std::vector<bool> median_filter_matches(const OrbFeatures::Match& match_results) {
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

int main(int argc, char* args[]) {
  OrbFeatures matcher;
  TickMeter tm;
  
  cv::Mat frame0 = cv::imread(IMAGES_DIR + "aloe_l.jpg");
  cv::Mat frame1 = cv::imread(IMAGES_DIR + "aloe_r.jpg");
  
//  cv::resize(frame0, frame0, frame0.size() / 2);
//  cv::resize(frame1, frame1, frame0.size());
  
  OrbFeatures::Detection prev = matcher.detect(frame0);
  OrbFeatures::Detection curr = matcher.detect(frame1);

  OrbFeatures::Match match_results = matcher.match(prev, curr);
  
  vector<bool> mask = median_filter_matches(match_results);
  
//  features::draw_points(frame1, match_results.outliers, Scalar(0,0,255));
  features::draw_history(frame1, match_results.matched_src, match_results.matched);
  
//  cvtColor(frame0, frame0, CV_BGR2GRAY);
//  cvtColor(frame1, frame1, CV_BGR2GRAY);
//  Ptr<StereoMatcher> stereo = StereoBM::create();
//  cv::Mat disparity;
//  stereo->compute(frame0, frame1, disparity);
  
  imshow("orb_tracking", frame1);
  cv::waitKey();
}
