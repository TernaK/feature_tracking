#include <features/features.h>
#include <features/orb_features.h>
#include <iostream>

using namespace std;
using namespace features;
using namespace cv;

std::string IMAGES_DIR = TEST_IMAGES_DIR;

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
  
  vector<bool> mask = OrbFeatures::median_filter_matches(match_results);
  
//  features::draw_points(frame1, match_results.outliers, Scalar(0,0,255));
  features::draw_history(frame1, match_results.matched_src, match_results.matched, mask);
  
//  cvtColor(frame0, frame0, CV_BGR2GRAY);
//  cvtColor(frame1, frame1, CV_BGR2GRAY);
//  Ptr<StereoMatcher> stereo = StereoBM::create();
//  cv::Mat disparity;
//  stereo->compute(frame0, frame1, disparity);
  
  imshow("orb_tracking", frame1);
  cv::waitKey();
}
