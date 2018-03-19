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
  
  cv::Mat frame0 = cv::imread(IMAGES_DIR + "path0.jpg");
  cv::Mat frame1 = cv::imread(IMAGES_DIR + "path1.jpg");
  
  OrbFeatures::Detection prev = matcher.detect(frame0);
  OrbFeatures::Detection curr = matcher.detect(frame1);

  OrbFeatures::Match results = matcher.match(prev, curr);
//  features::draw_points(frame1, results.outliers, Scalar(0,0,255));
  features::draw_history(frame1, results.matched_src, results.matched);
  
  imshow("orb_tracking", frame1);
  cv::waitKey();
}
