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
  
  cv::VideoCapture cam1(1);
  cam1.set(CV_CAP_PROP_FRAME_WIDTH, 640);
  cam1.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
  
  cv::VideoCapture cam2(2);
  cam2.set(CV_CAP_PROP_FRAME_WIDTH, 640);
  cam2.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
  
  for(;;) {
    cv::Mat frame1;// = cv::imread(IMAGES_DIR + "aloe_l.jpg");
    cv::Mat frame2;// = cv::imread(IMAGES_DIR + "aloe_r.jpg");
    
    cam1 >> frame1;
    cam2 >> frame2;
    
  //  cv::resize(frame0, frame0, frame0.size() / 2);
  //  cv::resize(frame1, frame1, frame0.size());
    
    OrbFeatures::Detection feat1 = matcher.detect(frame1);
    OrbFeatures::Detection feat2 = matcher.detect(frame2);

    OrbFeatures::Match match_results = matcher.match(feat2, feat1);
    
    vector<bool> mask = OrbFeatures::median_filter_matches(match_results);
    
    features::draw_points(frame1, match_results.matched);
//    features::draw_history(frame2, match_results.matched_src, match_results.matched, mask);
    
//    cvtColor(frame1, frame1, CV_BGR2GRAY);
//    cvtColor(frame2, frame2, CV_BGR2GRAY);
//    Ptr<StereoMatcher> stereo = StereoBM::create();
//    cv::Mat disparity;
//    stereo->compute(frame1, frame2, disparity);
//
//    imshow("orb_tracking", disparity * 1000);
    imshow("orb_tracking", frame1);
    if(cv::waitKey(30) == 27) break;
  }
}
