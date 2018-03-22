#include <features/features.h>
#include <features/orb_features.h>
#include <iostream>

using namespace std;
using namespace features;
using namespace cv;

std::string IMAGES_DIR = TEST_IMAGES_DIR;

int main(int argc, char* args[]) {
  cv::Mat camera_matrix = (cv::Mat_<float>(3,3) << 420,0,0, 0,420,0, 0,0,1);
  cv::Mat dist_coeffs = (cv::Mat_<float>(5, 1) <<
                         -3.8976412000174743e-01, 1.9566837393430092e-01,
                         1.3830934799587706e-03, -8.5691494242946136e-04,
                         -5.9753725384274280e-02);
  
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
    
    cv::undistort(frame1, frame1, camera_matrix, dist_coeffs);
    cv::undistort(frame2, frame2, camera_matrix, dist_coeffs);
    
    OrbFeatures::Detection feat1 = matcher.detect(frame1);
    OrbFeatures::Detection feat2 = matcher.detect(frame2);

    OrbFeatures::Match match_results = matcher.match(feat2, feat1);
    
    vector<bool> mask = OrbFeatures::median_filter_matches(match_results);
    
    features::draw_points(frame1, match_results.matched);
    features::draw_points(frame2, match_results.matched_src, cv::Scalar(255,255,0));
    
    cv::Mat concat;
    cv::hconcat(frame1, frame2, concat);
    for(size_t k = 0; k < match_results.matched_src.size(); k++) {
      if(!mask.empty() && !mask[k]) continue;
      cv::Point pt1(frame1.cols + match_results.matched_src[k].pt.x, match_results.matched_src[k].pt.y);
      cv::Point pt2(match_results.matched[k].pt.x, match_results.matched[k].pt.y);
      cv::line(concat, pt1, pt2, cv::Scalar(0,255,0));
    }
//    features::draw_history(frame2, match_results.matched_src, match_results.matched, mask);
    
//    cvtColor(frame1, frame1, CV_BGR2GRAY);
//    cvtColor(frame2, frame2, CV_BGR2GRAY);
//    Ptr<StereoMatcher> stereo = StereoBM::create();
//    cv::Mat disparity;
//    stereo->compute(frame1, frame2, disparity);
//
//    imshow("orb_tracking", disparity * 1000);
    imshow("orb_tracking", concat);
    if(cv::waitKey(30) == 27) break;
  }
}
