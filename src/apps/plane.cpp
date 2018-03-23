#include <features/features.h>
#include <features/orb_features.h>
#include <iostream>

using namespace std;
using namespace features;
using namespace cv;

std::string IMAGES_DIR = TEST_IMAGES_DIR;

int main(int argc, char* args[]) {
  float focal = 394;
  cv::Mat camera_matrix = (cv::Mat_<float>(3,3) << focal,0,320, 0,focal,240, 0,0,1);
  cv::Mat dist_coeffs = (cv::Mat_<float>(5, 1) <<
                         -3.8929828864558985e-01, 2.3330816257346310e-01,
       -2.8870364538126701e-03, -5.6174933073543411e-03,
       -8.7092628090518112e-02);
  
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
    cv::Mat temp1, temp2;
    cam1 >> temp1;
    cam2 >> temp2;
    
    cv::undistort(temp1, frame1, camera_matrix, dist_coeffs);
    cv::undistort(temp2, frame2, camera_matrix, dist_coeffs);
    
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
    imshow("orb_tracking", concat);
    
//    features::draw_history(frame2, match_results.matched_src, match_results.matched, mask);
    
    /*
    cvtColor(frame1, frame1, CV_BGR2GRAY);
    cvtColor(frame2, frame2, CV_BGR2GRAY);
    int block = 15;
    Ptr<StereoMatcher> stereo = StereoSGBM::create(0, 16, block);
    cv::Mat disparity;
    stereo->compute(frame1, frame2, disparity);
    
    disparity.convertTo(disparity, CV_32F);
    cv::Mat depth = 0.105f * focal / disparity;
    imshow("orb_tracking", depth);
    */
    if(cv::waitKey(30) == 27) break;
  }
}
