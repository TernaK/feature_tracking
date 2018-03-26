#include <features/features.h>
#include <features/orb_features.h>
#include <iostream>

using namespace std;
using namespace features;
using namespace cv;

std::string IMAGES_DIR = PROJ_TEST_IMAGES_DIR;

int main(int argc, char* args[]) {
  if(argc != 3) {
    cerr << "usage: stereo <intrinsics>.yml <extrinsics>.yml" << endl;
    exit(EXIT_FAILURE);
  }
  cv::FileStorage in_data(args[1], FileStorage::READ);
  cv::FileStorage ex_data(args[2], FileStorage::READ);

  cv::Mat camera_matrix_l, camera_matrix_r;
  cv::Mat distortion_l, distortion_r;
  in_data["M1"] >> camera_matrix_l;
  in_data["M2"] >> camera_matrix_r;
  in_data["D1"] >> distortion_l;
  in_data["D2"] >> distortion_r;

  cv::Mat rectification_l, rectification_r;
  cv::Mat projection_l, projection_r;
  ex_data["R1"] >> rectification_l;
  ex_data["R2"] >> rectification_r;
  ex_data["P1"] >> projection_l;
  ex_data["P2"] >> projection_r;

  cv::Mat mapl1, mapl2, mapr1, mapr2;
  cv::initUndistortRectifyMap(camera_matrix_l, distortion_l,
                              rectification_l, projection_l,
                              cv::Size(640,480), CV_32FC1, mapl1, mapl2);
  cv::initUndistortRectifyMap(camera_matrix_r, distortion_r,
                              rectification_r, projection_r,
                              cv::Size(640,480), CV_32FC1, mapr1, mapr2);

  cv::VideoCapture cam1(1);
  cam1.set(CV_CAP_PROP_FRAME_WIDTH, 640);
  cam1.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

  cv::VideoCapture cam2(2);
  cam2.set(CV_CAP_PROP_FRAME_WIDTH, 640);
  cam2.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

  for(;;) {
    cv::Mat frame1;
    cv::Mat frame2;
    cv::Mat temp1, temp2;
    cam1 >> temp1;
    cam2 >> temp2;

    cv::remap(temp1, frame1, mapl1, mapl2, INTER_LINEAR);
    cv::remap(temp2, frame2, mapr1, mapr2, INTER_LINEAR);

    cv::Mat concat;
    cv::hconcat(frame1, frame2, concat);
    imshow("orb_tracking", concat);

    //cv::undistort(temp1, frame1, camera_matrix, dist_coeffs);
    //cv::undistort(temp2, frame2, camera_matrix, dist_coeffs);
    /*
     //OrbFeatures matcher;
     //TickMeter tm;
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
    */

//    features::draw_history(frame2, match_results.matched_src, match_results.matched, mask);


    cvtColor(frame1, frame1, CV_BGR2GRAY);
    cvtColor(frame2, frame2, CV_BGR2GRAY);
    int block = 21;
    Ptr<StereoMatcher> stereo = StereoBM::create();
    stereo->setMinDisparity(4);
    stereo->setNumDisparities(128);
    stereo->setBlockSize(block);
    stereo->setSpeckleRange(16);
    stereo->setSpeckleWindowSize(45);
    cv::Mat disparity;
    stereo->compute(frame1, frame2, disparity);

    disparity.convertTo(disparity, CV_8UC1);
//    //cv::Mat depth = 0.105f * focal / disparity;
    cv::Mat colorized;
    cv::applyColorMap(disparity, colorized, cv::COLORMAP_JET);
    imshow("disparity", colorized);

    if(cv::waitKey(30) == 27) break;
  }
}
