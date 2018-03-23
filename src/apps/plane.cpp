#include <features/features.h>
#include <features/orb_features.h>
#include <iostream>

using namespace std;
using namespace features;
using namespace cv;

std::string IMAGES_DIR = TEST_IMAGES_DIR;

int main(int argc, char* args[]) {
  cv::Mat camera_matrix_l = (cv::Mat_<float>(3,3) << 4.7501191048986709e+02, 0., 320.,
                                                   0., 4.7858798481066032e+02, 240.,
                                                   0., 0., 1.);
  cv::Mat camera_matrix_r = (cv::Mat_<float>(3,3) << 4.8107424191850265e+02, 0., 320.,
                                                   0., 4.8035520879056799e+02, 240.,
                                                   0., 0., 1.);
  cv::Mat rectification_l = (cv::Mat_<float>(3,3) << 
       8.9457818307285086e-01, -9.0561398684157607e-03, 4.4681971834371920e-01,
       -2.0479997752500551e-03, 9.9970109882601965e-01, 2.4362239284794397e-02,
       -4.4690679125181954e-01, -2.2709014437722608e-02, 8.9429224563129806e-01);
  cv::Mat rectification_r = (cv::Mat_<float>(3,3) << 
       8.6908558257604940e-01, 2.0296179366034808e-02, 4.9424519750989043e-01,
       -7.9267422012155916e-03, 9.9960102351439162e-01, -2.7110155791808921e-02,
       -4.9459823788255081e-01, 1.9643291275201955e-02, 8.6889972044612451e-01);
       
  cv::Mat projection_l = (cv::Mat_<float>(3,4) << 
       -1.3324889283290645e+03, 0., -1.0541644167900085e+02, 0., 0.,
       -1.3324889283290645e+03, 2.4214758110046387e+02, 0., 0., 0., 1.,
       0. );
              
  cv::Mat projection_r = (cv::Mat_<float>(3,4) << 
       -1.3324889283290645e+03, 0., -1.0541644167900085e+02,
       1.4675061829794714e+02, 0., -1.3324889283290645e+03,
       2.4214758110046387e+02, 0., 0., 0., 1., 0. );
                                                   
                                                   
  cv::Mat mapl1, mapl2, mapr1, mapr2;
  cv::initUndistortRectifyMap(camera_matrix_l, cv::Mat(), rectification_l, projection_l, cv::Size(640,480), CV_32FC1, mapl1, mapl2);
  cv::initUndistortRectifyMap(camera_matrix_r, cv::Mat(), rectification_r, projection_r, cv::Size(640,480), CV_32FC1, mapr1, mapr2);
  
  //cv::Mat dist_coeffs_l = (cv::Mat_<float>(5, 1) <<
  //                       -3.8929828864558985e-01, 2.3330816257346310e-01,
  //     -2.8870364538126701e-03, -5.6174933073543411e-03,
  //     -8.7092628090518112e-02);
  
  //OrbFeatures matcher;
  //TickMeter tm;
  
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
    
    cv::remap(temp1, frame1, mapl1, mapl2, INTER_LINEAR);
    cv::remap(temp2, frame2, mapr1, mapr2, INTER_LINEAR);
    //cv::undistort(temp1, frame1, camera_matrix, dist_coeffs);
    //cv::undistort(temp2, frame2, camera_matrix, dist_coeffs);
    /*
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
    int block = 15;
    Ptr<StereoMatcher> stereo = StereoBM::create(16, block);
    cv::Mat disparity;
    stereo->compute(frame1, frame2, disparity);
    
    //disparity.convertTo(disparity, CV_32F);
    //cv::Mat depth = 0.105f * focal / disparity;
    imshow("orb_tracking", frame2);
    
    if(cv::waitKey(30) == 27) break;
  }
}
