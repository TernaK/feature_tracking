#include <features/features.h>
#include <features/orb_features.h>
#include <iostream>

using namespace std;
using namespace features;
using namespace cv;

std::string IMAGES_DIR = TEST_IMAGES_DIR;

int main(int argc, char* args[]) {
  float focal = 420;
  cv::Mat camera_matrix = (cv::Mat_<float>(3,3) << focal,0,320, 0,focal,240, 0,0,1);
  cv::Mat dist_coeffs = (cv::Mat_<float>(5, 1) <<
                         -3.8976412000174743e-01, 1.9566837393430092e-01,
-                         1.3830934799587706e-03, -8.5691494242946136e-04,
-                         -5.9753725384274280e-02);
  
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
    
    vector<vector<cv::Point3f>> pts(6, vector<cv::Point3f>());
    for(int i = 0; i < 6; i++) 
      for(int j = 0; j < 9; j++) 
        pts[i].push_back(cv::Point3f(j*0.024, i*0.024, 0));
        
    
    vector<cv::Point2f> _pts1, _pts2;
    bool found1 = cv::findChessboardCorners(frame1, cv::Size(9,6), _pts1);
    bool found2 = cv::findChessboardCorners(frame2, cv::Size(9,6), _pts2);
    vector<vector<cv::Point2f>> pts1(6);
    vector<vector<cv::Point2f>> pts2(6);
    int idx = 0;
    for(int i = 0; i < 6; i++) {
      for(int j = 0; j < 9; j++) {
        pts1[i].push_back(_pts1[idx]);
        pts2[i].push_back(_pts2[idx]);
        idx++;
      }
    } 
    
    if(!(found1 && found2)) continue;
    
    cv::Mat R, T, E, F;
    cv::stereoCalibrate(pts, pts1, pts2, camera_matrix, dist_coeffs, camera_matrix, dist_coeffs,
                        cv::Size(640,480), R, T, E, F);
                        
    cout << "R" << endl << R << endl << endl;
    cout << "T" << endl << T << endl << endl;
    cout << "E" << endl << E << endl << endl;
    cout << "F" << endl << F << endl << endl;
    break;
  }
}
