#include <features/features.h>
#include <features/orb_features.h>
#include <iostream>
#include <chrono>
#include <thread>

using namespace std;
using namespace features;
using namespace cv;

std::string IMAGES_DIR = TEST_IMAGES_DIR;

int main(int argc, char* args[]) {
  float focal = 420;
  cv::Mat camera_matrix = (cv::Mat_<float>(3,3) << focal,0,320, 0,focal,240, 0,0,1);
  cv::Mat dist_coeffs = (cv::Mat_<float>(5, 1) <<
                         -3.8976412000174743e-01, 1.9566837393430092e-01,
                         1.3830934799587706e-03, -8.5691494242946136e-04,
                         -5.9753725384274280e-02);

  cv::VideoCapture cam1(1);
  cam1.set(CV_CAP_PROP_FRAME_WIDTH, 640);
  cam1.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

  cv::VideoCapture cam2(2);
  cam2.set(CV_CAP_PROP_FRAME_WIDTH, 640);
  cam2.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
   for(int i = 0; i < 10; i++) {
    /*
    for(int t = 2; t >= 0; t--) {
      cout << t << "..." << std::flush;
      this_thread::sleep_for(chrono::milliseconds(500));
    }
    */

    cv::Mat frame1;
    cv::Mat frame2;
    cv::Mat temp1, temp2;
    cam1 >> temp1;
    cam2 >> temp2;
    }

  for(int i = -5; i < 10; i++) {
    /*
    for(int t = 2; t >= 0; t--) {
      cout << t << "..." << std::flush;
      this_thread::sleep_for(chrono::milliseconds(500));
    }
    */
    cv::waitKey(1000);

    cv::Mat frame1;
    cv::Mat frame2;
    cv::Mat temp1, temp2;
    cam1 >> temp1;
    cam2 >> temp2;

    cv::undistort(temp1, frame1, camera_matrix, dist_coeffs);
    cv::undistort(temp2, frame2, camera_matrix, dist_coeffs);

    cv::imwrite(cv::format("left_%d.jpg", i), frame1);
    cv::imwrite(cv::format("right_%d.jpg", i), frame2);
    cout << "captured" << endl;
  }
}
