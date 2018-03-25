#include <features/features.h>

using namespace features;
using namespace cv;
using namespace std;

/// Camera
//--------------------------------------------------
Camera::Camera(cv::Mat camera_matrix, cv::Mat distortion)
: camera_matrix(camera_matrix), distortion(distortion) {

}

Camera::Camera(std::string calib_path, cv::Size size) {
  FileStorage fs(calib_path, FileStorage::READ);
  fs["camera_matrix"] >> camera_matrix;
  fs["distortion_coefficients"] >> distortion;
  if(size.area() > 0)
    rescale_camera_matrix(size);
}

void Camera::rescale_camera_matrix(cv::Size size) {
  camera_matrix = rescale_camera_matrix(camera_matrix, size);
}

cv::Mat Camera::rescale_camera_matrix(cv::Mat cam_mat, cv::Size size) {
  float x_scale = size.width / cam_mat.at<double>(0,0);
  float y_scale = size.width / cam_mat.at<double>(1,1);
  cv::Mat output = cam_mat.clone();
  output.at<double>(0,0) = x_scale * cam_mat.at<double>(0,0);
  output.at<double>(1,1) = y_scale * cam_mat.at<double>(1,1);
  output.at<double>(0,2) = size.width / 2.0;
  output.at<double>(1,2) = size.height / 2.0;
  return output;
}
