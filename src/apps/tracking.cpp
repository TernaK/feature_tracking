#include <features/orb_features.h>
#include <iostream>

using namespace std;
using namespace features;
using namespace cv;

int main(int argc, char* args[]) {

  auto draw_points = [](cv::Mat frame, vector<KeyPoint> kps, Scalar color = Scalar(0,255,0)) {
    for(auto &kp: kps) {
      Point pt(kp.pt.x, kp.pt.y);
      drawMarker(frame, pt, color);
    }
  };

  auto draw_history = [](cv::Mat frame, const vector<KeyPoint>& kps_src, const vector<KeyPoint>& kps_dst) {
    for(size_t k = 0; k < kps_src.size(); k++) {
      Point pt1(kps_src[k].pt.x, kps_src[k].pt.y);
      Point pt2(kps_dst[k].pt.x, kps_dst[k].pt.y);
      arrowedLine(frame, pt1, pt2, Scalar(0,255,0));
    }
  };

  VideoCapture v_cap;
  v_cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
  v_cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
  v_cap.open(0);

  OrbFeatures matcher;
  TickMeter tm;

  for(;;) {
    Mat frame;
    v_cap >> frame;
    static OrbFeatures::Detection prev = matcher.detect(frame);
    OrbFeatures::Detection curr = matcher.detect(frame);

    OrbFeatures::Match results = matcher.match(prev, curr);
    draw_points(frame, results.outliers, Scalar(0,0,255));
    draw_history(frame, results.matched_src, results.matched);

    std::swap(prev, curr);
    imshow("orb_tracking", frame);
    if(waitKey(30) == 27) break;
  }
}
