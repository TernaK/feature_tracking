#include <features/features.h>
#include <features/orb_features.h>
#include <iostream>

using namespace std;
using namespace features;
using namespace cv;

int main(int argc, char* args[]) {
  VideoCapture v_cap;
  v_cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
  v_cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
  v_cap.open(0);

  OrbFeatures matcher;
  TickMeter tm;

  for(;;) {
    Mat frame;
    v_cap >> frame;
    v_cap >> frame;
    v_cap >> frame;
    static OrbFeatures::Detection prev = matcher.detect(frame);
    OrbFeatures::Detection curr = matcher.detect(frame);

    OrbFeatures::Match results = matcher.match(prev, curr);
    vector<bool> mask = OrbFeatures::median_filter_matches(results);
    features::draw_points(frame, results.outliers, Scalar(0,0,255));
    features::draw_points(frame, results.matched);
    features::draw_history(frame, results.matched_src, results.matched, mask);

    std::swap(prev, curr);
    imshow("orb_tracking", frame);
    if(waitKey(30) == 27) break;
  }
}
