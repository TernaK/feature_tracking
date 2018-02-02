#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

class OrbTracker {
  Ptr<ORB> orb_detector;
  Ptr<DescriptorMatcher> matcher;
  Mat old_descriptors;
  vector<KeyPoint> old_keypoints;

  struct tracker_results_t {
    vector<KeyPoint> matched;
    vector<KeyPoint> matched_src;
    vector<KeyPoint> outliers;
  };

public:
  OrbTracker() {
    orb_detector = ORB::create();
    matcher = DescriptorMatcher::create("BruteForce-Hamming");
  }

  tracker_results_t update(const cv::Mat frame) {
    static bool _first = true;
    Mat gray;
    cvtColor(frame, gray, CV_BGR2GRAY);
    cv::GaussianBlur(gray, gray, Size(3,3), 1.0);
    vector<KeyPoint> keypoints;
    Mat descriptors;

    orb_detector->detectAndCompute(gray, noArray(), keypoints, descriptors);

    //first frame
    if(_first) {
      old_descriptors = descriptors;
      old_keypoints = keypoints;
      _first = false;
    }

    //match
    std::vector<std::vector<DMatch>> matches;
    matcher->knnMatch(descriptors, old_descriptors, matches, 2);

    //get best matches
    tracker_results_t results;
    for(auto &match: matches) {
      if(match[0].distance < match[1].distance * 0.8) {
        results.matched.push_back(keypoints[match[0].queryIdx]);
        results.matched_src.push_back(old_keypoints[match[0].trainIdx]);
      }
      else {
        results.outliers.push_back(keypoints[match[0].queryIdx]);
      }
    }

    //track
    old_descriptors = descriptors;
    old_keypoints = keypoints;

    return results;
  }
};

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

int main(int argc, char* args[]) {
  VideoCapture v_cap(0);
  v_cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
  v_cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

  OrbTracker orb_tracker;
  TickMeter tm;

  for(;;) {
    Mat frame;
    v_cap >> frame;

    auto results = orb_tracker.update(frame);
    draw_points(frame, results.outliers, Scalar(0,0,255));
    draw_history(frame, results.matched_src, results.matched);

    imshow("orb_tracking", frame);
    if(waitKey(30) == 27) break;
  }
}
