#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

class OrbTracker {
  Ptr<ORB> orb_detector;
  Ptr<BFMatcher> matcher;
};

int main(int argc, char* args[]) {
  cout << "hello" << endl;
}
