//王欣怡编写
#ifndef RECOGNITION_H
#define RECOGNITION_H

#include <opencv2/opencv.hpp>
#include "BPNeuralNetwork.h"

using namespace std;
using namespace cv;

// 识别功能函数
void handWritingRecognize(const Ptr<BPNeuralNetwork>& bpNet);
void imageRecognize(const Ptr<BPNeuralNetwork>& bpNet);

#endif 