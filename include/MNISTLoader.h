//王欣怡编写
#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

// MNIST数据加载相关函数
int ReverseInt(int i);
void ReadMNIST(int NumberOfImages, const string& fileName, vector<Mat>& arr);
void ReadMNISTLabel(int NumberOfImages, const string& fileName, vector<int>& arr);
void loadMNISTData(vector<Mat>& trainImageVec, vector<int>& trainLabelVec, int trainNumber,
    vector<Mat>& testImageVec, vector<int>& testLabelVec, int testNumber);
void testFileLoad(const vector<Mat>& trainImage, const vector<int>& trainLabel);

#endif 