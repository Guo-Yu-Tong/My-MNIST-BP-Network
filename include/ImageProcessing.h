//郭雨彤编写
#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "Constants.h"

using namespace std;
using namespace cv;

// 图像预处理函数
void preProcess(const Mat& srcImage, Mat& dstImage);
void resizeImage(Mat& srcImage, Size size);
void binaryInit(Mat& srcDataMat, Size size);
void normal(const Mat& srcImage, Mat& dstImage);
void image2Vec(const Mat& srcImage, Mat& dstImage);

// 特征提取函数
float calBigNumber(const Mat& srcDataMat, float thresh);
void getHistogram(const Mat& srcMat, Mat& histogramMat, int flag);
void getAllPixelsFeature(const Mat& srcDataMat, Mat& dstDataMat, Size size);
void getHistogramFeature(const Mat& srcDataMat, Mat& dstDataMat, Size size);
void getMixedFeature(const Mat& srcDataMat, Mat& dstDataMat, Size size);

// 图像分割函数
void showProcessingStep(const string& windowName, const Mat& image, const string& stepName, int waitTime = 1000);
void getSegment(const Mat& srcImage, vector<Mat>& arr, Mat& showImage);

// 矩形排序辅助函数
bool cmp(const Rect& a, const Rect& b);
bool cmp2(const Rect& a, const Rect& b);
void sortRect(vector<Rect>& arr);

#endif 