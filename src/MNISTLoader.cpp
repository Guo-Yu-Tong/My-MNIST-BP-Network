//王欣怡编写
#include "MNISTLoader.h"
#include "Constants.h"
#include <iostream>
#include <fstream>
#include <algorithm>

using namespace std;
using namespace cv;

int ReverseInt(int i) {
    unsigned char ch1 = i & 255;
    unsigned char ch2 = (i >> 8) & 255;
    unsigned char ch3 = (i >> 16) & 255;
    unsigned char ch4 = (i >> 24) & 255;
    return (int)ch1 << 24 | (int)ch2 << 16 | (int)ch3 << 8 | ch4;
}

// 读取MNIST图像数据
void ReadMNIST(int NumberOfImages, const string& fileName, vector<Mat>& arr) {
    ifstream file(fileName, ios::binary);
    if (!file.is_open()) {
        cerr << "无法打开MNIST图像文件: " << fileName << endl;
        return;
    }

    int magic_number = 0, number_of_images = 0, n_rows = 0, n_cols = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = ReverseInt(magic_number);
    file.read((char*)&number_of_images, sizeof(number_of_images));
    number_of_images = ReverseInt(number_of_images);
    file.read((char*)&n_rows, sizeof(n_rows));
    n_rows = ReverseInt(n_rows);
    file.read((char*)&n_cols, sizeof(n_cols));
    n_cols = ReverseInt(n_cols);

    int readCount = min(NumberOfImages, number_of_images);
    arr.reserve(readCount);

    for (int i = 0; i < readCount; ++i) {
        Mat tmpMat(n_rows, n_cols, CV_8UC1);
        for (int r = 0; r < n_rows; ++r) {
            file.read((char*)tmpMat.ptr<uchar>(r), sizeof(uchar) * n_cols);
        }
        arr.push_back(tmpMat);
    }
    file.close();
}

// 读取MNIST标签数据
void ReadMNISTLabel(int NumberOfImages, const string& fileName, vector<int>& arr) {
    ifstream file(fileName, ios::binary);
    if (!file.is_open()) {
        cerr << "无法打开MNIST标签文件: " << fileName << endl;
        return;
    }

    int magic_number = 0, number_of_labels = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = ReverseInt(magic_number);
    file.read((char*)&number_of_labels, sizeof(number_of_labels));
    number_of_labels = ReverseInt(number_of_labels);

    int readCount = min(NumberOfImages, number_of_labels);
    arr.reserve(readCount);

    for (int i = 0; i < readCount; ++i) {
        unsigned char tmpLabel;
        file.read((char*)&tmpLabel, sizeof(tmpLabel));
        arr.push_back((int)tmpLabel);
    }
    file.close();
}

// 加载MNIST数据
void loadMNISTData(vector<Mat>& trainImageVec, vector<int>& trainLabelVec, int trainNumber,
    vector<Mat>& testImageVec, vector<int>& testLabelVec, int testNumber) {
    ReadMNIST(trainNumber, g_TrainImageFileName, trainImageVec);
    ReadMNISTLabel(trainNumber, g_TrainLabelFileName, trainLabelVec);
    ReadMNIST(testNumber, g_TestImageFileName, testImageVec);
    ReadMNISTLabel(testNumber, g_TestLabelFileName, testLabelVec);
}

// 测试数据加载
void testFileLoad(const vector<Mat>& trainImage, const vector<int>& trainLabel) {
    if (trainImage.empty() || trainLabel.empty()) {
        cout << "数据加载为空，无法测试" << endl;
        return;
    }
    cout << "测试加载的前10个标签: " << endl;
    for (int i = 0; i < min(10, (int)trainLabel.size()); ++i) {
        cout << trainLabel[i] << " ";
    }
    cout << endl;
    cout << "显示前10个图像（每个图像显示1秒）" << endl;
    for (int i = 0; i < min(10, (int)trainImage.size()); ++i) {
        Mat showMat;
        resize(trainImage[i], showMat, Size(200, 200));
        imshow("MNIST Test", showMat);
        waitKey(1000);
    }
    destroyWindow("MNIST Test");
}