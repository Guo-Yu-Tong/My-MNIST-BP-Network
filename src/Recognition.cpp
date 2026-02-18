//王欣怡编写
#include "Recognition.h"
#include "Constants.h"
#include "BoxExtractor.h"
#include "ImageProcessing.h"
#include <iostream>

using namespace std;
using namespace cv;

// 鼠标手写识别
void handWritingRecognize(const Ptr<BPNeuralNetwork>& bpNet) {
    if (bpNet.empty()) {
        cout << "BP模型未初始化！" << endl;
        return;
    }

    const string windowName = WINDOW_NAME;
    bool windowCreated = false;
    BoxExtractor drawer;

    try {
        namedWindow(windowName, WINDOW_AUTOSIZE);
        windowCreated = true;

        while (true) {
            drawer.reset();
            Mat canvas(500, 500, CV_8UC1, Scalar::all(0));
            Mat drawMat = canvas.clone();
            int key = drawer.MouseDraw(windowName, drawMat, Scalar::all(255), 5);

            if (key == 27) {
                break;
            }
            else if (key == 13) {
                vector<Mat> digit_mats;
                Mat showMat = drawMat.clone();
                cvtColor(showMat, showMat, COLOR_GRAY2BGR);

                cout << "\n开始图像处理和数字分割..." << endl;

                // 使用带可视化的分割函数
                getSegment(drawMat, digit_mats, showMat);

                if (digit_mats.empty()) {
                    cout << "未检测到数字区域！请检查:" << endl;
                    cout << "1. 数字是否足够清晰" << endl;
                    cout << "2. 数字大小是否合适" << endl;
                    cout << "3. 背景是否干净" << endl;
                    imshow(windowName, showMat);
                    waitKey(2000);
                    continue;
                }

                cout << "\n开始数字识别..." << endl;
                cout << "预测结果: ";

                for (size_t i = 0; i < digit_mats.size(); ++i) {
                    const auto& digit_mat = digit_mats[i];
                    Mat feature_mat;
                    getMixedFeature(digit_mat, feature_mat, Size(28, 28));

                    vector<double> feat_vec(feature_mat.cols);
                    for (int j = 0; j < feature_mat.cols; ++j) {
                        feat_vec[j] = static_cast<double>(feature_mat.at<float>(0, j));
                    }

                    int pred = bpNet->predict(feat_vec);
                    cout << pred << " ";

                    // 在图像上显示预测结果
                    // 这里可以添加将预测结果叠加到图像上的代码
                }
                cout << endl;

                imshow(windowName, showMat);
                cout << "\n按ESC退出，按任意其他键继续绘制..." << endl;
                key = waitKey(0);
                if (key == 27) {
                    break;
                }
                drawMat.setTo(0);
                imshow(windowName, drawMat);
            }
        }
    }
    catch (const cv::Exception& e) {
        cout << "OpenCV异常: " << e.what() << endl;
    }

    if (windowCreated && !windowName.empty()) {
        if (cv::getWindowProperty(windowName, cv::WND_PROP_VISIBLE) >= 0) {
            destroyWindow(windowName);
        }
    }
}

// 图片文件识别
void imageRecognize(const Ptr<BPNeuralNetwork>& bpNet) {
    if (bpNet.empty()) {
        cout << "BP模型未初始化！" << endl;
        return;
    }
    string img_path;
    Mat src_img;
    do {
        cout << "请输入图片路径（相对路径或绝对路径）: ";
        cin >> img_path;
        src_img = imread(img_path);
        if (src_img.empty()) {
            cout << "图片读取失败，请重新输入！" << endl;
        }
    } while (src_img.empty());
    Mat gray_img;
    cvtColor(src_img, gray_img, COLOR_BGR2GRAY);
    vector<Mat> digit_mats;
    Mat show_mat = src_img.clone();
    getSegment(gray_img, digit_mats, show_mat);
    if (digit_mats.empty()) {
        cout << "未检测到数字区域！" << endl;
        imshow("识别结果", show_mat);
        waitKey(0);
        destroyWindow("识别结果");
        return;
    }
    cout << "预测结果: ";
    for (const auto& digit_mat : digit_mats) {
        Mat feature_mat;
        getMixedFeature(digit_mat, feature_mat, Size(28, 28));
        vector<double> feat_vec(feature_mat.cols);
        for (int i = 0; i < feature_mat.cols; ++i) {
            feat_vec[i] = static_cast<double>(feature_mat.at<float>(0, i));
        }
        int pred = bpNet->predict(feat_vec);
        cout << pred << " ";
    }
    cout << endl;
    imshow("识别结果", show_mat);
    cout << "按任意键关闭窗口..." << endl;
    waitKey(0);
    destroyWindow("识别结果");
}