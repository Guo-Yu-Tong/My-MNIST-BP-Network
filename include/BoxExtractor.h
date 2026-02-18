//杨亚童编写
#ifndef BOX_EXTRACTOR_H
#define BOX_EXTRACTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

// 鼠标绘制类（负责交互绘制数字）
class BoxExtractor {
public:
    struct handlerT {
        Mat image;
        bool isDrawing;
        vector<Point> points;
    } params;

    BoxExtractor();
    void reset();
    static void mouseHandler(int event, int x, int y, int flags, void* param);
    void opencv_mouse_callback(int event, int x, int y, int flags, void* param);
    Mat MouseDraw(Mat& img);
    int MouseDraw(const string& windowName, Mat& img, Scalar color, int border);
};

#endif 