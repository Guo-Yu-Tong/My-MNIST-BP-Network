//杨亚童编写
#include "BoxExtractor.h"
#include "Constants.h"
#include <iostream>

using namespace std;
using namespace cv;

BoxExtractor::BoxExtractor() {
    reset();
}

void BoxExtractor::reset() {
    params.isDrawing = false;
    params.points.clear();
    params.image.release();
}

void BoxExtractor::mouseHandler(int event, int x, int y, int flags, void* param) {
    BoxExtractor* self = static_cast<BoxExtractor*>(param);
    self->opencv_mouse_callback(event, x, y, flags, param);
}

void BoxExtractor::opencv_mouse_callback(int event, int x, int y, int flags, void* param) {
    switch (event) {
    case EVENT_MOUSEMOVE:
        if (params.isDrawing) {
            params.points.push_back(Point(x, y));
        }
        break;
    case EVENT_LBUTTONDOWN:
        params.isDrawing = true;
        params.points.clear();
        params.points.push_back(Point(x, y));
        break;
    case EVENT_LBUTTONUP:
        params.isDrawing = false;
        break;
    }
}

Mat BoxExtractor::MouseDraw(Mat& img) {
    MouseDraw(WINDOW_NAME, img, Scalar(255, 255, 255), 5);
    return img;
}

int BoxExtractor::MouseDraw(const string& windowName, Mat& img, Scalar color, int border) {
    int key = 0;
    imshow(windowName, img);
    printf("利用鼠标写下一个或多个数字，按回车输出预测结果，按ESC退出\n");
    params.image = img.clone();
    setMouseCallback(windowName, mouseHandler, this);

    while (!(key == 27 || key == 13)) {
        int length = params.points.size();
        for (int i = 0; i < length - 1 && length > 1; i++) {
            line(params.image, params.points[i], params.points[i + 1], color, border);
        }
        imshow(windowName, params.image);
        key = waitKey(1);
    }
    params.image.copyTo(img);
    return key;
}