//郭雨彤编写
#include "ImageProcessing.h"
#include <iostream>
#include <algorithm>
#include <numeric>

using namespace std;
using namespace cv;

// 预处理函数（双边滤波保留边缘，优化数字1识别）
void preProcess(const Mat& srcImage, Mat& dstImage) {
    Mat tmpImage = srcImage.clone();
    if (tmpImage.type() != CV_8UC1) {
        cvtColor(tmpImage, tmpImage, COLOR_BGR2GRAY);
    }
    // 双边滤波：保留边缘同时降噪（对细长数字1关键）
    bilateralFilter(tmpImage, tmpImage, 9, 75, 75);
    Canny(tmpImage, dstImage, 50, 150, 3);
    // 椭圆结构元膨胀，增强边缘连续性
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(dstImage, dstImage, MORPH_DILATE, element, Point(-1, -1), 1);
}

// 图像调整大小（根据面积选择插值方式）
void resizeImage(Mat& srcImage, Size size) {
    if (srcImage.rows * srcImage.cols < size.area()) {
        resize(srcImage, srcImage, size, 0, 0, INTER_CUBIC);  // 放大用立方插值
    }
    else if (srcImage.rows * srcImage.cols > size.area()) {
        resize(srcImage, srcImage, size, 0, 0, INTER_AREA);   // 缩小用面积插值
    }
}

// 图像二值初始化（归一化+二值化+尺寸调整）
void binaryInit(Mat& srcDataMat, Size size) {
    srcDataMat.convertTo(srcDataMat, CV_8UC1);
    Mat normalizedMat;
    normal(srcDataMat, normalizedMat);  // 数字居中归一化
    srcDataMat = normalizedMat;  // 避免clone()
    resizeImage(srcDataMat, size);      // 调整到目标尺寸
    // OTSU自动阈值二值化（适应不同亮度）
    threshold(srcDataMat, srcDataMat, 0, 255, THRESH_OTSU + THRESH_BINARY);
    srcDataMat.convertTo(srcDataMat, CV_32FC1);  // 转为浮点型供后续计算
}

// 图像归一化（数字居中，解决细长数字1偏移问题）
void normal(const Mat& srcImage, Mat& dstImage) {
    Mat tmpImage;
    if (srcImage.type() != CV_8UC1) {
        srcImage.convertTo(tmpImage, CV_8UC1);
    }
    else {
        tmpImage = srcImage;  // 避免不必要的拷贝
    }

    // 找到数字的有效区域（非零像素边界）
    int bottom = tmpImage.rows, top = 0;
    int left = tmpImage.cols, right = 0;
    int nonZeroCount = 0;

    for (int i = 0; i < tmpImage.rows; ++i) {
        for (int j = 0; j < tmpImage.cols; ++j) {
            if (tmpImage.at<uchar>(i, j) > 0) {
                bottom = min(bottom, i);
                top = max(top, i);
                left = min(left, j);
                right = max(right, j);
                nonZeroCount++;
            }
        }
    }

    // 处理空区域（无有效数字）
    if (nonZeroCount < 5 || bottom > top || left > right) {
        dstImage = Mat(28, 28, CV_8UC1, Scalar::all(0));
        return;
    }

    // 提取数字ROI
    Rect rec(left, bottom, right - left + 1, top - bottom + 1);
    Mat roi = tmpImage(rec);

    // 确保细长数字（如1）有足够背景空间，避免缩放变形
    int longLen = max(max(roi.cols, roi.rows), 12);  // 最小边长12像素
    dstImage = Mat(longLen, longLen, CV_8UC1, Scalar::all(0));
    // 精确居中放置数字
    roi.copyTo(dstImage(Rect((longLen - roi.cols) / 2, (longLen - roi.rows) / 2, roi.cols, roi.rows)));
}

// 图像转特征向量（将2D矩阵转为1D向量）
void image2Vec(const Mat& srcImage, Mat& dstImage) {
    if (srcImage.type() != CV_32FC1) {
        srcImage.convertTo(dstImage, CV_32FC1);
    }
    else {
        dstImage = srcImage;
    }
    dstImage = dstImage.reshape(1, 1);
}

// 优化calBigNumber：用countNonZero
float calBigNumber(const Mat& srcDataMat, float thresh) {
    Mat mask;
    threshold(srcDataMat, mask, thresh, 1, THRESH_BINARY);
    return (float)countNonZero(mask);
}

// 计算直方图特征
void getHistogram(const Mat& srcMat, Mat& histogramMat, int flag) {
    int size = (flag == 0) ? srcMat.rows : srcMat.cols;
    histogramMat = Mat::zeros(1, size * 2, CV_32FC1);

    vector<float> base_dist(size, 0.0);
    for (int i = 0; i < size; ++i) {
        Mat rowColMat = (flag == 0) ? srcMat.row(i) : srcMat.col(i);
        base_dist[i] = calBigNumber(rowColMat, 0);
    }

    vector<float> grad_dist(size, 0.0);
    for (int i = 1; i < size; ++i) {
        grad_dist[i] = abs(base_dist[i] - base_dist[i - 1]) * 0.5;
    }
    grad_dist[0] = grad_dist[1];

    for (int i = 0; i < size; ++i) {
        histogramMat.at<float>(0, i) = base_dist[i];
        histogramMat.at<float>(0, i + size) = grad_dist[i];
    }

    normalize(histogramMat, histogramMat, 1.0, 0.0, NORM_MINMAX);
}

// 全部像素特征
void getAllPixelsFeature(const Mat& srcDataMat, Mat& dstDataMat, Size size) {
    Mat tmpDataMat = srcDataMat.clone();
    binaryInit(tmpDataMat, size);
    image2Vec(tmpDataMat, dstDataMat);
}

// 直方图特征
void getHistogramFeature(const Mat& srcDataMat, Mat& dstDataMat, Size size) {
    Mat tmpDataMat = srcDataMat.clone();
    binaryInit(tmpDataMat, size);

    Mat rowHist, colHist;
    getHistogram(tmpDataMat, rowHist, 0);
    getHistogram(tmpDataMat, colHist, 1);

    hconcat(rowHist, colHist, dstDataMat);
}

// 混合特征（像素特征+直方图特征）
void getMixedFeature(const Mat& srcDataMat, Mat& dstDataMat, Size size) {
    // 1. 8x8像素特征（64维）
    Mat pixelFeat;
    getAllPixelsFeature(srcDataMat, pixelFeat, Size(8, 8));
    normalize(pixelFeat, pixelFeat, 1.0, 0.0, NORM_MINMAX);

    // 2. 28x28直方图特征（112维）
    Mat histFeat;
    getHistogramFeature(srcDataMat, histFeat, Size(28, 28));

    // 合并像素和直方图特征 (64 + 112 = 176维)
    hconcat(pixelFeat, histFeat, dstDataMat);
}

// 矩形排序辅助函数
bool cmp(const Rect& a, const Rect& b) {
    return a.x < b.x;
}

bool cmp2(const Rect& a, const Rect& b) {
    return a.tl().y < b.tl().y;
}

// 矩形排序
void sortRect(vector<Rect>& arr) {
    if (arr.empty()) return;

    sort(arr.begin(), arr.end(), cmp2);
    auto rowStart = arr.begin();
    auto prevRect = arr.begin();
    auto currRect = next(prevRect);

    for (; currRect != arr.end(); ++currRect, ++prevRect) {
        if (currRect->tl().y > prevRect->br().y) {
            sort(rowStart, currRect, cmp);
            rowStart = currRect;
        }
    }
    sort(rowStart, arr.end(), cmp);
}

// 新增：可视化处理函数
void showProcessingStep(const string& windowName, const Mat& image, const string& stepName, int waitTime) {
    if (!image.empty()) {
        Mat displayImg;
        if (image.type() == CV_32FC1) {
            image.convertTo(displayImg, CV_8UC1, 255.0);
        }
        else {
            displayImg = image;
        }

        // 如果是单通道图像，放大显示
        if (displayImg.channels() == 1) {
            resize(displayImg, displayImg, Size(600, 400), 0, 0, INTER_NEAREST);
        }

        imshow(windowName, displayImg);
        cout << "显示步骤: " << stepName << endl;
        waitKey(waitTime);
    }
}

// 平衡版getSegment函数 - 智能噪点过滤，保留有效数字
void getSegment(const Mat& srcImage, vector<Mat>& arr, Mat& showImage) {
    arr.clear();
    Mat tmpImage = srcImage.clone();

    cout << "\n=== 开始数字分割处理（平衡版） ===" << endl;

    // 步骤1：原始图像显示
    showProcessingStep("1-原始图像", tmpImage, "原始输入图像", 1000);

    // 步骤2：转为灰度图
    if (tmpImage.type() != CV_8UC1) {
        cvtColor(tmpImage, tmpImage, COLOR_BGR2GRAY);
    }
    showProcessingStep("2-灰度图", tmpImage, "转换为灰度图像", 1000);

    // 步骤3：适度去噪（保持数字完整性）
    Mat denoised;
    GaussianBlur(tmpImage, denoised, Size(3, 3), 0.8);  // 减弱模糊强度
    showProcessingStep("3-适度去噪", denoised, "轻度高斯模糊", 1000);

    // 步骤4：自适应阈值二值化
    Mat binary;
    int blockSize = 11;  // 恢复适中的块大小
    double C = 3;        // 适中的常数
    adaptiveThreshold(denoised, binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
        THRESH_BINARY_INV, blockSize, C);
    showProcessingStep("4-自适应二值化", binary, "自适应阈值二值化", 1000);

    // 步骤5：轻度形态学清理
    Mat cleaned;
    Mat kernel_open = getStructuringElement(MORPH_RECT, Size(2, 2));  // 减小核大小
    morphologyEx(binary, cleaned, MORPH_OPEN, kernel_open);

    Mat kernel_close = getStructuringElement(MORPH_RECT, Size(1, 1));
    morphologyEx(cleaned, cleaned, MORPH_CLOSE, kernel_close);

    showProcessingStep("5-轻度清理", cleaned, "轻度形态学处理", 1000);

    // 步骤6：智能的面积和形状过滤
    vector<vector<Point>> contours;
    Mat contour_image = cleaned.clone();
    findContours(contour_image, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 可视化所有轮廓
    Mat contour_display;
    cvtColor(cleaned, contour_display, COLOR_GRAY2BGR);
    for (size_t i = 0; i < contours.size(); i++) {
        drawContours(contour_display, contours, (int)i, Scalar(0, 255, 0), 1);
    }
    showProcessingStep("6-轮廓检测", contour_display, "检测到的所有轮廓", 1500);

    cout << "检测到 " << contours.size() << " 个轮廓" << endl;

    // 步骤7：平衡的轮廓筛选
    vector<Rect> rects;
    Mat rect_display;
    cvtColor(cleaned, rect_display, COLOR_GRAY2BGR);

    // 计算图像的基本统计信息用于动态阈值
    int img_width = tmpImage.cols;
    int img_height = tmpImage.rows;
    int img_area = img_width * img_height;

    // 更宽松的阈值设置
    int min_area = max(50, img_area / 5000);      // 降低最小面积要求
    int max_area = img_area / 10;                 // 适中的最大面积
    int min_width = 3;                            // 降低最小宽度
    int min_height = 10;                          // 降低最小高度

    cout << "筛选阈值: 最小面积=" << min_area << ", 最大面积=" << max_area
        << ", 最小尺寸=" << min_width << "x" << min_height << endl;

    for (const auto& cnt : contours) {
        Rect rect = boundingRect(cnt);

        // 计算轮廓特征
        double contour_area = contourArea(cnt);
        double rect_area = rect.area();
        double compactness = (rect_area > 0) ? contour_area / rect_area : 0;
        double aspect_ratio = (double)rect.width / rect.height;

        // 宽松的基本筛选条件
        bool valid_basic = (
            rect.area() >= min_area &&                    // 面积要求
            rect.area() <= max_area &&                    // 最大面积限制
            rect.width >= min_width &&                    // 最小宽度
            rect.height >= min_height &&                 // 最小高度
            rect.width <= img_width * 0.8 &&             // 最大宽度
            rect.height <= img_height * 0.8 &&           // 最大高度
            compactness >= 0.08                          // 降低紧实度要求
            );

        // 特殊情况处理
        bool valid_special = false;

        // 数字"1"等细长形状的特殊处理
        if (!valid_basic &&
            rect.height >= 15 &&
            rect.width >= 2 &&
            aspect_ratio >= 0.1 && aspect_ratio <= 15 &&  // 宽松的长宽比
            contour_area >= 20 &&                          // 更低的面积要求
            compactness >= 0.05) {                         // 很低的紧实度要求
            valid_special = true;
            cout << "特殊检测: 疑似细长数字 (宽度:" << rect.width << " 高度:" << rect.height << ")" << endl;
        }

        // 小数字的特殊处理（如远距离拍摄的数字）
        if (!valid_basic && !valid_special &&
            rect.width >= 8 && rect.height >= 8 &&        // 很小的数字
            rect.width <= 50 && rect.height <= 50 &&      // 限制大小范围
            compactness >= 0.1 &&                          // 适中的紧实度
            contour_area >= 30) {                          // 最低面积要求
            valid_special = true;
            cout << "特殊检测: 疑似小尺寸数字" << endl;
        }

        bool valid_rect = valid_basic || valid_special;

        if (valid_rect) {
            rects.push_back(rect);
            rectangle(rect_display, rect, Scalar(0, 255, 0), 2);
            // 显示信息
            string info = to_string(rect.width) + "x" + to_string(rect.height);
            putText(rect_display, info, Point(rect.x, rect.y - 5),
                FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 255, 0), 1);
        }
        else {
            rectangle(rect_display, rect, Scalar(0, 0, 255), 1); // 红色表示被过滤
            // 简单标记被过滤的区域
            putText(rect_display, "X", Point(rect.x, rect.y + rect.height / 2),
                FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 0, 255), 1);
        }
    }

    showProcessingStep("7-平衡筛选", rect_display, "平衡筛选后的矩形区域", 2000);
    cout << "筛选后保留 " << rects.size() << " 个区域" << endl;

    // 步骤8：智能分割过宽区域（保持原逻辑但调整参数）
    vector<Rect> separatedRects;
    Mat separated_display;
    cvtColor(cleaned, separated_display, COLOR_GRAY2BGR);

    for (const auto& rect : rects) {
        // 放宽分割条件，避免误分割
        if (rect.width > rect.height * 3.0) {
            cout << "尝试分割宽区域 (" << rect.width << "x" << rect.height << ")..." << endl;

            Mat roi = cleaned(rect);
            vector<int> projection(roi.cols, 0);

            // 计算垂直投影
            for (int x = 0; x < roi.cols; ++x) {
                for (int y = 0; y < roi.rows; ++y) {
                    if (roi.at<uchar>(y, x) > 0) {
                        projection[x]++;
                    }
                }
            }

            // 寻找分割点
            vector<int> split_points;
            split_points.push_back(0);

            int min_gap_width = max(2, roi.cols / 30);  // 降低最小间隔要求
            bool in_gap = false;
            int gap_start = 0;

            for (int x = 0; x < roi.cols; ++x) {
                if (projection[x] <= max(1, (int)(roi.rows * 0.08))) {  // 更敏感的间隔检测
                    if (!in_gap) {
                        gap_start = x;
                        in_gap = true;
                    }
                }
                else {
                    if (in_gap && (x - gap_start) >= min_gap_width) {
                        int split_x = gap_start + (x - gap_start) / 2;
                        if (split_x - split_points.back() > roi.cols / 10) {  // 降低最小分割距离
                            split_points.push_back(split_x);
                        }
                    }
                    in_gap = false;
                }
            }

            if (split_points.back() != roi.cols) {
                split_points.push_back(roi.cols);
            }

            // 创建子矩形
            if (split_points.size() > 2) {
                cout << "分割为 " << (split_points.size() - 1) << " 个子区域" << endl;
                for (size_t i = 0; i < split_points.size() - 1; ++i) {
                    int left = split_points[i];
                    int right = split_points[i + 1];

                    if (right - left >= 4) {  // 降低最小宽度要求
                        Rect sub_rect(rect.x + left, rect.y, right - left, rect.height);
                        separatedRects.push_back(sub_rect);
                        rectangle(separated_display, sub_rect, Scalar(0, 255, 255), 2);
                    }
                }
            }
            else {
                separatedRects.push_back(rect);
                rectangle(separated_display, rect, Scalar(0, 255, 0), 2);
            }
        }
        else {
            separatedRects.push_back(rect);
            rectangle(separated_display, rect, Scalar(0, 255, 0), 2);
        }
    }

    showProcessingStep("8-智能分割", separated_display, "分割后的区域", 2000);

    // 步骤9：排序
    sortRect(separatedRects);

    // 步骤10：提取并显示最终结果
    Mat final_display = srcImage.clone();
    if (final_display.channels() == 1) {
        cvtColor(final_display, final_display, COLOR_GRAY2BGR);
    }

    for (size_t i = 0; i < separatedRects.size(); ++i) {
        const auto& rec = separatedRects[i];

        Mat roi = cleaned(rec);

        // 轻度最终清理
        Mat cleaned_roi;
        Mat clean_kernel = getStructuringElement(MORPH_RECT, Size(1, 1));
        morphologyEx(roi, cleaned_roi, MORPH_OPEN, clean_kernel);

        // 归一化处理
        Mat normalizedRoi;
        normal(cleaned_roi, normalizedRoi);
        arr.push_back(normalizedRoi);

        // 绘制最终结果
        rectangle(final_display, rec, Scalar(0, 255, 0), 2);
        rectangle(showImage, rec, Scalar(0, 255, 0), 2);

        // 显示编号
        string label = to_string(i);
        putText(final_display, label, Point(rec.x, rec.y - 5),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
        putText(showImage, label, Point(rec.x, rec.y - 5),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
    }

    showProcessingStep("9-最终结果", final_display, "平衡处理后的最终结果", 3000);

    // 关闭处理窗口
    destroyWindow("1-原始图像");
    destroyWindow("2-灰度图");
    destroyWindow("3-适度去噪");
    destroyWindow("4-自适应二值化");
    destroyWindow("5-轻度清理");
    destroyWindow("6-轮廓检测");
    destroyWindow("7-平衡筛选");
    destroyWindow("8-智能分割");
    destroyWindow("9-最终结果");

    cout << "\n=== 平衡版分割处理完成 ===" << endl;
    cout << "最终检测到 " << arr.size() << " 个数字区域" << endl;
}