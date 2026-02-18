//杨亚童编写
#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <string>

using namespace std;

// 全局常量定义
extern const string WINDOW_NAME;
extern const string g_TrainImageFileName;
extern const string g_TrainLabelFileName;
extern const string g_TestImageFileName;
extern const string g_TestLabelFileName;
extern const string BP_MODEL_PATH;

// 特征维度常量
extern const int PIXEL_FEATURE_DIM;
extern const int HIST_FEATURE_DIM;
extern const int ASPECT_RATIO_DIM;
extern const int TOTAL_FEATURE_DIM;

#endif 