//杨亚童编写
#include "Constants.h"

using namespace std;

// 全局常量实现
const string WINDOW_NAME = "Handwriting Recognition";
const string g_TrainImageFileName = "train-images.idx3-ubyte";
const string g_TrainLabelFileName = "train-labels.idx1-ubyte";
const string g_TestImageFileName = "t10k-images.idx3-ubyte";
const string g_TestLabelFileName = "t10k-labels.idx1-ubyte";
const string BP_MODEL_PATH = "mnist_bp_model.bin";

// 特征维度常量实现
const int PIXEL_FEATURE_DIM = 64;    // 8x8像素特征
const int HIST_FEATURE_DIM = 112;    // 28x28行列直方图特征
const int ASPECT_RATIO_DIM = 0;      // 不再使用独立的长宽比特征
const int TOTAL_FEATURE_DIM = PIXEL_FEATURE_DIM + HIST_FEATURE_DIM + ASPECT_RATIO_DIM;