//郭雨彤编写
#include <iostream>
#include <locale>
#include "Constants.h"
#include "BPNeuralNetwork.h"
#include "Recognition.h"

using namespace std;

int main() {
    setlocale(LC_ALL, "zh_CN.GB2312");

    Ptr<BPNeuralNetwork> bpNet = makePtr<BPNeuralNetwork>(vector<int>{TOTAL_FEATURE_DIM, 256, 128, 64, 10}, 0.001);

    while (true) {
        cout << "\n===== 手写数字识别系统 =====" << endl;
        cout << "1. 训练新模型" << endl;
        cout << "2. 加载已有模型" << endl;
        cout << "3. 鼠标手写识别" << endl;
        cout << "4. 图片文件识别" << endl;
        cout << "5. 退出系统" << endl;
        cout << "请选择功能 (1-5): ";
        int choice;
        cin >> choice;
        switch (choice) {
        case 1: {
            int train_num, test_num, epochs, batch_size;
            cout << "请输入训练样本数量 (建议1000-60000): "; cin >> train_num;
            cout << "请输入测试样本数量 (建议1000-10000): "; cin >> test_num;
            cout << "请输入训练轮次 (建议30): "; cin >> epochs;
            cout << "请输入批次大小 (建议64): "; cin >> batch_size;
            bpNet->printNetworkInfo();
            bpNet->train(train_num, test_num, epochs, batch_size);
            char save_choice;
            cout << "是否保存训练好的模型? (y/n): "; cin >> save_choice;
            if (save_choice == 'y' || save_choice == 'Y') {
                bpNet->saveModel(BP_MODEL_PATH);
            }
            break;
        }
        case 2: {
            if (bpNet->loadModel(BP_MODEL_PATH)) {
                bpNet->printNetworkInfo();
            }
            break;
        }
        case 3: {
            handWritingRecognize(bpNet);
            break;
        }
        case 4: {
            imageRecognize(bpNet);
            break;
        }
        case 5: {
            cout << "感谢使用，再见！" << endl;
            return 0;
        }
        default:
            cout << "无效选择，请输入1-5之间的数字！" << endl;
        }
    }
    return 0;
}