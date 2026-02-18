//郭雨彤编写
#include "BPNeuralNetwork.h"
#include "ImageProcessing.h"
#include "MNISTLoader.h"
#include "Constants.h"
#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <cmath>

using namespace std;
using namespace cv;

void BPNeuralNetwork::softmax(const vector<double>& z, vector<double>& result) {
    double maxVal = *max_element(z.begin(), z.end());
    double sum = 0.0;
    result.resize(z.size());
    for (size_t i = 0; i < z.size(); i++) {
        result[i] = exp(z[i] - maxVal);
        sum += result[i];
    }
    for (size_t i = 0; i < z.size(); i++) {
        result[i] /= sum;
    }
}

double BPNeuralNetwork::weighted_cross_entropy_loss(const vector<double>& y_true, const vector<double>& y_pred) {
    static const vector<double> class_weights = { 1.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    double loss = 0.0;
    for (size_t i = 0; i < y_true.size(); i++) {
        if (y_true[i] > 0) {
            loss -= class_weights[i] * y_true[i] * log(y_pred[i] + 1e-12);
        }
    }
    return loss;
}

void BPNeuralNetwork::one_hot(int label, vector<double>& result, int num_classes) {
    result.assign(num_classes, 0.0);
    result[label] = 1.0;
}

BPNeuralNetwork::BPNeuralNetwork(const vector<int>& layer_sizes, double lr)
    : layers(layer_sizes), learning_rate(lr) {
    initializeNetwork();
    initializeWeights();
    initializeAdamParameters();
    cout << "BP神经网络初始化完成！" << endl;
}

void BPNeuralNetwork::initializeNetwork() {
    activations.resize(layers.size());
    z_values.resize(layers.size());
    for (size_t i = 0; i < layers.size(); ++i) {
        activations[i].resize(layers[i], 0.0);
        z_values[i].resize(layers[i], 0.0);
    }
    weights_flat.resize(layers.size() - 1);
    for (size_t i = 0; i < weights_flat.size(); ++i) {
        int weight_count = layers[i] * layers[i + 1];
        weights_flat[i].resize(weight_count, 0.0);
    }
    biases.resize(layers.size() - 1);
    for (size_t i = 0; i < biases.size(); ++i) {
        biases[i].resize(layers[i + 1], 0.0);
    }
}

void BPNeuralNetwork::initializeAdamParameters() {
    m_weights_flat.resize(weights_flat.size());
    v_weights_flat.resize(weights_flat.size());
    for (size_t i = 0; i < weights_flat.size(); ++i) {
        m_weights_flat[i].resize(weights_flat[i].size(), 0.0);
        v_weights_flat[i].resize(weights_flat[i].size(), 0.0);
    }
    m_biases.resize(biases.size());
    v_biases.resize(biases.size());
    for (size_t i = 0; i < biases.size(); ++i) {
        m_biases[i].resize(biases[i].size(), 0.0);
        v_biases[i].resize(biases[i].size(), 0.0);
    }
}

void BPNeuralNetwork::initializeWeights() {
    random_device rd;
    mt19937 gen(rd());
    for (size_t i = 0; i < weights_flat.size(); ++i) {
        double limit = sqrt(2.0 / static_cast<double>(layers[i]));
        normal_distribution<double> dis(0.0, limit);
        for (size_t j = 0; j < weights_flat[i].size(); ++j) {
            weights_flat[i][j] = dis(gen);
        }
        for (size_t j = 0; j < biases[i].size(); ++j) {
            biases[i][j] = dis(gen) * 0.1;
        }
    }
    cout << "权重初始化完成！" << endl;
}

void BPNeuralNetwork::forward(const vector<double>& input, vector<double>& output) {
    if (input.size() != layers[0]) {
        cerr << "输入维度不匹配" << endl;
        return;
    }

    // 输入层
    activations[0] = input;

    // 隐藏层计算
    for (size_t layer = 0; layer < weights_flat.size() - 1; ++layer) {
        int M = layers[layer + 1]; // 下一层神经元数量
        int K = layers[layer];     // 当前层神经元数量

        // 计算 z = W * a_prev + b
        for (int i = 0; i < M; ++i) {
            double sum = 0.0;
            for (int j = 0; j < K; ++j) {
                sum += weights_flat[layer][i * K + j] * activations[layer][j];
            }
            z_values[layer + 1][i] = sum + biases[layer][i];
            // LeakyReLU激活函数
            activations[layer + 1][i] = z_values[layer + 1][i] > 0 ? z_values[layer + 1][i] : 0.01 * z_values[layer + 1][i];
        }
    }

    // 输出层（Softmax）
    size_t output_layer = weights_flat.size() - 1;
    int M = layers[output_layer + 1];
    int K = layers[output_layer];

    for (int i = 0; i < M; ++i) {
        double sum = 0.0;
        for (int j = 0; j < K; ++j) {
            sum += weights_flat[output_layer][i * K + j] * activations[output_layer][j];
        }
        z_values[output_layer + 1][i] = sum + biases[output_layer][i];
    }

    softmax(z_values.back(), output);
    activations.back() = output;
}

void BPNeuralNetwork::backpropagate(const vector<double>& target) {
    t++; // Adam optimizer timestep
    size_t L = layers.size() - 1; // Index of the last layer

    // 1. 计算输出层误差
    vector<double> delta_L(layers[L]);
    for (size_t i = 0; i < layers[L]; ++i) {
        delta_L[i] = activations[L][i] - target[i];
    }

    // 从输出层开始，反向传播
    vector<double> prev_delta = delta_L;

    for (int layer = L - 1; layer >= 0; --layer) {
        int M = layers[layer + 1]; // 当前层的神经元数
        int K = layers[layer];     // 前一层的神经元数

        // 2. 计算当前层的梯度
        vector<double> grad_weights_flat(M * K);
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < K; ++j) {
                grad_weights_flat[i * K + j] = prev_delta[i] * activations[layer][j];
            }
        }
        const vector<double>& grad_biases = prev_delta;

        // 3. 计算传递到前一层的误差
        vector<double> next_delta(K);
        if (layer > 0) {
            for (int j = 0; j < K; ++j) {
                double sum = 0.0;
                for (int i = 0; i < M; ++i) {
                    sum += weights_flat[layer][i * K + j] * prev_delta[i];
                }
                // 乘以激活函数的导数
                double derivative = z_values[layer][j] > 0 ? 1.0 : 0.01;
                next_delta[j] = sum * derivative;
            }
        }

        // 4. 使用Adam优化器更新权重和偏置
        for (int i = 0; i < (int)weights_flat[layer].size(); ++i) {
            double grad = grad_weights_flat[i];
            m_weights_flat[layer][i] = beta1 * m_weights_flat[layer][i] + (1 - beta1) * grad;
            v_weights_flat[layer][i] = beta2 * v_weights_flat[layer][i] + (1 - beta2) * grad * grad;
            double m_hat = m_weights_flat[layer][i] / (1 - pow(beta1, t));
            double v_hat = v_weights_flat[layer][i] / (1 - pow(beta2, t));
            weights_flat[layer][i] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
        }

        for (int i = 0; i < (int)biases[layer].size(); ++i) {
            double b_grad = grad_biases[i];
            m_biases[layer][i] = beta1 * m_biases[layer][i] + (1 - beta1) * b_grad;
            v_biases[layer][i] = beta2 * v_biases[layer][i] + (1 - beta2) * b_grad * b_grad;
            double b_m_hat = m_biases[layer][i] / (1 - pow(beta1, t));
            double b_v_hat = v_biases[layer][i] / (1 - pow(beta2, t));
            biases[layer][i] -= learning_rate * b_m_hat / (sqrt(b_v_hat) + epsilon);
        }

        // 更新误差，为下一个循环做准备
        prev_delta = next_delta;
    }
}

Mat BPNeuralNetwork::augmentImage(const Mat& src) {
    Mat dst = src.clone();
    double angle = (rand() % 20 - 10) * CV_PI / 180.0;
    Point2f center(src.cols / 2.0f, src.rows / 2.0f);
    Mat rot = getRotationMatrix2D(center, angle * 180 / CV_PI, 1.0);
    warpAffine(dst, dst, rot, src.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(0));
    int dx = rand() % 5 - 2;
    int dy = rand() % 5 - 2;
    Mat trans = (Mat_<float>(2, 3) << 1, 0, dx, 0, 1, dy);
    warpAffine(dst, dst, trans, src.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(0));
    return dst;
}

void BPNeuralNetwork::extractFeaturesOptimized(const Mat& mnist_image, vector<double>& feature_vec, bool augment) {
    static Mat pixel_feat, hist_feat, feature_mat;
    static Mat tmpMat, rowHist, colHist, ratioMat;
    const Mat* src_img = &mnist_image;
    Mat aug_img;
    if (augment) {
        aug_img = augmentImage(mnist_image);
        src_img = &aug_img;
    }
    tmpMat = src_img->clone();
    binaryInit(tmpMat, Size(8, 8));
    image2Vec(tmpMat, pixel_feat);
    normalize(pixel_feat, pixel_feat, 1.0, 0.0, NORM_MINMAX);
    tmpMat = src_img->clone();
    binaryInit(tmpMat, Size(28, 28));
    getHistogram(tmpMat, rowHist, 0);
    getHistogram(tmpMat, colHist, 1);
    hconcat(rowHist, colHist, hist_feat);
    hconcat(pixel_feat, hist_feat, feature_mat);
    feature_vec.resize(feature_mat.cols);
    for (int k = 0; k < feature_mat.cols; ++k) {
        feature_vec[k] = static_cast<double>(feature_mat.at<float>(0, k));
    }
}

void BPNeuralNetwork::readMNISTWithFeaturesOptimized(const string& img_file, const string& label_file,
    vector<vector<double>>& features, vector<int>& labels,
    int max_count, bool augment) {
    ifstream img_stream(img_file, ios::binary);
    if (!img_stream.is_open()) {
        cerr << "无法打开MNIST图像文件: " << img_file << endl;
        return;
    }
    uint32_t magic_num = 0, num_images = 0, rows = 0, cols = 0;
    img_stream.read(reinterpret_cast<char*>(&magic_num), 4);
    img_stream.read(reinterpret_cast<char*>(&num_images), 4);
    img_stream.read(reinterpret_cast<char*>(&rows), 4);
    img_stream.read(reinterpret_cast<char*>(&cols), 4);
    magic_num = ReverseInt(magic_num);
    num_images = ReverseInt(num_images);
    rows = ReverseInt(rows);
    cols = ReverseInt(cols);
    if (max_count > 0 && max_count < (int)num_images) {
        num_images = max_count;
    }
    ifstream label_stream(label_file, ios::binary);
    if (!label_stream.is_open()) {
        cerr << "无法打开MNIST标签文件: " << label_file << endl;
        img_stream.close();
        return;
    }
    uint32_t label_magic = 0, num_labels = 0;
    label_stream.read(reinterpret_cast<char*>(&label_magic), 4);
    label_stream.read(reinterpret_cast<char*>(&num_labels), 4);
    label_magic = ReverseInt(label_magic);
    num_labels = ReverseInt(num_labels);
    cout << "读取MNIST数据: " << num_images << "个样本，尺寸: " << rows << "x" << cols << endl;
    int total_samples = augment ? num_images * 2 : num_images;
    features.reserve(total_samples);
    labels.reserve(total_samples);
    for (uint32_t i = 0; i < num_images; ++i) {
        Mat mnist_mat(rows, cols, CV_8UC1);
        img_stream.read(reinterpret_cast<char*>(mnist_mat.data), rows * cols);
        unsigned char label_byte;
        label_stream.read(reinterpret_cast<char*>(&label_byte), 1);
        int label = static_cast<int>(label_byte);
        vector<double> feature_vec;
        extractFeaturesOptimized(mnist_mat, feature_vec, false);
        features.push_back(move(feature_vec));
        labels.push_back(label);
        if (augment) {
            vector<double> aug_feature_vec;
            extractFeaturesOptimized(mnist_mat, aug_feature_vec, true);
            features.push_back(move(aug_feature_vec));
            labels.push_back(label);
        }
    }
    img_stream.close();
    label_stream.close();
    cout << "特征提取完成，共" << features.size() << "个样本" << endl;
}

void BPNeuralNetwork::train(int train_count, int test_count, int epochs, int batch_size) {
    cout << "=== 开始训练前的特征预处理检查 ===" << endl;
    string train_feat_path = "mnist_train_feats.bin";
    string train_label_path = "mnist_train_labels.bin";
    string test_feat_path = "mnist_test_feats.bin";
    string test_label_path = "mnist_test_labels.bin";
    cout << "开始批量预处理训练集并保存..." << endl;
    preprocessAndSaveMNISTFeatures(
        g_TrainImageFileName,
        g_TrainLabelFileName,
        train_feat_path,
        train_label_path,
        train_count,
        false
    );
    cout << "开始批量预处理测试集并保存..." << endl;
    preprocessAndSaveMNISTFeatures(
        g_TestImageFileName,
        g_TestLabelFileName,
        test_feat_path,
        test_label_path,
        test_count,
        false
    );
    cout << "\n开始加载预处理特征..." << endl;
    vector<vector<double>> train_feats;
    vector<int> train_labels;
    vector<vector<double>> test_feats;
    vector<int> test_labels;
    loadPreprocessedFeatures(train_feat_path, train_label_path, train_feats, train_labels);
    loadPreprocessedFeatures(test_feat_path, test_label_path, test_feats, test_labels);
    if (train_feats.empty() || train_labels.empty() || train_feats.size() != train_labels.size()) {
        cerr << "训练集特征加载失败！请检查文件路径或重新运行程序" << endl;
        return;
    }
    if (test_feats.empty() || test_labels.empty() || test_feats.size() != test_labels.size()) {
        cerr << "测试集特征加载失败！请检查文件路径或重新运行程序" << endl;
        return;
    }
    int sample_count = train_feats.size();
    cout << "开始训练：样本数=" << sample_count << ", 轮次=" << epochs << ", 批次大小=" << batch_size << endl;
    vector<double> lr_schedule = { 0.001, 0.001, 0.0005, 0.0005, 0.0001 };
    int epochs_per_stage = max(1, epochs / (int)lr_schedule.size());
    vector<double> output(10), target(10);
    for (int epoch = 0; epoch < epochs; ++epoch) {
        int stage = min((int)lr_schedule.size() - 1, epoch / epochs_per_stage);
        learning_rate = lr_schedule[stage];
        double total_loss = 0.0;
        int correct_count = 0;
        vector<int> indices(sample_count);
        iota(indices.begin(), indices.end(), 0);
        shuffle(indices.begin(), indices.end(), mt19937(random_device()()));
        for (int batch_start = 0; batch_start < sample_count; batch_start += batch_size) {
            int batch_end = min(batch_start + batch_size, sample_count);
            for (int i = batch_start; i < batch_end; ++i) {
                int idx = indices[i];
                const vector<double>& feat = train_feats[idx];
                int label = train_labels[idx];
                forward(feat, output);
                one_hot(label, target);
                total_loss += weighted_cross_entropy_loss(target, output);
                if (predict(feat) == label) {
                    correct_count++;
                }
                backpropagate(target);
            }
        }
        double avg_loss = total_loss / sample_count;
        double accuracy = static_cast<double>(correct_count) / sample_count * 100.0;
        cout << "轮次 " << setw(2) << (epoch + 1) << "/" << epochs
            << " | 学习率: " << fixed << setprecision(4) << learning_rate
            << " | 平均损失: " << fixed << setprecision(6) << avg_loss
            << " | 准确率: " << setprecision(2) << accuracy << "%" << endl;
    }
    cout << "训练完成！开始测试..." << endl;
    testAccuracy(test_feats, test_labels);
}

int BPNeuralNetwork::predict(const vector<double>& input) {
    static vector<double> output(10);
    forward(input, output);
    if (output.empty()) return -1;
    return max_element(output.begin(), output.end()) - output.begin();
}

double BPNeuralNetwork::testAccuracy(const vector<vector<double>>& test_feats, const vector<int>& test_labels) {
    if (test_feats.empty() || test_labels.empty() || test_feats.size() != test_labels.size()) {
        cerr << "测试数据无效！" << endl;
        return 0.0;
    }
    int correct_count = 0;
    int test_count = test_feats.size();
    cout << "开始测试：测试样本数=" << test_count << endl;
    for (int i = 0; i < test_count; ++i) {
        if (predict(test_feats[i]) == test_labels[i]) {
            correct_count++;
        }
    }
    double accuracy = static_cast<double>(correct_count) / test_count * 100.0;
    cout << "测试完成：正确数=" << correct_count << ", 准确率=" << fixed << setprecision(2) << accuracy << "%" << endl;
    return accuracy;
}

void BPNeuralNetwork::preprocessAndSaveMNISTFeatures(const string& img_file, const string& label_file,
    const string& feat_save_path, const string& label_save_path,
    int max_count, bool augment) {
    vector<vector<double>> all_feats;
    vector<int> all_labels;
    readMNISTWithFeaturesOptimized(img_file, label_file, all_feats, all_labels, max_count, augment);
    ofstream feat_out(feat_save_path, ios::binary);
    int feat_dim = all_feats.empty() ? 0 : all_feats[0].size();
    int sample_count = all_feats.size();
    feat_out.write((char*)&sample_count, sizeof(int));
    feat_out.write((char*)&feat_dim, sizeof(int));
    for (const auto& feat : all_feats) {
        feat_out.write((char*)feat.data(), sizeof(double) * feat_dim);
    }
    feat_out.close();
    ofstream label_out(label_save_path, ios::binary);
    label_out.write((char*)&sample_count, sizeof(int));
    label_out.write((char*)all_labels.data(), sizeof(int) * sample_count);
    label_out.close();
    cout << "特征预处理完成，保存至: " << feat_save_path << " (样本数: " << sample_count << ")" << endl;
}

void BPNeuralNetwork::loadPreprocessedFeatures(const string& feat_path, const string& label_path,
    vector<vector<double>>& feats, vector<int>& labels) {
    ifstream feat_in(feat_path, ios::binary);
    ifstream label_in(label_path, ios::binary);
    int sample_count, feat_dim;
    feat_in.read((char*)&sample_count, sizeof(int));
    feat_in.read((char*)&feat_dim, sizeof(int));
    label_in.read((char*)&sample_count, sizeof(int));
    feats.resize(sample_count, vector<double>(feat_dim));
    labels.resize(sample_count);
    for (int i = 0; i < sample_count; ++i) {
        feat_in.read((char*)feats[i].data(), sizeof(double) * feat_dim);
        label_in.read((char*)&labels[i], sizeof(int));
    }
    feat_in.close();
    label_in.close();
}

void BPNeuralNetwork::saveModel(const string& filename) {
    ofstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "无法保存模型文件: " << filename << endl;
        return;
    }
    size_t layer_count = layers.size();
    file.write(reinterpret_cast<const char*>(&layer_count), sizeof(layer_count));
    file.write(reinterpret_cast<const char*>(layers.data()), sizeof(int) * layer_count);
    for (const auto& layer_weights : weights_flat) {
        size_t weight_size = layer_weights.size();
        file.write(reinterpret_cast<const char*>(&weight_size), sizeof(weight_size));
        file.write(reinterpret_cast<const char*>(layer_weights.data()), sizeof(double) * weight_size);
    }
    for (const auto& layer_biases : biases) {
        size_t bias_size = layer_biases.size();
        file.write(reinterpret_cast<const char*>(&bias_size), sizeof(bias_size));
        file.write(reinterpret_cast<const char*>(layer_biases.data()), sizeof(double) * bias_size);
    }
    file.close();
    cout << "模型已保存至: " << filename << endl;
}

bool BPNeuralNetwork::loadModel(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "无法加载模型文件: " << filename << endl;
        return false;
    }
    size_t layer_count = 0;
    file.read(reinterpret_cast<char*>(&layer_count), sizeof(layer_count));
    layers.resize(layer_count);
    file.read(reinterpret_cast<char*>(layers.data()), sizeof(int) * layer_count);
    initializeNetwork();
    initializeAdamParameters();
    for (auto& layer_weights : weights_flat) {
        size_t weight_size;
        file.read(reinterpret_cast<char*>(&weight_size), sizeof(weight_size));
        layer_weights.resize(weight_size);
        file.read(reinterpret_cast<char*>(layer_weights.data()), sizeof(double) * weight_size);
    }
    for (auto& layer_biases : biases) {
        size_t bias_size;
        file.read(reinterpret_cast<char*>(&bias_size), sizeof(bias_size));
        layer_biases.resize(bias_size);
        file.read(reinterpret_cast<char*>(layer_biases.data()), sizeof(double) * bias_size);
    }
    file.close();
    cout << "模型加载成功: " << filename << endl;
    return true;
}

void BPNeuralNetwork::printNetworkInfo() {
    cout << "\n=== BP神经网络信息 ===" << endl;
    cout << "网络结构: ";
    for (size_t i = 0; i < layers.size(); ++i) {
        cout << layers[i];
        if (i < layers.size() - 1) cout << " -> ";
    }
    cout << endl;
    int total_params = 0;
    for (size_t i = 0; i < weights_flat.size(); ++i) {
        int layer_weights = layers[i] * layers[i + 1];
        int layer_biases = layers[i + 1];
        int layer_params = layer_weights + layer_biases;
        total_params += layer_params;
        cout << "第" << (i + 1) << "层参数: " << layer_weights << "(权重) + " << layer_biases << "(偏置) = " << layer_params << endl;
    }
    cout << "总参数数量: " << total_params << endl;
    cout << "学习率: " << learning_rate << endl;
    cout << "优化器: Adam (beta1=" << beta1 << ", beta2=" << beta2 << ")" << endl;
    cout << "=====================" << endl;
}