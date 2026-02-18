//郭雨彤编写
#ifndef BP_NEURAL_NETWORK_H
#define BP_NEURAL_NETWORK_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

// BP神经网络类
class BPNeuralNetwork {
private:
    vector<int> layers;
    vector<vector<double>> weights_flat;
    vector<vector<double>> biases;
    vector<vector<double>> activations;
    vector<vector<double>> z_values;
    double learning_rate;

    vector<vector<double>> m_weights_flat, v_weights_flat;
    vector<vector<double>> m_biases, v_biases;
    double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
    int t = 0;

    void softmax(const vector<double>& z, vector<double>& result);
    double weighted_cross_entropy_loss(const vector<double>& y_true, const vector<double>& y_pred);
    void one_hot(int label, vector<double>& result, int num_classes = 10);

public:
    BPNeuralNetwork(const vector<int>& layer_sizes, double lr = 0.001);
    void initializeNetwork();
    void initializeAdamParameters();
    void initializeWeights();

    void forward(const vector<double>& input, vector<double>& output);
    void backpropagate(const vector<double>& target);

    Mat augmentImage(const Mat& src);
    void extractFeaturesOptimized(const Mat& mnist_image, vector<double>& feature_vec, bool augment = false);
    void readMNISTWithFeaturesOptimized(const string& img_file, const string& label_file,
        vector<vector<double>>& features, vector<int>& labels, int max_count = -1, bool augment = false);

    void train(int train_count = 60000, int test_count = 10000, int epochs = 50, int batch_size = 64);
    int predict(const vector<double>& input);
    double testAccuracy(const vector<vector<double>>& test_feats, const vector<int>& test_labels);

    void preprocessAndSaveMNISTFeatures(const string& img_file, const string& label_file,
        const string& feat_save_path, const string& label_save_path,
        int max_count = -1, bool augment = false);
    void loadPreprocessedFeatures(const string& feat_path, const string& label_path,
        vector<vector<double>>& feats, vector<int>& labels);

    void saveModel(const string& filename);
    bool loadModel(const string& filename);
    void printNetworkInfo();
};

#endif 