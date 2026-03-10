#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <iomanip>
#include <stdexcept>

struct ResultRow {
    int n_layers;
    int n_units;
    double accuracy;
};

struct IrisData {
    torch::Tensor X;
    torch::Tensor y;
};

IrisData load_iris_csv(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    std::getline(file, line); // skip header

    std::vector<float> features;
    std::vector<int64_t> labels;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;

        while (std::getline(ss, cell, ',')) {
            row.push_back(cell);
        }

        if (row.size() != 5) {
            throw std::runtime_error("Invalid row in CSV: " + line);
        }

        features.push_back(std::stof(row[0]));
        features.push_back(std::stof(row[1]));
        features.push_back(std::stof(row[2]));
        features.push_back(std::stof(row[3]));

        std::string label = row[4];

// optional: trim spaces
label.erase(0, label.find_first_not_of(" \t\r\n"));
label.erase(label.find_last_not_of(" \t\r\n") + 1);

if (label == "Setosa" || label == "setosa" || label == "Iris-setosa") {
    labels.push_back(0);
} else if (label == "Versicolor" || label == "versicolor" || label == "Iris-versicolor") {
    labels.push_back(1);
} else if (label == "Virginica" || label == "virginica" || label == "Iris-virginica") {
    labels.push_back(2);
} else {
    throw std::runtime_error("Unknown label: " + label);
}
    }

    const int64_t n = static_cast<int64_t>(labels.size());

    auto X = torch::from_blob(features.data(), {n, 4}, torch::kFloat32).clone();
    auto y = torch::from_blob(labels.data(), {n}, torch::kInt64).clone();

    return {X, y};
}

void train_test_split_tensors(
    const torch::Tensor& X,
    const torch::Tensor& y,
    double train_size,
    int seed,
    torch::Tensor& X_train,
    torch::Tensor& X_test,
    torch::Tensor& y_train,
    torch::Tensor& y_test
) {
    const int64_t n = X.size(0);
    std::vector<int64_t> indices(n);
    for (int64_t i = 0; i < n; ++i) indices[i] = i;

    std::mt19937 gen(seed);
    std::shuffle(indices.begin(), indices.end(), gen);

    int64_t n_train = static_cast<int64_t>(n * train_size);

    std::vector<int64_t> train_idx(indices.begin(), indices.begin() + n_train);
    std::vector<int64_t> test_idx(indices.begin() + n_train, indices.end());

    auto train_idx_tensor = torch::tensor(train_idx, torch::kInt64);
    auto test_idx_tensor = torch::tensor(test_idx, torch::kInt64);

    X_train = X.index_select(0, train_idx_tensor);
    X_test = X.index_select(0, test_idx_tensor);
    y_train = y.index_select(0, train_idx_tensor);
    y_test = y.index_select(0, test_idx_tensor);
}

struct Net2Impl : torch::nn::Module {
    int n_layers;
    torch::nn::Linear input{nullptr};
    torch::nn::Linear output{nullptr};
    std::vector<torch::nn::Linear> hidden_layers;

    Net2Impl(int n_units, int n_layers_) : n_layers(n_layers_) {
        input = register_module("input", torch::nn::Linear(4, n_units));

        for (int i = 0; i < n_layers; ++i) {
            auto layer = torch::nn::Linear(n_units, n_units);
            hidden_layers.push_back(register_module("hidden_" + std::to_string(i), layer));
        }

        output = register_module("output", torch::nn::Linear(n_units, 3));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = input->forward(x);

        for (int i = 0; i < n_layers; ++i) {
            x = torch::relu(hidden_layers[i]->forward(x));
        }

        x = output->forward(x);
        return x;
    }
};
TORCH_MODULE(Net2);

std::pair<double, double> train_model(
    const std::vector<std::pair<torch::Tensor, torch::Tensor>>& train_loader,
    const std::vector<std::pair<torch::Tensor, torch::Tensor>>& test_loader,
    Net2& model,
    double lr = 0.01,
    int num_epochs = 200
) {
    std::vector<double> train_accuracies, test_accuracies, losses;

    torch::nn::CrossEntropyLoss loss_function;
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(lr));

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        torch::Tensor pred_labels, y_batch;

        for (const auto& batch : train_loader) {
            auto X = batch.first;
            auto y = batch.second;

            auto preds = model->forward(X);
            pred_labels = torch::argmax(preds, 1);
            y_batch = y;

            auto loss = loss_function(preds, y);
            losses.push_back(loss.item<double>());

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }

        double train_acc = 100.0 *
            pred_labels.eq(y_batch).to(torch::kFloat32).mean().item<double>();
        train_accuracies.push_back(train_acc);

        auto X_test = test_loader[0].first;
        auto y_test = test_loader[0].second;
        auto test_preds = model->forward(X_test);
        auto test_pred_labels = torch::argmax(test_preds, 1);

        double test_acc = 100.0 *
            test_pred_labels.eq(y_test).to(torch::kFloat32).mean().item<double>();
        test_accuracies.push_back(test_acc);
    }

    return {train_accuracies.back(), test_accuracies.back()};
}

std::vector<std::pair<torch::Tensor, torch::Tensor>> make_train_loader(
    const torch::Tensor& X_train,
    const torch::Tensor& y_train,
    int64_t batch_size,
    int seed
) {
    const int64_t n = X_train.size(0);

    std::vector<int64_t> indices(n);
    for (int64_t i = 0; i < n; ++i) indices[i] = i;

    std::mt19937 gen(seed);
    std::shuffle(indices.begin(), indices.end(), gen);

    std::vector<std::pair<torch::Tensor, torch::Tensor>> loader;

    for (int64_t start = 0; start < n; start += batch_size) {
        int64_t end = std::min(start + batch_size, n);

        std::vector<int64_t> batch_idx(indices.begin() + start, indices.begin() + end);
        auto idx_tensor = torch::tensor(batch_idx, torch::kInt64);

        auto X_batch = X_train.index_select(0, idx_tensor);
        auto y_batch = y_train.index_select(0, idx_tensor);

        loader.push_back({X_batch, y_batch});
    }

    return loader;
}

std::vector<std::pair<torch::Tensor, torch::Tensor>> make_test_loader(
    const torch::Tensor& X_test,
    const torch::Tensor& y_test
) {
    return { {X_test, y_test} };
}

void print_tensor_shape(const torch::Tensor& X, const torch::Tensor& y) {
    std::cout << "[" << X.size(0) << ", " << X.size(1) << "] "
              << "[" << y.size(0) << "]" << std::endl;
}

int main() {
    try {
        torch::manual_seed(42);

        auto iris = load_iris_csv("iris.csv");

        std::cout << "First 5 rows of iris.csv:" << std::endl;
        for (int i = 0; i < std::min<int64_t>(5, iris.X.size(0)); ++i) {
            std::cout
                << iris.X[i][0].item<float>() << ", "
                << iris.X[i][1].item<float>() << ", "
                << iris.X[i][2].item<float>() << ", "
                << iris.X[i][3].item<float>() << ", "
                << iris.y[i].item<int64_t>() << std::endl;
        }

        std::cout << "\nX shape: [" << iris.X.size(0) << ", " << iris.X.size(1) << "]"
                  << " y shape: [" << iris.y.size(0) << "]" << std::endl;

        torch::Tensor X_train, X_test, y_train, y_test;
        train_test_split_tensors(
            iris.X, iris.y,
            0.8, 42,
            X_train, X_test, y_train, y_test
        );

        auto train_loader = make_train_loader(X_train, y_train, 12, 42);
        auto test_loader = make_test_loader(X_test, y_test);

        std::cout << "\nTraining data batches:" << std::endl;
        for (const auto& batch : train_loader) {
            print_tensor_shape(batch.first, batch.second);
        }

        std::cout << "\nTest data batches:" << std::endl;
        for (const auto& batch : test_loader) {
            print_tensor_shape(batch.first, batch.second);
        }

        std::vector<int> n_layers = {1, 2, 3, 4};
        std::vector<int> n_units = {8, 16, 24, 32, 40, 48, 56, 64};

        std::vector<ResultRow> train_accuracies;
        std::vector<ResultRow> test_accuracies;

        for (int units : n_units) {
            for (int layers : n_layers) {
                Net2 model(units, layers);

                auto result = train_model(train_loader, test_loader, model);
                double train_acc = result.first;
                double test_acc = result.second;

                train_accuracies.push_back({layers, units, train_acc});
                test_accuracies.push_back({layers, units, test_acc});
            }
        }

        std::sort(test_accuracies.begin(), test_accuracies.end(),
            [](const ResultRow& a, const ResultRow& b) {
                if (a.n_layers != b.n_layers) return a.n_layers < b.n_layers;
                return a.n_units < b.n_units;
            });

        std::cout << "\nFirst 5 rows of test_accuracies:" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        for (size_t i = 0; i < std::min<size_t>(5, test_accuracies.size()); ++i) {
            std::cout
                << "n_layers=" << test_accuracies[i].n_layers
                << ", n_units=" << test_accuracies[i].n_units
                << ", accuracy=" << test_accuracies[i].accuracy
                << std::endl;
        }

        double max_acc = -1.0;
        for (const auto& row : test_accuracies) {
            if (row.accuracy > max_acc) {
                max_acc = row.accuracy;
            }
        }

        std::cout << "\nBest test accuracy rows:" << std::endl;
        for (const auto& row : test_accuracies) {
            if (row.accuracy == max_acc) {
                std::cout
                    << "n_layers=" << row.n_layers
                    << ", n_units=" << row.n_units
                    << ", accuracy=" << row.accuracy
                    << std::endl;
            }
        }
    }
    catch (const c10::Error& e) {
        std::cerr << "LibTorch error: " << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "Standard error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}