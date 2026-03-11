#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <iomanip>
using namespace std;
using namespace torch;
//namespace nn = torch::nn;
//#include <stdexcept>

struct ResultRow 
{
    int n_layers, n_units;
    float accuracy;
};

struct IrisData 
{
    Tensor X,y;
};

IrisData load_iris_csv(string filename)  
{
    ifstream file(filename);
    string line;
    getline(file, line); //skip header

    vector<float> features;
    vector<int64_t> labels;

    while (getline(file, line)) 
    {
        if (line.empty()) continue;
        
        stringstream ss(line);
        string cell;
        vector<string> row;

        while (getline(ss, cell, ',')) 
        {
            row.push_back(cell);
        }

     /*   if (row.size() != 5) {
            throw std::runtime_error("Invalid row in CSV: " + line);
        } */

        features.push_back(stof(row[0]));
        features.push_back(stof(row[1]));
        features.push_back(stof(row[2]));
        features.push_back(stof(row[3]));

        string label = row[4];
       // label.erase(remove(label.begin(), label.end(), '"'), label.end());

        if (label == "Setosa") 
        {
            labels.push_back(0);
        } 
        else if (label == "Versicolor") 
        {
            labels.push_back(1);
        } 
        else  
        {
            labels.push_back(2);
        } 
        
    }

    const int n = static_cast<int>(labels.size()); //size will give unsigned int, we need signed int, so converts

    Tensor X = from_blob(features.data(), {n, 4}, kFloat32).clone(); //blob=binary large object
    Tensor y = from_blob(labels.data(), {n}, kInt64).clone();

    return {X, y};
}

void train_test_split_tensors(
    Tensor& X,
    Tensor& y,
    float train_size,
    int seed,
    Tensor& X_train,
    Tensor& X_test,
    Tensor& y_train,
    Tensor& y_test
)
{
    manual_seed(seed);   

    int n = X.size(0);
    int n_train = static_cast<int>(n * train_size);

    Tensor indices = randperm(n); //randperm returns tensor 
    Tensor train_idx = indices.slice(0, 0, n_train);
    Tensor test_idx  = indices.slice(0, n_train);

    X_train = X.index_select(0, train_idx).clone();
    X_test  = X.index_select(0, test_idx).clone();
    y_train = y.index_select(0, train_idx).clone();
    y_test  = y.index_select(0, test_idx).clone();
}

struct Net2Impl : nn::Module  //inherit from nn::Module 
{
    int n_layers;
    nn::Linear input{nullptr}; 
    nn::Linear output{nullptr};
    vector<nn::Linear> hidden_layers; 

    Net2Impl(int n_units,int n_layers_) 
     {
        n_layers = n_layers_;
        input = register_module("input", nn::Linear(4, n_units));

        for (int i = 0; i < n_layers; ++i) 
        {
            hidden_layers.push_back(register_module("hidden_" + to_string(i), nn::Linear(n_units, n_units)));
        }
        output = register_module("output", nn::Linear(n_units, 3));
    }

    Tensor forward(Tensor x) 
    {
        x = relu(input->forward(x));

        for (int i = 0; i < n_layers; ++i) 
        {
            x = relu(hidden_layers[i]->forward(x));
        }
        x = output->forward(x);
        return x;
    }
};
TORCH_MODULE(Net2); //Wrapper

vector<pair<Tensor,Tensor>> make_train_loader
(
    const Tensor& X_train,
    const Tensor& y_train,
    int batch_size,
    int seed
) 
{
    manual_seed(seed);
    int n = X_train.size(0);
    Tensor indices= randperm(n);
    vector<pair<Tensor,Tensor>> loader;

    for (int start = 0; start < n; start += batch_size)
    {
        int end = min(start + batch_size, n);
        Tensor batch_idx = indices.slice(0, start, end);
        Tensor X_batch = X_train.index_select(0, batch_idx);
        Tensor y_batch = y_train.index_select(0, batch_idx);
        loader.push_back({X_batch, y_batch});
    }
    return loader;
}

vector<pair<Tensor,Tensor>> make_test_loader
(
  const Tensor& X_test,
  const Tensor& y_test
) 
{
    return {{X_test, y_test}}; //because we are initializing a vector and a pair 
}

void print_tensor_shape(const Tensor& X,const Tensor& y) 
{
    cout<<"["<<X.size(0)<<", "<< X.size(1)<<"]"<<"["<<y.size(0)<<"]"<<endl;
}

pair<float,float> train_model
(
    const vector<pair<Tensor,Tensor>>& train_loader,
    const vector<pair<Tensor,Tensor>>& test_loader,
    Net2& model,
    float lr=0.01,
    int num_epochs=200
) 
{
    vector<float> train_accuracies,test_accuracies;
    vector<Tensor> losses;

    nn::CrossEntropyLoss loss_function;
    optim::Adam optimizer(model->parameters(), optim::AdamOptions(lr));

    for(int epoch= 0; epoch<num_epochs; ++epoch) 
    {
        int correct_preds=0, total_preds= 0;

        for(const auto& [X, y] : train_loader) 
        {
            auto preds = model->forward(X);
            auto pred_labels = argmax(preds, 1);
            auto loss = loss_function(preds, y);

            losses.push_back(loss.detach()); //detach avoids tracking

            optimizer.zero_grad(); //clear prev gradients
            loss.backward(); //compute gradients
            optimizer.step(); //update parameters

            correct_preds += pred_labels.eq(y).sum().item<int>(); //eq(y) compares elementwise and returns tensor with 0 or 1 
            total_preds += y.size(0);
        }

        train_accuracies.push_back(100.0 * correct_preds / total_preds);

        auto X = test_loader[0].first;
        auto y = test_loader[0].second;
        auto pred_labels = argmax(model->forward(X), 1);

        test_accuracies.push_back(
            100.0 * pred_labels.eq(y).to(kFloat32).mean().item<float>()
        );
    }

    return {train_accuracies.back(), test_accuracies.back()};
}

int main() 
{
    cout<<"Welcome"<<endl;

        torch::manual_seed(42);
        IrisData iris = load_iris_csv("iris.csv");  //returns iris.X and iris.y as tensors
        cout<<"First 5 rows of iris.csv:"<<endl;

        for (int i = 0; i < 5; ++i) 
        {
            cout
                <<iris.X[i][0].item<float>()<< ", "
                <<iris.X[i][1].item<float>()<< ", "
                <<iris.X[i][2].item<float>()<< ", "
                <<iris.X[i][3].item<float>()<< ", "
                <<iris.y[i].item<int>()<<endl;
        }

        cout<<"\n X shape: ["<<iris.X.size(0)<< ", "<<iris.X.size(1)<<"]"
                  <<" y shape: [" << iris.y.size(0)<<"]"<<endl;

        Tensor X_train, X_test, y_train, y_test;

        train_test_split_tensors(
            iris.X, iris.y,
            0.8, 42,
            X_train, X_test, y_train, y_test
        );

        auto train_loader = make_train_loader(X_train, y_train, 12, 42);
        auto test_loader = make_test_loader(X_test, y_test);

        cout<< "\n Training data batches:"<<endl;
        for(const auto& batch : train_loader) 
        {
            print_tensor_shape(batch.first, batch.second);
        }

        cout <<"\n Test data batches:"<<endl;
        for(const auto& batch : test_loader)
         {
            print_tensor_shape(batch.first,batch.second);
        }

        vector<int> n_layers = {1, 2, 3, 4};
        vector<int> n_units = {8, 16, 24, 32, 40, 48, 56, 64};

        vector<ResultRow> train_accuracies;
        vector<ResultRow> test_accuracies;

        for (int units: n_units) 
        {
            for (int layers: n_layers) 
            {
                manual_seed(42);
                Net2 model(units, layers);

                auto result = train_model(train_loader, test_loader, model);
                float train_acc= result.first;
                float test_acc = result.second;

                train_accuracies.push_back({layers, units, train_acc});
                test_accuracies.push_back({layers, units, test_acc});
            }
        }

      /*  sort(test_accuracies.begin(), test_accuracies.end(),
            [](const ResultRow& a, const ResultRow& b) {
                if (a.n_layers != b.n_layers) return a.n_layers < b.n_layers;
                return a.n_units < b.n_units;
            }); */

        cout<<"\n First 5 rows of test_accuracies:"<<endl;
        cout<<fixed<<setprecision(2);
        for(int i = 0; i < 5; ++i) 
        {
            cout
                <<"n_layers="<<test_accuracies[i].n_layers
                <<", n_units="<<test_accuracies[i].n_units
                <<", accuracy="<<test_accuracies[i].accuracy
                <<endl;
        }

        float max_acc = -1.0;
        for(const auto& row : test_accuracies) 
        {
            if(row.accuracy > max_acc) 
            {
                max_acc = row.accuracy;
            }
        }

        cout<<"\n Best test accuracy rows:"<<endl;
        for(const auto& row : test_accuracies) 
        {
            if (row.accuracy == max_acc) 
            {
                cout
                    <<"n_layers="<<row.n_layers
                    <<", n_units="<<row.n_units
                    <<", accuracy="<<row.accuracy
                    <<endl;
            }
        }
   
   return 0;
}
   

   
