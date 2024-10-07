#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>

class Tensor {
public:
    std::vector<std::vector<double>> data;

    Tensor() {}

    Tensor(const std::vector<std::vector<double>>& input_data) : data(input_data) {}

    Tensor operator+(const Tensor& other) const {
        std::vector<std::vector<double>> result = data;
        for (size_t i = 0; i < data.size(); ++i) {
            for (size_t j = 0; j < data[i].size(); ++j) {
                result[i][j] += other.data[i][j];
            }
        }
        return Tensor(result);
    }

    // Messy implementation of matrix multiplication
    Tensor dot(const Tensor& other) const {
        size_t rows = data.size();
        size_t cols = other.data[0].size();
        size_t inner_dim = data[0].size();
        std::vector<std::vector<double>> result(rows, std::vector<double>(cols, 0));

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                for (size_t k = 0; k < inner_dim; ++k) {
                    result[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }
        return Tensor(result);
    }

    std::vector<double> flatten() const {
        std::vector<double> flat;
        for (const auto& row : data) {
            for (double val : row) {
                flat.push_back(val);
            }
        }
        return flat;
    }

    void print() const {
        for (const auto& row : data) {
            for (double val : row) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }

    std::pair<size_t, size_t> shape() const {
        return {data.size(), data[0].size()};
    }
};

class Layer {
public:
    virtual Tensor forward(const Tensor& input) = 0;
};

class DenseLayer : public Layer {
private:
    Tensor weights;
    Tensor bias;

public:
    DenseLayer(int input_size, int output_size) {
        std::srand(static_cast<unsigned>(std::time(0)));
        std::vector<std::vector<double>> weight_data(input_size, std::vector<double>(output_size));
        for (int i = 0; i < input_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                weight_data[i][j] = static_cast<double>(std::rand()) / RAND_MAX;
            }
        }
        weights = Tensor(weight_data);

        std::vector<std::vector<double>> bias_data(1, std::vector<double>(output_size, 0.0));
        bias = Tensor(bias_data);
    }

    Tensor forward(const Tensor& input) override {
        auto [input_rows, input_cols] = input.shape();
        std::vector<double> flattened_input = input.flatten();

        std::vector<std::vector<double>> reshaped_input(1, flattened_input);
        Tensor reshaped_tensor(reshaped_input);

        return reshaped_tensor.dot(weights) + bias;
    }
};

class Conv2D : public Layer {
private:
    int num_filters;
    int kernel_size;
    std::vector<Tensor> filters;

public:
    Conv2D(int num_filters, int kernel_size) : num_filters(num_filters), kernel_size(kernel_size) {
        std::srand(static_cast<unsigned>(std::time(0)));
        for (int i = 0; i < num_filters; ++i) {
            std::vector<std::vector<double>> filter(kernel_size, std::vector<double>(kernel_size));
            for (int j = 0; j < kernel_size; ++j) {
                for (int k = 0; k < kernel_size; ++k) {
                    filter[j][k] = static_cast<double>(std::rand()) / RAND_MAX;
                }
            }
            filters.push_back(Tensor(filter));
        }
    }

    Tensor convolve(const Tensor& input) {
        auto [input_rows, input_cols] = input.shape();

        //TO-DO: Implement better error handling
        if (input_rows < kernel_size || input_cols < kernel_size) {
            std::cerr << "Erro: O tamanho do kernel (" << kernel_size << ") é maior que o tamanho do input (" << input_rows << ", " << input_cols << ").\n";
            exit(EXIT_FAILURE);
        }

        size_t output_dim = input_rows - kernel_size + 1;
        std::vector<std::vector<double>> output(output_dim, std::vector<double>(output_dim, 0.0));

        for (int f = 0; f < num_filters; ++f) {
            for (size_t i = 0; i < output_dim; ++i) {
                for (size_t j = 0; j < output_dim; ++j) {
                    double sum = 0.0;
                    for (int ki = 0; ki < kernel_size; ++ki) {
                        for (int kj = 0; kj < kernel_size; ++kj) {
                            sum += input.data[i + ki][j + kj] * filters[f].data[ki][kj];
                        }
                    }
                    output[i][j] = sum;
                }
            }
        }
        return Tensor(output);
    }

    Tensor forward(const Tensor& input) override {
        return convolve(input);
    }
};

class Model {
private:
    std::vector<Layer*> layers;

public:
    void add(Layer* layer) {
        layers.push_back(layer);
    }

    // Think about further improvments in this method
    Tensor forward(const Tensor& input) {
        Tensor output = input;
        for (Layer* layer : layers) {
            output = layer->forward(output);
        }
        return output;
    }

    void summary() {
        std::cout << "Sumário do modelo criado:" << std::endl;
        for (Layer* layer : layers) {
            std::cout << typeid(*layer).name() << std::endl;
        }
    }

    // Maybe there is a better approach to dynamically delete the layers? Ask Nicola about it
    ~Model() {
        for (Layer* layer : layers) {
            delete layer;
        }
    }
};

int main() {
    // Simple test case, i didn't implement the backpropagation yet
    std::vector<std::vector<double>> input_data = {
        {1, 0, 1, 0, 1},
        {0, 1, 0, 1, 0},
        {1, 0, 1, 0, 1},
        {0, 1, 0, 1, 0},
        {1, 0, 1, 0, 1}
    };
    Tensor input_tensor(input_data);

    Model model;

    model.add(new Conv2D(1, 5));

    model.add(new DenseLayer(9, 1));

    model.add(new DenseLayer(1, 5));

    model.summary();

    Tensor output = model.forward(input_tensor);

    std::cout << "Resultado do Forward Pass:" << std::endl;
    output.print();

    return 0;
}

// I guess i've missed some point, but i need to read a file? I'm not sure about it. I'll ask Nicola about it.