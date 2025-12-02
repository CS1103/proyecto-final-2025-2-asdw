//
// Created by Fernando on 25/11/2025.
//

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
#pragma once
#include "nn_interfaces.h"
#include "nn_loss.h"
#include "nn_optimizer.h"
#include <fstream>
#include <iostream>

namespace utec::nn {

class NeuralNetwork {
public:
    NeuralNetwork() = default;

    void add(LayerPtr&& layer) {
        layers_.push_back(std::move(layer));
    }

    Tensor1 forward(const Tensor1& x) const {
        Tensor1 out = x;
        for (auto& l : layers_) out = l->forward(out);
        return out;
    }

    // simple SGD training for batch=1 or mini-batch simulated externally
    void fit(std::vector<Tensor1>& X, std::vector<Tensor1>& Y, size_t epochs, double lr) {
        SGD opt(lr);
        for (size_t ep = 0; ep < epochs; ++ep) {
            double epoch_loss = 0.0;
            for (size_t i = 0; i < X.size(); ++i) {
                Tensor1 y_pred = forward(X[i]);
                double loss = BinaryCrossEntropy::loss(y_pred, Y[i]);
                epoch_loss += loss;
                Tensor1 grad = BinaryCrossEntropy::grad(y_pred, Y[i]);
                // backpropagate through layers in reverse
                Tensor1 g = grad;
                for (int j = (int)layers_.size()-1; j >= 0; --j) {
                    g = layers_[j]->backward(g);
                }
                // update
                // prepare raw pointers for opt
                std::vector<Layer*> raw;
                for (auto& lp : layers_) raw.push_back(lp.get());
                opt.step(raw);
            }
            if (ep % 10 == 0) {
                std::cout << "Epoch " << ep << " loss: " << (epoch_loss / X.size()) << "\n";
            }
        }
    }

    Tensor1 predict(const Tensor1& x) const {
        return forward(x);
    }

    void save(const std::string& filename) const {
        std::ofstream os(filename, std::ios::binary);
        if (!os) throw std::runtime_error("No se puede abrir archivo para guardar");
        // simple save: number of layers then each layer
        size_t n = layers_.size();
        os.write(reinterpret_cast<const char*>(&n), sizeof(n));
        for (auto& l : layers_) l->save(os);
    }

    void load(const std::string& filename) {
        // Loading would need layer-by-layer matching; left as exercise
        std::ifstream is(filename, std::ios::binary);
        if (!is) throw std::runtime_error("No se puede abrir archivo para cargar");
        size_t n;
        is.read(reinterpret_cast<char*>(&n), sizeof(n));
        // can't generically instantiate polymorphic layers without metadata
        // here we assume the network architecture is reconstructed in code then load weights layer-by-layer
    }

private:
    std::vector<LayerPtr> layers_;
};

} // namespace utec::nn

#endif //NEURAL_NETWORK_H
