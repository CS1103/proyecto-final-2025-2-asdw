//
// Created by Fernando on 25/11/2025.
//

#ifndef NN_LOSS_H
#define NN_LOSS_H

#pragma once
#include "nn_interfaces.h"
#include <algorithm>
#include <cmath>

namespace utec::nn {

    struct BinaryCrossEntropy {
        // input: prediction in (0,1), target: 0 or 1
        static double loss(const Tensor1& pred, const Tensor1& target) {
            double s = 0.0;
            for (size_t i = 0; i < pred.size(); ++i) {
                double p = std::clamp(pred(i), 1e-12, 1.0 - 1e-12);
                s += - (target(i) * std::log(p) + (1.0 - target(i)) * std::log(1.0 - p));
            }
            return s / pred.size();
        }
        static Tensor1 grad(const Tensor1& pred, const Tensor1& target) {
            Tensor1 g(pred.shape());
            for (size_t i = 0; i < pred.size(); ++i) {
                double p = std::clamp(pred(i), 1e-12, 1.0 - 1e-12);
                g(i) = (p - target(i)) / (p * (1.0 - p)) / static_cast<double>(pred.size());
                // but for stability we use (p - y) simple derivative for logits with Sigmoid handled differently.
                // We'll return (p - y)
                g(i) = (p - target(i)) / static_cast<double>(pred.size());
            }
            return g;
        }
    };

} // namespace utec::nn



#endif //NN_LOSS_H
