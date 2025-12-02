//
// Created by Fernando on 1/12/2025.
//

#ifndef NN_ACTIVATION_H
#define NN_ACTIVATION_H

#pragma once
#include "nn_interfaces.h"
#include <cmath>

namespace utec::nn {
    class Sigmoid : public Layer {
    public:
        Tensor1 forward(const Tensor1& input) override {
            last_in = input;
            Tensor1 out(input.shape()[0]);
            for (size_t i = 0; i < input.size(); ++i) {
                out(i) = 1.0 / (1.0 + std::exp(-input(i)));
            }
            last_out = out;
            return out;
        }
        Tensor1 backward(const Tensor1& grad_output) override {
            Tensor1 grad(last_out.size());
            for (size_t i = 0; i < grad.size(); ++i) {
                double s = last_out(i);
                grad(i) = grad_output(i) * s * (1.0 - s);
            }
            return grad;
        }
        void update(double) override { }
        void save(std::ostream&) const override {}
        void load(std::istream&) override {}
    private:
        Tensor1 last_in;
        Tensor1 last_out;
    };
    class ReLU : public Layer {
    public:
        Tensor1 forward(const Tensor1& input) override {
            last_in = input;
            Tensor1 out(input.shape()[0]);
            for (size_t i = 0; i < input.size(); ++i) out(i) = input(i) > 0.0 ? input(i) : 0.0;
            return out;
        }
        Tensor1 backward(const Tensor1& grad_output) override {
            Tensor1 grad(grad_output.shape()[0]);
            for (size_t i = 0; i < grad.size(); ++i) grad(i) = (last_in(i) > 0.0) ? grad_output(i) : 0.0;
            return grad;
        }
        void update(double) override {}
        void save(std::ostream&) const override {}
        void load(std::istream&) override {}
    private:
        Tensor1 last_in;
    };
} // namespace utec::nn
#endif //NN_ACTIVATION_H