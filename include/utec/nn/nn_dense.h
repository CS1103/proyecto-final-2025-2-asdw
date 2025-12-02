//
// Created by Fernando on 25/11/2025.
//

#ifndef NN_DENSE_H
#define NN_DENSE_H

#pragma once
#include "nn_interfaces.h"
#include <random>
#include <iostream>

namespace utec::nn {

class Dense : public Layer {
public:
    Dense(size_t in_features, size_t out_features)
    : in_(in_features), out_(out_features),
      W(out_features, in_features), b(out_features)
    {
        // init shapes
        W.random_normal(0.0, 0.1);
        b.fill(0.0);
    }

    Tensor1 forward(const Tensor1& input) override {
        input_ = input; // store for backward
        Tensor1 out(out_);
        for (size_t i = 0; i < out_; ++i) {
            double s = b(i);
            for (size_t j = 0; j < in_; ++j) {
                s += W(i, j) * input(j);
            }
            out(i) = s;
        }
        return out;
    }

    Tensor1 backward(const Tensor1& grad_output) override {
        // grad_output: size out_
        // compute gradients
        dW = Tensor2(out_, in_);
        db = Tensor1(out_);
        dW.fill(0.0);
        db.fill(0.0);

        for (size_t i = 0; i < out_; ++i) {
            db(i) = grad_output(i);
            for (size_t j = 0; j < in_; ++j) {
                dW(i,j) = grad_output(i) * input_(j);
            }
        }
        // compute grad_input = W^T * grad_output
        Tensor1 grad_in(in_);
        for (size_t j = 0; j < in_; ++j) {
            double s = 0.0;
            for (size_t i = 0; i < out_; ++i) s += W(i,j) * grad_output(i);
            grad_in(j) = s;
        }
        return grad_in;
    }

    void update(double lr) override {
        for (size_t i = 0; i < out_; ++i) {
            b(i) -= lr * db(i);
            for (size_t j = 0; j < in_; ++j) {
                W(i,j) -= lr * dW(i,j);
            }
        }
    }

    void save(std::ostream& os) const override {
        // write dims then data
        os.write(reinterpret_cast<const char*>(&in_), sizeof(in_));
        os.write(reinterpret_cast<const char*>(&out_), sizeof(out_));
        os.write(reinterpret_cast<const char*>(W.data()), sizeof(double) * W.size());
        os.write(reinterpret_cast<const char*>(b.data()), sizeof(double) * b.size());
    }

    void load(std::istream& is) override {
        size_t inr=0, outr=0;
        is.read(reinterpret_cast<char*>(&inr), sizeof(inr));
        is.read(reinterpret_cast<char*>(&outr), sizeof(outr));
        if (inr != in_ || outr != out_) {
            // handle mismatch: resize
        }
        is.read(reinterpret_cast<char*>(const_cast<double*>(W.data())), sizeof(double)*W.size());
        is.read(reinterpret_cast<char*>(const_cast<double*>(b.data())), sizeof(double)*b.size());
    }

private:
    size_t in_, out_;
    Tensor2 W;
    Tensor1 b;
    // cached
    Tensor1 input_;
    // grads
    Tensor2 dW;
    Tensor1 db;
};

} // namespace utec::nn


#endif //NN_DENSE_H
