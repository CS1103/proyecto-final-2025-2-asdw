//
// Created by Fernando on 25/11/2025.
//

#ifndef NN_OPTIMIZER_H
#define NN_OPTIMIZER_H

#pragma once
#include <vector>
#include <memory>
#include "nn_interfaces.h"

namespace utec::nn {

struct SGD {
    SGD(double lr): lr_(lr) {}
    void step(std::vector<Layer*>& layers) {
        for (auto* l : layers) l->update(lr_);
    }
private:
    double lr_;
};

} // namespace utec::nn


#endif //NN_OPTIMIZER_H
