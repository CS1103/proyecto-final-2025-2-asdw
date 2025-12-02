//
// Created by Fernando on 25/11/2025.
//

#ifndef NN_INTERFACES_H
#define NN_INTERFACES_H
#pragma once
#include "../algebra/Tensor.h"
#include <memory>
#include <vector>

namespace utec::nn {

    using Tensor1 = utec::algebra::Tensor<double,1>;
    using Tensor2 = utec::algebra::Tensor<double,2>;

    // Layer interface
    struct Layer {
        virtual ~Layer() = default;
        // forward: input -> output
        virtual Tensor1 forward(const Tensor1& input) = 0;
        // backward: dL/dout -> returns dL/din
        virtual Tensor1 backward(const Tensor1& grad_output) = 0;
        // update parameters with learning rate
        virtual void update(double lr) = 0;
        // save/load minimal (optional)
        virtual void save(std::ostream& os) const = 0;
        virtual void load(std::istream& is) = 0;
    };

    using LayerPtr = std::unique_ptr<Layer>;

} // namespace utec::nn

#endif //NN_INTERFACES_H
