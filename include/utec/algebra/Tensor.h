#ifndef TENSOR_H
#define TENSOR_H

#pragma once
#include <array>
#include <vector>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <numeric>
#include <random>

namespace utec::algebra {

template<typename T, size_t Rank>
class Tensor {
    static_assert(Rank >= 1 && Rank <= 2, "Esta implementación soporta Rank 1 o 2 (suficiente para este proyecto).");
public:
    using shape_t = std::array<size_t, Rank>;

    Tensor() = default;

    // constructor from shape values
    Tensor(const shape_t& shape): shape_(shape) {
        size_ = 1;
        for (size_t i = 0; i < Rank; ++i) { size_ *= shape_[i]; }
        data_.assign(size_, T{});
    }

    // convenience constructor for Rank 1 or 2
    template<typename... Args>
    Tensor(Args... dims) {
        static_assert(sizeof...(Args) == Rank, "Número de dimensiones debe coincidir con Rank");
        shape_ = {static_cast<size_t>(dims)...};
        size_ = 1;
        for (size_t i = 0; i < Rank; ++i) size_ *= shape_[i];
        data_.assign(size_, T{});
    }

    size_t size() const noexcept { return size_; }
    const shape_t& shape() const noexcept { return shape_; }
    T* data() noexcept { return data_.data(); }
    const T* data() const noexcept { return data_.data(); }

    void fill(const T& v) { std::fill(data_.begin(), data_.end(), v); }

    // simple random init (normal)
    void random_normal(double mean = 0.0, double stddev = 0.01) {
        std::mt19937 gen(std::random_device{}());
        std::normal_distribution<> d(mean, stddev);
        for (auto &x: data_) x = static_cast<T>(d(gen));
    }

    // operator() variadic for Rank 1 and 2
    template<typename... Idx>
    T& operator()(Idx... idx) {
        static_assert(sizeof...(Idx) == Rank, "Número de índices incorrecto");
        std::array<size_t, Rank> a{static_cast<size_t>(idx)...};
        size_t flat = 0;
        if constexpr (Rank == 1) {
            flat = a[0];
        } else {
            flat = a[0] * shape_[1] + a[1];
        }
        return data_.at(flat);
    }

    template<typename... Idx>
    const T& operator()(Idx... idx) const {
        static_assert(sizeof...(Idx) == Rank, "Número de índices incorrecto");
        std::array<size_t, Rank> a{static_cast<size_t>(idx)...};
        size_t flat = 0;
        if constexpr (Rank == 1) {
            flat = a[0];
        } else {
            flat = a[0] * shape_[1] + a[1];
        }
        return data_.at(flat);
    }

    // reshape supports same total size
    void reshape(const shape_t& new_shape) {
        size_t new_size = 1;
        for (size_t i = 0; i < Rank; ++i) new_size *= new_shape[i];
        if (new_size != size_) throw std::runtime_error("reshape: tamaño no coincide");
        shape_ = new_shape;
    }

private:
    shape_t shape_{};
    size_t size_{0};
    std::vector<T> data_;
};

} // namespace utec::algebra

#endif //TENSOR_H