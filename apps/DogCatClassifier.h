//
// Created by Fernando on 29/11/2025.
//
#ifndef DOGCATCLASSIFIER_H
#define DOGCATCLASSIFIER_H

#pragma once
#include "../include/utec/nn/neural_network.h"
#include "../include/utec/nn/nn_dense.h"
#include "../include/utec/nn/nn_activation.h"

#include <string>
#include <vector>
#include <algorithm>
#include <random>

class DogCatClassifier {
public:
    DogCatClassifier(int input_dim, int hidden = 128);

    // **********************************************
    // * CORRECCIÓN CLAVE: Agregamos predict
    // **********************************************
    std::vector<double> predict(const std::vector<double>& input) const;

    void train(const std::string& dataset_folder,
               int epochs,
               double lr,
               int img_size = 64,
               double split_ratio = 1.0);

    double evaluate(const std::string& dataset_folder, int img_size = 64);

    void save_model(const std::string& file);
    void load_model(const std::string& file);

private:
    std::vector<std::pair<std::vector<double>, double>>
    load_dataset(const std::string& folder, int img_size);

    std::vector<double> preprocess_image(const std::string& path, int img_size);

    double evaluate_internal(const std::vector<std::pair<std::vector<double>, double>>& data);

    utec::nn::NeuralNetwork net;
    int input_dim_;
};


#endif //DOGCATCLASSIFIER_H