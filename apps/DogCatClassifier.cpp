#include "DogCatClassifier.h"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <stdexcept>

// Asegúrate que esta ruta es correcta para tu proyecto
#include "../external/image_loader.h"

namespace fs = std::filesystem;
using namespace utec::nn; // Hacemos el namespace global aquí para simplificar

// ----------------------------------------------------------------------
// CONSTRUCTOR
// ----------------------------------------------------------------------
DogCatClassifier::DogCatClassifier(int input_dim, int hidden)
    : input_dim_(input_dim)
{
    // Arquitectura: FC(4096->64) → ReLU → FC(64->1) → Sigmoid
    net.add(std::make_unique<Dense>(input_dim_, hidden));
    net.add(std::make_unique<ReLU>());
    net.add(std::make_unique<Dense>(hidden, 1));
    net.add(std::make_unique<Sigmoid>());
}

// ----------------------------------------------------------------------
// IMPLEMENTACIÓN DEL MÉTODO PREDICT (SOLUCIÓN FINAL)
// ----------------------------------------------------------------------
std::vector<double> DogCatClassifier::predict(const std::vector<double>& input) const {
    // 1. Verificar el tamaño
    if (input.size() != input_dim_) {
        throw std::runtime_error("El tamaño de la entrada para predict no coincide con el input_dim del modelo.");
    }

    // 2. Convertir la entrada (std::vector<double>) a Tensor1
    Tensor1 x(input_dim_);
    for (size_t i = 0; i < input.size(); ++i) {
        x(i) = input[i];
    }

    // 3. Ejecutar la predicción en la red neuronal (net.predict(x) devuelve Tensor1)
    Tensor1 pred_tensor = net.predict(x);

    // 4. Convertir la salida (Tensor1) a std::vector<double> y calcular las probabilidades
    // La salida de tu red Sigmoid es un Tensor1 de tamaño 1 [Prob_Perro]
    if (pred_tensor.size() != 1) {
        throw std::runtime_error("La red neuronal devolvió una dimensión de salida inesperada.");
    }

    double prob_perro = pred_tensor(0);
    double prob_gato = 1.0 - prob_perro;

    // Devolvemos [Prob. Perro, Prob. Gato]
    return {prob_perro, prob_gato};
}

// ----------------------------------------------------------------------
// PREPROCESS_IMAGE
// ----------------------------------------------------------------------
std::vector<double> DogCatClassifier::preprocess_image(const std::string& path, int img_size)
{
    unsigned width = 0, height = 0;
    std::vector<uint8_t> data;
    try {
        data = load_image_grayscale(path, width, height);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Error al cargar imagen: ") + path + " -> " + e.what());
    }

    if (data.empty() || width == 0 || height == 0) {
        throw std::runtime_error("Imagen vacía o inválida: " + path);
    }

    std::vector<double> resized(img_size * img_size);

    // Redimensionado "nearest neighbor" hecho a mano
    for (int y = 0; y < img_size; ++y) {
        for (int x = 0; x < img_size; ++x) {
            int srcY = y * static_cast<int>(height) / img_size;
            int srcX = x * static_cast<int>(width)  / img_size;

            uint8_t pixel = data[srcY * width + srcX];

            // Normalización a [0.0, 1.0]
            resized[y * img_size + x] = static_cast<double>(pixel) / 255.0;
        }
    }

    return resized;
}

// ----------------------------------------------------------------------
// LOAD_DATASET
// ----------------------------------------------------------------------
std::vector<std::pair<std::vector<double>, double>>
DogCatClassifier::load_dataset(const std::string& folder, int img_size)
{
    std::vector<std::pair<std::vector<double>, double>> data;

    for (auto cls : {"dog", "cat"}) {
        std::string cls_path = folder + "/" + cls;
        if (!fs::exists(cls_path)) continue;

        for (auto& p : fs::directory_iterator(cls_path)) {
            if (!p.is_regular_file()) continue;
            try {
                auto v = preprocess_image(p.path().string(), img_size);
                // Etiqueta: 1.0 para perro, 0.0 para gato
                double label = (std::string(cls) == "dog") ? 1.0 : 0.0;
                data.push_back({std::move(v), label});
            }
            catch (...) {
                std::cout << "Saltando archivo corrupto: " << p.path() << "\n";
            }
        }
    }
    return data;
}

// ----------------------------------------------------------------------
// EVALUATE_INTERNAL
// ----------------------------------------------------------------------
double DogCatClassifier::evaluate_internal(
    const std::vector<std::pair<std::vector<double>, double>>& data)
{
    size_t correct = 0;

    for (auto& kv : data) {
        Tensor1 x(input_dim_);
        for (size_t i = 0; i < kv.first.size(); ++i) x(i) = kv.first[i];

        auto pred = net.predict(x);
        double p = pred(0);

        int predicted = (p > 0.5) ? 1 : 0;
        int truth     = (kv.second > 0.5) ? 1 : 0;

        if (predicted == truth) correct++;
    }

    if (data.empty()) return 0.0;
    return 100.0 * correct / data.size();
}

// ----------------------------------------------------------------------
// TRAIN
// ----------------------------------------------------------------------
void DogCatClassifier::train(
    const std::string& dataset_folder,
    int epochs,
    double lr,
    int img_size,
    double split_ratio)
{
    auto data = load_dataset(dataset_folder, img_size);
    if (data.empty()) {
        std::cout << "No hay datos en el dataset: " << dataset_folder << "\n";
        return;
    }

    // 1. Aleatorizar y dividir los datos
    std::mt19937 rng(std::random_device{}());
    std::shuffle(data.begin(), data.end(), rng);

    size_t train_size = static_cast<size_t>(data.size() * split_ratio);
    if (train_size == 0 && !data.empty()) train_size = 1;
    if (train_size > data.size()) train_size = data.size();

    std::vector<std::pair<std::vector<double>, double>> train_data(
        data.begin(), data.begin() + train_size);

    std::vector<std::pair<std::vector<double>, double>> validation_data;
    if (train_size < data.size())
        validation_data = std::vector<std::pair<std::vector<double>, double>>(
            data.begin() + train_size, data.end());

    std::vector<Tensor1> X_train;
    std::vector<Tensor1> Y_train;

    // 2. Convertir entrenamiento a Tensores
    for (auto& kv : train_data) {
        Tensor1 x(input_dim_);
        for (size_t i = 0; i < kv.first.size(); ++i) x(i) = kv.first[i];

        Tensor1 y(1);
        y(0) = kv.second;

        X_train.push_back(x);
        Y_train.push_back(y);
    }

    std::cout << "Training samples: " << X_train.size()
              << ", Validation samples: " << validation_data.size() << "\n";

    if (X_train.empty()) {
        std::cout << "No hay muestras de entrenamiento después del split. Abortando entrenamiento.\n";
        return;
    }

    // 3. Entrenar
    net.fit(X_train, Y_train, epochs, lr);

    // 4. Evaluar en el conjunto de VALIDACIÓN
    if (!validation_data.empty()) {
        double acc_val = evaluate_internal(validation_data);
        std::cout << "Validation Accuracy: " << acc_val << "%\n";
    }
}

// ----------------------------------------------------------------------
// EVALUATE, SAVE, LOAD
// ----------------------------------------------------------------------
double DogCatClassifier::evaluate(const std::string& dataset_folder, int img_size)
{
    auto data = load_dataset(dataset_folder, img_size);
    return evaluate_internal(data);
}

void DogCatClassifier::save_model(const std::string& file)
{
    net.save(file);
}

void DogCatClassifier::load_model(const std::string& file)
{
    // Carga la estructura existente con los pesos guardados
    net.load(file);
}