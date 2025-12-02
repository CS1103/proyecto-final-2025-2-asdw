#include "../apps/DogCatClassifier.h"
#include "../external/image_loader.h" // Necesario para load_image_grayscale
#include <iostream>
#include <vector>
#include <stdexcept>

// ----------------------------------------------------------------------------------
// Función Auxiliar para Cargar y Normalizar una Imagen de Prueba
// ----------------------------------------------------------------------------------
std::vector<double> load_and_normalize_test_image(const std::string& filepath) {
    unsigned w, h;

    // Cargar los bytes del archivo .raw
    std::vector<uint8_t> image_data = load_image_grayscale(filepath, w, h);

    if (image_data.size() != 64 * 64) {
        throw std::runtime_error("El archivo RAW tiene un tamano incorrecto. Debe ser 64x64 (4096 bytes).");
    }

    // Convertir y Normalizar a double [0.0, 1.0]
    std::vector<double> input;
    for (uint8_t byte : image_data) {
        input.push_back(static_cast<double>(byte) / 255.0);
    }
    return input;
}


int main(int argc, char** argv) {

    // --- PARTE DE ENTRENAMIENTO COMENTADA ---
    /*
    std::string dataset = "../dataset/train"; // Ruta de tu carpeta de entrenamiento
    int img_size = 64;
    int input_dim = img_size * img_size;

    // 2. Inicializar clasificador (4096 entradas, 128 neuronas ocultas)
    DogCatClassifier clf_train(input_dim, 64);

    // 3. Hiperparámetros de entrenamiento
    int epochs = 75;
    double learning_rate = 0.005;
    double train_validation_split = 0.9;

    std::cout << "--- Starting Training ---\n";
    // 4. Entrenar y Validar
    clf_train.train(dataset, epochs, learning_rate, img_size, train_validation_split);
    std::cout << "--- Training Finished ---\n";

    // 5. Guardar modelo
    clf_train.save_model("dogcat.model");

    // Si tienes un conjunto de datos de prueba (Test) separado, usa esta línea:
    // double acc_test = clf_train.evaluate("data/test", img_size);
    // std::cout << "Test Accuracy: " << acc_test << "%\n";
    */
    // ----------------------------------------


    // ----------------------------------------
    // --- MODO DE PRUEBA (TESTING MODE) ACTIVO ---
    // ----------------------------------------

    // 1. Configuración del Modelo (DEBE coincidir con el entrenamiento)
    const int img_size = 64;
    const int input_dim = img_size * img_size; // 4096
    const int hidden_dim = 64; // Crucial: Usar el valor que dio 72.7%
    const std::string model_file = "dogcat.model";

    // !!! IMPORTANTE: REEMPLAZA ESTA RUTA CON LA RUTA DE TU IMAGEN .RAW DE PRUEBA !!!
    const std::string test_image_path = "../dataset/test/cat/gatotest.raw";

    // 2. Inicializar y Cargar el Modelo
    DogCatClassifier clf_test(input_dim, hidden_dim);

    std::cout << "--- Starting Test Mode ---\n";
    try {
        clf_test.load_model(model_file);
        std::cout << "Modelo cargado exitosamente: " << model_file << "\n";
    } catch (const std::runtime_error& e) {
        std::cerr << "ERROR al cargar el modelo: " << e.what() << ". Asegurate de que el archivo existe.\n";
        return 1;
    }


    // 3. Preparar y Predecir la Imagen
    try {
        std::vector<double> input = load_and_normalize_test_image(test_image_path);

        std::vector<double> prediction = clf_test.predict(input);

        double cat_prob = prediction[0];
        double dog_prob = prediction[1];

        std::cout << "\nPrediccion para la imagen: " << test_image_path << "\n";
        std::cout << "-------------------------------------------\n";
        std::cout << "  Prob. Perro (Clase 0): " << dog_prob * 100.0 << "%\n";
        std::cout << "  Prob. Gato (Clase 1): " << cat_prob * 100.0 << "%\n";
        std::cout << "-------------------------------------------\n";

        if (cat_prob > dog_prob) {
            std::cout << "--> CLASIFICADO COMO: GATO  (Confianza: " << cat_prob * 100.0 << "%)\n";
        } else {
            std::cout << "--> CLASIFICADO COMO: PERRO  (Confianza: " << dog_prob * 100.0 << "%)\n";
        }

    } catch (const std::runtime_error& e) {
        std::cerr << "\nERROR durante la prueba: " << e.what() << "\n";
        return 1;
    }

    std::cout << "\n--- Test Finished ---\n";
    return 0;
}

