//
// Created by Fernando on 29/11/2025.
//
#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <iostream>

// ============================================================================
//  UTILITIES
// ============================================================================

// Lee un archivo binario completo en un vector<uint8_t>
inline std::vector<uint8_t> read_file_binary(const std::string& filename) {
    // Abrir en modo binario
    std::ifstream f(filename, std::ios::in | std::ios::binary);

    if (!f.is_open()) {
        // Excepción si no se encuentra el archivo (esto falló con los archivos .raw antes)
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // Leer todo el contenido
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                                 std::istreambuf_iterator<char>());
}

// Eliminamos todo el código y namespaces de tiny_png y tiny_jpeg.

// ============================================================================
//  LOADER PRINCIPAL (Carga Exclusiva de Archivos .raw)
// ============================================================================

inline std::vector<uint8_t> load_image_grayscale(const std::string& path,
                                                 unsigned& w,
                                                 unsigned& h)
{
    // 1. Verificar la extensión: Si el archivo no es .raw, lo ignoramos.
    // Esto es crucial para que el clasificador salte los .png y .jpg
    // que persisten en el directorio.
    if (path.length() < 4 || path.substr(path.length() - 4) != ".raw") {
        // Lanzamos una excepción que será atrapada y marcada como "Saltando archivo corrupto"
        throw std::runtime_error("Skipping non-.raw file type.");
    }

    const std::string& raw_path = path;

    try {
        // 2. Lee el array de bytes crudos
        auto raw_data = read_file_binary(raw_path);

        // 3. Verifica el tamaño fijo
        const unsigned IMG_SIZE = 64;
        const size_t EXPECTED_SIZE = IMG_SIZE * IMG_SIZE; // 4096 bytes

        if (raw_data.size() != EXPECTED_SIZE) {
            throw std::runtime_error("RAW file size mismatch. Expected " + std::to_string(EXPECTED_SIZE) +
                                     " bytes, got " + std::to_string(raw_data.size()));
        }

        // 4. Retorna los datos y las dimensiones fijas
        w = IMG_SIZE;
        h = IMG_SIZE;
        return raw_data;

    } catch (const std::runtime_error& e) {
        // Relanzamos cualquier error de lectura/apertura para que se imprima el error
        throw e;
    } catch (...) {
        throw std::runtime_error("Unknown image loading error.");
    }
}

#endif //IMAGE_LOADER_H