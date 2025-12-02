# 🐶🐱 Proyecto Final CS2013: DogCatMLP
## Clasificador de Perros y Gatos — MLP desde C++
**Curso:** CS2013 Programación III — Universidad UTEC — 2025-1  
**Tema:** Red Neuronal Multicapa (MLP) · Clasificación Binaria · Implementación desde cero

---

## 📝 Descripción del Proyecto

DogCatMLP es un clasificador binario de imágenes (perros vs. gatos) implementado desde cero en **C++**, usando una **Red Neuronal MultiCapa (MLP)** y entrenada mediante **Backpropagation** con **SGD**.

El sistema utiliza imágenes RAW de **64×64 píxeles**, una arquitectura simple pero funcional, y una librería interna (`utec/nn`) desarrollada como parte del curso.

---

## 📚 Contenidos

- Datos generales
- Requisitos e Instalación
- 1. Investigación Teórica
- 2. Diseño e Implementación
- 3. Ejecución y Pruebas
- 4. Análisis del Rendimiento
- 5. Trabajo en Equipo
- 6. Conclusiones
- 7. Bibliografía
- Licencia

---

## 📌 Datos generales

| Detalle | Valor                                                           |
|--------|-----------------------------------------------------------------|
| Tema | Clasificación Binaria de Imágenes usando Redes Neuronales (MLP) |
| Proyecto | DogCatMLP                                                       |
| Grupo | Grupo_ASWD                                                      |

### 👥 Integrantes

| Nombre del Alumno | Código    | Rol              |
|-------------------|-----------|------------------|
| Fernando Espinoza | 202420465 | Unico Integrante |

---

## 📦 Requisitos e Instalación

### Requisitos de Software

- **Compilador:** GCC 11+ (compatible con C++17)
- **Build System:** CMake 3.18+
- **Dependencias:** Librería interna `utec/nn`
- **Dataset:** Carpeta `../dataset/train/{dog,cat}` con imágenes RAW 64×64

---

### ⚙️ Instrucciones de Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/CS1103/proyecto-final-2025-2-asdw.git
cd EPIC3-ASWD

# 2. Configurar y compilar
mkdir build && cd build
cmake ..
make


---

# 📘 1. Investigación Teórica

Esta etapa estableció los fundamentos necesarios para la implementación de una red neuronal desde cero.

## Contenido Central

### 🧠 Historia y Evolución de las Redes Neuronales
Desde el Perceptrón y Adaline hasta las arquitecturas modernas.

### 🏗️ Arquitectura MLP
- Capacidad de aproximación universal  
- Adecuada para datos aplanados (imágenes 64×64 = 4096 pixeles)

### 🔧 Algoritmos de Entrenamiento
- **Backpropagation**: cálculo de gradientes  
- **SGD**: optimización estocástica  
- **Binary Cross Entropy (BCE)**: función de pérdida  

---

# 🏗️ 2. Diseño e Implementación

## 2.1 Arquitectura de la Solución

El modelo principal se encapsula en la clase **`DogCatClassifier`**, construida sobre la infraestructura de `utec/nn`.

### 🧩 Arquitectura de la Red MLP

| Capa | Configuración |
|------|---------------|
| Capa Densa (Input) | 4096 → 64 |
| Activación | ReLU |
| Capa Densa (Output) | 64 → 1 |
| Activación Final | Sigmoid |

### 🛠️ Patrones de Diseño

- **Strategy:** optimizador SGD  
- **Factory:** creación de capas (Dense, ReLU, Sigmoid)

### 📁 Estructura de Carpetas

proyecto-final-2025-2-asdw/  
├── src/  
│   ├── main.cpp  
├── include/  
│   ├── utec/
│   │   ├── nn/
│   │   │   ├── neural_network.h
│   │   │   ├── nn_activation.h
│   │   │   ├── nn_dense.h
│   │   │   ├── nn_interfaces.h
│   │   │   ├── nn_loss.h
│   │   │   ├── nn_optimizer.h
│   │   ├── algebra/
│   │   │   ├── Tensor.h
├── external/
│   ├── image_loader.h
├── dataset/  
│   ├── train/  
│   │   ├── dog/  
│   │   ├── cat/  
│   ├── test/  
│   │   ├── dog/  
│   │   ├── cat/
├── apps/
│   ├── DogCatClassifier.cpp
|   ├── DogCatClassifier.h
├── CMakeLists.txt
├── README.md

---

# 🧪 3. Ejecución y Pruebas

### 🎥 Demo
Video disponible en: `docs/demo.mp4`

### ⚙️ Parámetros de Entrenamiento

| Parámetro | Valor |
|-----------|--------|
| Épocas | 75 |
| Learning Rate | 0.005 |
| Train/Validation | 90% / 10% |
| Dimensión de Entrada | 4096 |
| Neuronas Ocultas | 64 |

---

# 📊 4. Análisis del Rendimiento

### 📈 Métricas

| Métrica | Valor |
|---------|--------|
| Iteraciones | 75 épocas |
| Tiempo total | *[Completar]* |
| Precisión de Validación | *[Completar, ej: 72.7%]* |
| Pérdida Final | *[Completar]* |

---

### ⚖️ Ventajas y Desventajas

| Aspecto | Ventaja | Desventaja |
|---------|----------|-------------|
| Código | Ligero, dependencias mínimas | No usa BLAS o Eigen |
| Rendimiento | Inferencia rápida | Entrenamiento sin paralelización |


---

# 🧠 5. Conclusiones

### 🏆 Logros
Se implementó un clasificador funcional basado en una **MLP desde cero en C++**, logrando una precisión aproximada de **[ej: 72.7%]** en un dataset real.

### 📘 Aprendizajes
- Entendimiento profundo del **Backpropagation**  
- Importancia de **normalización** e **inicialización de pesos**

### 🛠️ Recomendaciones
Para escalar el proyecto se sugiere optimizar memoria y cómputo usando:
- **BLAS**  
- **Eigen**  
- Paralelización mediante **mini-batches**  

---

# 📚 6. Bibliografía (Formato IEEE)

- Ringa Tech. (2021, 30 de noviembre). ¿Pocos datos de entrenamiento? Prueba esta técnica [Video]. YouTube. https://www.youtube.com/watch?v=9Dur_oUMGG8

---

# 📄 Licencia

Este proyecto está bajo la licencia **MIT**.  
Consulte el archivo `LICENSE` para más detalles.2