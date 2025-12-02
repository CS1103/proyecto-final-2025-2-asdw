cmake_minimum_required(VERSION 3.29)
project(EPIC3_ASWD)

set(CMAKE_CXX_STANDARD 20)

add_executable(EPIC3_ASWD src/main.cpp apps/DogCatClassifier.cpp)
