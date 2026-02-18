cmake_minimum_required(VERSION 3.12)
project(HandwritingRecognition LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

set(SOURCES
    src/main.cpp
    src/Constants.cpp
    src/ImageProcessing.cpp
    src/BoxExtractor.cpp
    src/MNISTLoader.cpp
    src/BPNeuralNetwork.cpp
    src/Recognition.cpp
)

add_executable(HandwritingRecognition ${SOURCES})

target_include_directories(HandwritingRecognition PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(HandwritingRecognition PRIVATE ${OpenCV_LIBS})

# Optional: OpenBLAS (OFF by default)
option(USE_OPENBLAS "Use OpenBLAS for acceleration" OFF)
if(USE_OPENBLAS)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(OPENBLAS REQUIRED openblas)
    target_compile_definitions(HandwritingRecognition PRIVATE USE_OPENBLAS)
    target_include_directories(HandwritingRecognition PRIVATE ${OPENBLAS_INCLUDE_DIRS})
    target_link_libraries(HandwritingRecognition PRIVATE ${OPENBLAS_LIBRARIES})
endif()
