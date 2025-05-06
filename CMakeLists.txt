cmake_minimum_required(VERSION 3.15)
project(FastInference VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 使用 Conan 生成的配置文件
include(${CMAKE_BINARY_DIR}/conan_toolchain.cmake)

find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)

add_executable(fast_inference src/main.cpp)

target_link_libraries(fast_inference PRIVATE
    ${OpenCV_LIBS}
    onnxruntime::onnxruntime
)

target_include_directories(fast_inference PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
) 