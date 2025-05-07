#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <chrono>
#include <cstring>
#include <climits>    // For PATH_MAX
#include <unistd.h>   // For realpath on Linux/Unix

class ONNXInference {
private:
    Ort::Env env;
    Ort::Session session;
    Ort::SessionOptions session_options;
    cv::Size input_size{640, 640};
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

public:
    ONNXInference(const std::string& model_path)
        : env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime"),
          session(nullptr)
    {
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        session = Ort::Session(env, model_path.c_str(), session_options);

        // 获取输入输出名
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = session.GetInputCount();
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = session.GetInputNameAllocated(i, allocator);
            input_names.push_back(input_name.get());
        }
        size_t num_output_nodes = session.GetOutputCount();
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = session.GetOutputNameAllocated(i, allocator);
            output_names.push_back(output_name.get());
        }
    }

    std::vector<uint16_t> preprocess(const cv::Mat& input) {
        cv::Mat resized;
        cv::resize(input, resized, input_size);
        cv::Mat rgb_img;
        cv::cvtColor(resized, rgb_img, cv::COLOR_BGR2RGB);
        cv::Mat float16_img;
        rgb_img.convertTo(float16_img, CV_16F, 1.0/255.0);
        std::vector<uint16_t> input_tensor_values(float16_img.total() * float16_img.channels());
        std::memcpy(input_tensor_values.data(), float16_img.data, input_tensor_values.size() * sizeof(uint16_t));
        return input_tensor_values;
    }

    std::vector<Ort::Value> inference(const cv::Mat& input) {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<uint16_t> input_tensor_values = preprocess(input);
        std::array<int64_t, 4> input_shape{1, 3, input_size.height, input_size.width};
        // NHWC to NCHW for float16
        std::vector<uint16_t> nchw(input_tensor_values.size());
        size_t hw = input_size.height * input_size.width;
        for (size_t h = 0; h < input_size.height; ++h) {
            for (size_t w = 0; w < input_size.width; ++w) {
                for (size_t c = 0; c < 3; ++c) {
                    nchw[c * hw + h * input_size.width + w] = input_tensor_values[(h * input_size.width + w) * 3 + c];
                }
            }
        }
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor(
            memory_info,
            nchw.data(),
            nchw.size() * sizeof(uint16_t),
            input_shape.data(),
            input_shape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
        );

        // 将 std::string 转换为 const char*
        std::vector<const char*> input_names_char;
        std::vector<const char*> output_names_char;
        for (const auto& name : input_names) input_names_char.push_back(name.c_str());
        for (const auto& name : output_names) output_names_char.push_back(name.c_str());

        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names_char.data(), &input_tensor, 1, output_names_char.data(), output_names_char.size());
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Inference time: " << duration.count() << "ms" << std::endl;
        return output_tensors;
    }
};

int main(int argc, char* argv[]) {
    std::string model_path = "/home/andao/andaoai/FastInference/yolov5n.onnx";
    std::string image_path = "/home/andao/andaoai/FastInference/1746581949986.jpg";

    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) image_path = argv[2];

    try {
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Error: Could not read image." << std::endl;
            return -1;
        }
        ONNXInference infer(model_path);
        auto outputs = infer.inference(image);
        // 输出第一个输出的 shape
        auto& out = outputs[0];
        auto type_info = out.GetTensorTypeAndShapeInfo();
        auto shape = type_info.GetShape();
        std::cout << "Output shape: ";
        for (auto s : shape) std::cout << s << " ";
        std::cout << std::endl;
        
        // 保存结果图片
        std::string result_path = "result.jpg";
        cv::imwrite(result_path, image);
        
        // 获取并打印结果图片的完整路径
        char resolved_path[PATH_MAX];
        if (realpath(result_path.c_str(), resolved_path) != nullptr) {
            std::cout << "结果图片已保存至: " << resolved_path << std::endl;
        } else {
            std::cout << "结果图片已保存至当前目录: " << result_path << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
} 