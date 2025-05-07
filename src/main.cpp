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
    float conf_threshold = 0.25f;
    float iou_threshold = 0.45f;

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
        
        // 转换为float32并归一化到[0,1]
        cv::Mat float32_img;
        rgb_img.convertTo(float32_img, CV_32F, 1.0/255.0);
        
        // 转换为float16
        cv::Mat float16_img;
        float32_img.convertTo(float16_img, CV_16F);
        
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

    struct Detection {
        cv::Rect box;
        float confidence;
        int class_id;
    };

    std::vector<Detection> postprocess(const std::vector<Ort::Value>& outputs, const cv::Size& original_size) {
        std::vector<Detection> detections;
        const float* output = outputs[0].GetTensorData<float>();
        auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        
        // 打印输出形状
        std::cout << "Output shape: [";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i] << (i < shape.size() - 1 ? ", " : "");
        }
        std::cout << "]" << std::endl;
        
        // YOLOv5 output shape: [1, 25200, 85] where 85 = 4 box coords + 1 conf + 80 classes
        int rows = shape[1];
        int cols = shape[2];
        
        // 打印前几个检测框的原始值
        std::cout << "First few detection values:" << std::endl;
        for (int i = 0; i < std::min(5, rows); ++i) {
            const float* row = output + i * cols;
            std::cout << "Box " << i << ": ";
            for (int j = 0; j < 5; ++j) {
                std::cout << row[j] << " ";
            }
            std::cout << std::endl;
        }
        
        // Scale factors for converting back to original image size
        float scale_x = static_cast<float>(original_size.width) / input_size.width;
        float scale_y = static_cast<float>(original_size.height) / input_size.height;

        for (int i = 0; i < rows; ++i) {
            const float* row = output + i * cols;
            float confidence = row[4];
            
            if (confidence < conf_threshold) continue;

            // Find class with highest confidence
            int class_id = 0;
            float max_class_conf = 0;
            for (int j = 5; j < cols; ++j) {
                if (row[j] > max_class_conf) {
                    max_class_conf = row[j];
                    class_id = j - 5;
                }
            }

            if (max_class_conf < conf_threshold) continue;

            // Convert normalized coordinates to pixel coordinates
            float x = row[0] * scale_x;
            float y = row[1] * scale_y;
            float w = row[2] * scale_x;
            float h = row[3] * scale_y;

            // Convert to top-left corner and width/height
            float x1 = x - w/2;
            float y1 = y - h/2;

            detections.push_back({
                cv::Rect(x1, y1, w, h),
                confidence * max_class_conf,
                class_id
            });
        }

        // Non-maximum suppression
        std::vector<Detection> nms_detections;
        std::sort(detections.begin(), detections.end(),
                 [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });

        for (const auto& det : detections) {
            bool should_keep = true;
            for (const auto& nms_det : nms_detections) {
                float iou = calculate_iou(det.box, nms_det.box);
                if (iou > iou_threshold) {
                    should_keep = false;
                    break;
                }
            }
            if (should_keep) {
                nms_detections.push_back(det);
            }
        }

        return nms_detections;
    }

    float calculate_iou(const cv::Rect& box1, const cv::Rect& box2) {
        int x1 = std::max(box1.x, box2.x);
        int y1 = std::max(box1.y, box2.y);
        int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
        int y2 = std::min(box1.y + box1.height, box2.y + box2.height);

        if (x2 < x1 || y2 < y1) return 0.0f;

        float intersection = (x2 - x1) * (y2 - y1);
        float area1 = box1.width * box1.height;
        float area2 = box2.width * box2.height;
        float union_area = area1 + area2 - intersection;

        return intersection / union_area;
    }

    void visualize_detections(cv::Mat& image, const std::vector<Detection>& detections) {
        // COCO class names
        std::vector<std::string> class_names = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        };

        for (const auto& det : detections) {
            cv::rectangle(image, det.box, cv::Scalar(0, 255, 0), 2);
            std::string label = class_names[det.class_id] + ": " + 
                              std::to_string(static_cast<int>(det.confidence * 100)) + "%";
            cv::putText(image, label, cv::Point(det.box.x, det.box.y - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }
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
        
        // 后处理和可视化
        auto detections = infer.postprocess(outputs, image.size());
        infer.visualize_detections(image, detections);
        
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