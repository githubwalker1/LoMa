#include <torch/script.h>
#include <torch/torch.h>

#include <opencv2/opencv.hpp>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Example for TorchScript modules exported by:
//
//   python scripts/export_loma_torchscript.py \
//     --output-dir exported_loma_b128 \
//     --model loma_b128 \
//     --num-keypoints 2048 \
//     --image-size 1024 \
//     --kind split
//
// The same C++ calling code works for loma_b/loma_l/loma_g/loma_r. The main
// difference is descriptor dimension:
//   loma_b128: D = 128
//   loma_b, loma_l, loma_g, loma_r: D = 256
//
// Load the three TorchScript files from the SAME export directory. Do not mix
// detector/descriptor/matcher files exported from different presets or K.

struct LomaFeatures {
    torch::Tensor keypoints;       // [K, 2], normalized coordinates in [-1, 1]
    torch::Tensor keypoint_probs;  // [K]
    torch::Tensor descriptors;     // [K, D]
};

struct LomaMatch {
    int query_index;       // index in image A keypoints
    int train_index;       // matching index in image B keypoints
    float score;           // matcher confidence score
    cv::Point2f point_a;   // pixel coordinate in model input image
    cv::Point2f point_b;   // pixel coordinate in model input image
};

// Convert an OpenCV BGR image to the tensor format used by the exported models:
//   shape: [1, 3, image_size, image_size]
//   layout: NCHW
//   color: RGB
//   dtype: float32
//   value range: [0, 1]
//
// This example uses direct square resize. That matches the current export
// script's fixed-shape path and is the simplest setup for LibTorch/TensorRT.
// If you need to reproduce Python path-mode exactly, implement its detector
// resize and descriptor resize separately instead.
torch::Tensor preprocessImage(
    const cv::Mat& bgr_image,
    int image_size,
    torch::Device device
) {
    if (bgr_image.empty()) {
        throw std::runtime_error("preprocessImage received an empty image.");
    }

    cv::Mat rgb_image;
    cv::cvtColor(bgr_image, rgb_image, cv::COLOR_BGR2RGB);

    cv::Mat resized_image;
    cv::resize(rgb_image, resized_image, cv::Size(image_size, image_size));

    cv::Mat float_image;
    resized_image.convertTo(float_image, CV_32FC3, 1.0 / 255.0);

    auto tensor = torch::from_blob(
        float_image.data,
        {1, image_size, image_size, 3},
        torch::TensorOptions().dtype(torch::kFloat32)
    );

    // from_blob does not own memory. clone() makes the tensor independent of
    // the local cv::Mat before returning.
    return tensor.permute({0, 3, 1, 2}).contiguous().clone().to(device);
}

// Convert LoMa normalized coordinates [-1, 1] to pixel coordinates in the
// model input image. For a square 1024 export, width=height=1024.
cv::Point2f normalizedToPixel(
    const torch::Tensor& point,
    int width,
    int height
) {
    const float x = point[0].item<float>();
    const float y = point[1].item<float>();
    return cv::Point2f(
        static_cast<float>(width) * (x + 1.0f) * 0.5f,
        static_cast<float>(height) * (y + 1.0f) * 0.5f
    );
}

// Optional helper if the model input was direct square-resized from an original
// image and you need coordinates back in the original image coordinate system.
cv::Point2f modelPixelToOriginalPixel(
    const cv::Point2f& point,
    int model_width,
    int model_height,
    int original_width,
    int original_height
) {
    return cv::Point2f(
        point.x * static_cast<float>(original_width) / static_cast<float>(model_width),
        point.y * static_cast<float>(original_height) / static_cast<float>(model_height)
    );
}

// Run the split detector + descriptor modules for one image.
//
// detector input:
//   image: [1, 3, H, W]
// detector output:
//   keypoints: [K, 2]
//   keypoint_probs: [K]
//
// descriptor input:
//   image: [1, 3, H, W]
//   keypoints: [K, 2]
// descriptor output:
//   descriptors: [K, D]
LomaFeatures extractFeatures(
    torch::jit::script::Module& detector,
    torch::jit::script::Module& descriptor,
    const torch::Tensor& image
) {
    auto detector_output = detector.forward({image}).toTuple();
    torch::Tensor keypoints = detector_output->elements()[0].toTensor();
    torch::Tensor keypoint_probs = detector_output->elements()[1].toTensor();

    auto descriptor_output = descriptor.forward({image, keypoints});
    torch::Tensor descriptors = descriptor_output.toTensor();

    return {
        keypoints.contiguous(),
        keypoint_probs.contiguous(),
        descriptors.contiguous(),
    };
}

// Run the split matcher module.
//
// matcher input:
//   keypoints0: [K, 2]
//   descriptors0: [K, D]
//   keypoints1: [K, 2]
//   descriptors1: [K, D]
//
// matcher output:
//   matches0: [K], int64. matches0[i] = j means A[i] matches B[j].
//              matches0[i] = -1 means no accepted match.
//   match_scores0: [K], float32.
std::vector<LomaMatch> matchFeatures(
    torch::jit::script::Module& matcher,
    const LomaFeatures& features_a,
    const LomaFeatures& features_b,
    int image_width,
    int image_height
) {
    auto matcher_output = matcher
        .forward({
            features_a.keypoints,
            features_a.descriptors,
            features_b.keypoints,
            features_b.descriptors,
        })
        .toTuple();

    torch::Tensor matches0 = matcher_output->elements()[0].toTensor().to(torch::kCPU);
    torch::Tensor scores0 = matcher_output->elements()[1].toTensor().to(torch::kCPU);
    torch::Tensor keypoints_a = features_a.keypoints.to(torch::kCPU);
    torch::Tensor keypoints_b = features_b.keypoints.to(torch::kCPU);

    std::vector<LomaMatch> matches;
    matches.reserve(static_cast<size_t>(matches0.size(0)));

    for (int64_t i = 0; i < matches0.size(0); ++i) {
        const int64_t j = matches0[i].item<int64_t>();
        if (j < 0) {
            continue;
        }

        const cv::Point2f point_a =
            normalizedToPixel(keypoints_a[i], image_width, image_height);
        const cv::Point2f point_b =
            normalizedToPixel(keypoints_b[j], image_width, image_height);

        matches.push_back({
            static_cast<int>(i),
            static_cast<int>(j),
            scores0[i].item<float>(),
            point_a,
            point_b,
        });
    }

    return matches;
}

// Load one TorchScript module and move it to the selected device.
torch::jit::script::Module loadModule(
    const std::string& path,
    torch::Device device
) {
    torch::jit::script::Module module = torch::jit::load(path, device);
    module.eval();
    return module;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr
            << "Usage:\n"
            << "  " << argv[0]
            << " <export_dir> <image_a> <image_b> <image_size>\n\n"
            << "Example:\n"
            << "  " << argv[0]
            << " exported_loma_b128 assets/c.jpeg assets/d.jpeg 1024\n";
        return 1;
    }

    const std::string export_dir = argv[1];
    const std::string image_a_path = argv[2];
    const std::string image_b_path = argv[3];
    const int image_size = std::stoi(argv[4]);

    torch::NoGradGuard no_grad;
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    torch::jit::script::Module detector =
        loadModule(export_dir + "/loma_detector.pt", device);
    torch::jit::script::Module descriptor =
        loadModule(export_dir + "/loma_descriptor.pt", device);
    torch::jit::script::Module matcher =
        loadModule(export_dir + "/loma_matcher.pt", device);

    cv::Mat image_a_bgr = cv::imread(image_a_path, cv::IMREAD_COLOR);
    cv::Mat image_b_bgr = cv::imread(image_b_path, cv::IMREAD_COLOR);
    if (image_a_bgr.empty() || image_b_bgr.empty()) {
        throw std::runtime_error("Failed to read one or both input images.");
    }

    torch::Tensor image_a = preprocessImage(image_a_bgr, image_size, device);
    torch::Tensor image_b = preprocessImage(image_b_bgr, image_size, device);

    LomaFeatures features_a = extractFeatures(detector, descriptor, image_a);
    LomaFeatures features_b = extractFeatures(detector, descriptor, image_b);

    std::vector<LomaMatch> matches =
        matchFeatures(matcher, features_a, features_b, image_size, image_size);

    std::cout << "Detected keypoints A: " << features_a.keypoints.size(0) << "\n";
    std::cout << "Detected keypoints B: " << features_b.keypoints.size(0) << "\n";
    std::cout << "Accepted matches: " << matches.size() << "\n";

    const size_t max_print = std::min<size_t>(matches.size(), 10);
    for (size_t idx = 0; idx < max_print; ++idx) {
        const LomaMatch& match = matches[idx];
        std::cout
            << "match " << idx
            << ": A[" << match.query_index << "] " << match.point_a
            << " -> B[" << match.train_index << "] " << match.point_b
            << ", score=" << match.score
            << "\n";
    }

    return 0;
}
