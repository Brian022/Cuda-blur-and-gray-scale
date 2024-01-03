#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

__global__ void rgb_to_grayscale_kernel(const uchar3* rgbImage, unsigned char* grayImage, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        grayImage[idx] = static_cast<unsigned char>(0.299f * rgbImage[idx].x + 0.587f * rgbImage[idx].y + 0.114f * rgbImage[idx].z);
    }
}

void rgb_to_grayscale_cuda(const uchar3* rgbImage, unsigned char* grayImage, int width, int height) {
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    rgb_to_grayscale_kernel<<<gridDim, blockDim>>>(rgbImage, grayImage, width, height);

    cudaDeviceSynchronize();
}

int main() {
    // Load RGB image using OpenCV
    cv::String image = "../cat.jpg";
    cv::Mat rgbImage = cv::imread(image, cv::IMREAD_COLOR);

    if (rgbImage.empty()) {
        std::cerr << "Error: Unable to load the image:  " << image << std::endl;
        return -1;
    }

    // Allocate memory for GPU buffers
    uchar3* d_rgbImage;
    unsigned char* d_grayImage;
    cudaMalloc(&d_rgbImage, rgbImage.total() * sizeof(uchar3));
    cudaMalloc(&d_grayImage, rgbImage.total());

    // Copy RGB image data to GPU
    cudaMemcpy(d_rgbImage, rgbImage.ptr<uchar3>(), rgbImage.total() * sizeof(uchar3), cudaMemcpyHostToDevice);

    // Convert RGB to grayscale using CUDA
    rgb_to_grayscale_cuda(d_rgbImage, d_grayImage, rgbImage.cols, rgbImage.rows);

    // Copy the result back to the CPU
    unsigned char* grayImage = new unsigned char[rgbImage.total()];
    cudaMemcpy(grayImage, d_grayImage, rgbImage.total(), cudaMemcpyDeviceToHost);

    // Create grayscale image using OpenCV
    cv::Mat grayscaleImage(rgbImage.size(), CV_8UC1, grayImage);

    // Display the original and grayscale images
    imwrite("../gray_Scale.jpg", grayscaleImage);
    //cv::imshow("Original", rgbImage);
    //cv::imshow("Grayscale (CUDA)", grayscaleImage);
    //cv::waitKey(0);

    // Clean up
    delete[] grayImage;
    cudaFree(d_rgbImage);
    cudaFree(d_grayImage);

    return 0;
}
