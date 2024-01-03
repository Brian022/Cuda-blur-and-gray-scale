#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// Kernel para aplicar un filtro de desenfoque a una imagen RGB
__global__ void blurKernel(const uchar3* inputImage, uchar3* outputImage, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        int blur_radius = 50; // Radio del filtro de desenfoque

        // Aplicar el filtro de desenfoque a cada canal de color
        float blur_value_r = 0.0f, blur_value_g = 0.0f, blur_value_b = 0.0f;
        int count = 0;

        for (int i = -blur_radius; i <= blur_radius; ++i) {
            for (int j = -blur_radius; j <= blur_radius; ++j) {
                int neighbor_x = x + i;
                int neighbor_y = y + j;

                if (neighbor_x >= 0 && neighbor_x < width && neighbor_y >= 0 && neighbor_y < height) {
                    uchar3 neighbor_pixel = inputImage[neighbor_y * width + neighbor_x];
                    blur_value_r += neighbor_pixel.x;
                    blur_value_g += neighbor_pixel.y;
                    blur_value_b += neighbor_pixel.z;
                    count++;
                }
            }
        }

        outputImage[idx].x = static_cast<unsigned char>(blur_value_r / count);
        outputImage[idx].y = static_cast<unsigned char>(blur_value_g / count);
        outputImage[idx].z = static_cast<unsigned char>(blur_value_b / count);
    }
}

// Función para aplicar un filtro de desenfoque en la GPU usando CUDA
void applyBlurCUDA(const uchar3* inputImage, uchar3* outputImage, int width, int height) {
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    blurKernel<<<gridDim, blockDim>>>(inputImage, outputImage, width, height);

    cudaDeviceSynchronize();
}

int main() {
    // Cargar la imagen RGB usando OpenCV
    cv::Mat rgbImage = cv::imread("../cat.jpg", cv::IMREAD_COLOR);

    if (rgbImage.empty()) {
        std::cerr << "Error: Unable to load the image." << std::endl;
        return -1;
    }

    // Obtener las dimensiones de la imagen
    int width = rgbImage.cols;
    int height = rgbImage.rows;

    // Copiar la imagen a la GPU
    uchar3* d_inputImage;
    uchar3* d_outputImage;
    cudaMalloc(&d_inputImage, width * height * sizeof(uchar3));
    cudaMalloc(&d_outputImage, width * height * sizeof(uchar3));
    cudaMemcpy(d_inputImage, rgbImage.ptr<uchar3>(), width * height * sizeof(uchar3), cudaMemcpyHostToDevice);

    // Aplicar el filtro de desenfoque en la GPU
    applyBlurCUDA(d_inputImage, d_outputImage, width, height);

    // Copiar el resultado de vuelta a la CPU
    uchar3* outputImage = new uchar3[width * height];
    cudaMemcpy(outputImage, d_outputImage, width * height * sizeof(uchar3), cudaMemcpyDeviceToHost);

    // Crear una imagen de salida usando OpenCV
    cv::Mat blurredImage(height, width, CV_8UC3, outputImage);

    // Mostrar la imagen original y la imagen desenfocada
    cv::imwrite("../blurredImage.jpg", blurredImage);
    //cv::imshow("Original", rgbImage);
    //cv::imshow("Blurred (CUDA)", blurredImage);
    //cv::waitKey(0);

    // Liberar la memoria
    delete[] outputImage;
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    return 0;
}



    

