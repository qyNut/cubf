#include "tensor.cuh"
#include <cassert>
#include <cuda_runtime.h>

template <typename T>
Tensor<T>::Tensor() {
    shape_ = { 0 };
    size_ = 0;
}

template <typename T>
Tensor<T>::Tensor(const std::vector<size_t>& shape) : shape_(shape) {
    size_ = 1;
    for (size_t dim : shape_) {
        size_ *= dim;
    }

    if (cudaMalloc(&data_, size_ * sizeof(T)) != cudaSuccess) {
        printf("Tensor Memory allocation Failed!\n");
    }
}

template <typename T>
Tensor<T>::Tensor(const std::vector<size_t>& shape, const T* buf) : shape_(shape) {
    size_ = 1;
    for (size_t dim : shape_) {
        size_ *= dim;
    }

    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc(&data_, size_ * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        printf("Tensor Memory allocation Failed! Error Type: %d\n", cudaStatus);
    }
    cudaStatus = cudaMemcpy(data_, buf, size_ * sizeof(T), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("Tensor Memory Transfer Failed! Error Type: %d\n", cudaStatus);
    }
}

template <typename T>
Tensor<T>::~Tensor() {
    cudaFree(data_);
}

template<typename T>
void Tensor<T>::setData(const T* buf){

    cudaMemcpy(data_, buf, size_ * sizeof(T), cudaMemcpyHostToDevice);

}


template <typename T>
void Tensor<T>::set(const std::vector<size_t>& indices, const T& value) {
    size_t index = calculateIndex(indices);
    cudaMemcpy(&data_[index], &value, sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
T Tensor<T>::get(const std::vector<size_t>& indices) const {
    size_t index = calculateIndex(indices);
    T value;
    cudaMemcpy(&value, &data_[index], sizeof(T), cudaMemcpyDeviceToHost);
    return value;
}
template <typename T>
T* Tensor<T>::data() const {
    return data_;
}

template <typename T>
const std::vector<size_t>& Tensor<T>::shape() const {
    return shape_;
}

template <typename T>
void Tensor<T>::loadData(const T* buf) {
    cudaMemcpy(data_, buf, size_ * sizeof(T), cudaMemcpyHostToDevice);
}


template <typename T>
size_t Tensor<T>::write(const char* path) {

    T* buf = (T*)malloc(sizeof(T) * size_);
    cudaMemcpy(buf, data_, sizeof(T)*size_, cudaMemcpyDeviceToHost);
    FILE* fp = std::fopen(path, "wb+");
    if (!fp) {
        std::perror("File opening failed");
    }
    size_t n = std::fwrite(buf, sizeof(T), size_, fp);

    if (n != size_) {
        std::perror("File writing error");
        free(buf);
        fclose(fp);
        return 0;
    }
    else {
        //std::printf("Writing %d size of data\n", n);
        free(buf);
        fclose(fp);
        return n;
    }

}


template <typename T>
size_t Tensor<T>::calculateIndex(const std::vector<size_t>& indices) const {
    assert(indices.size() == shape_.size());
    size_t index = 0;
    for (size_t i = 0; i < indices.size(); i++) {
        assert(indices[indices.size()-i-1] < shape_[indices.size()-i-1]);
        index = index * shape_[indices.size() - i - 1] + indices[indices.size() - i - 1];
    }
    return index;
}





// Explicit template instantiation
template class Tensor<short>;
template class Tensor<float>;
template class Tensor<float2>;
template class Tensor<double>;
template class Tensor<size_t>;