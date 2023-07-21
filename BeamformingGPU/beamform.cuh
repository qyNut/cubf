#ifndef BEAMFORM_H
#define BEAMFORM_H

#include "tensor.cuh"
#include "kernel.cuh"
#include "cuda_runtime.h"
#include "kernel.cuh"

class Beamform {
private:
    dim3 threadsPerBlock;
    dim3 numBlocks;
    bfParameter bfPara_;

public:
    Beamform(bfParameter bfPara_);
    ~Beamform();
    void CPW(const Tensor<int16_t>* RF, const Tensor<float>* delay, Tensor<float2>* IQ);
    void CPW(const Tensor<int16_t>* RF, const Tensor<float>* txDelay, const Tensor<float>* rcvDelay, const Tensor<float>* eleSens, Tensor<float2>* IQ);

    void CPW3D(const Tensor<int16_t>* RF, const Tensor<float>* txDelay, const Tensor<float>* rcvDelay, const Tensor<float>* eleSens, const Tensor<size_t>* aperture, Tensor<float2>* IQ);
    void envDetect(Tensor<float2>* IQ);
};


namespace kernels::Beamform {

    __global__ void DAS(int16_t* RF, float* delay, float2* IQ, bfParameter bfPara_);
    __global__ void DAS(int16_t* RF, float* txDelay, float* rcvDelay, float* eleSens, float2* IQ, bfParameter bfPara_);
    __global__ void DAS3D(int16_t* RF, float* txDelay, float* rcvDelay, float* eleSens, size_t* aperture, float2* IQ, bfParameter bfPara_);
    __global__ void hilbertKernelMultiply(float2* IQ, bfParameter bfPara_);
    __global__ void abs(float2* IQ, bfParameter bfPara_);
}

#endif 