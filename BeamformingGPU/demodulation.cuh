#ifndef DEMODULATION_H
#define DEMODULATION_H

#include "kernel.cuh"
#include "tensor.cuh"
#include "cuda_runtime.h"
#include <stdio.h>

template<typename TIn, typename TOut>
class Demodulation {

private:
	bfParameter bfPara_;
	dim3 threadsPerBlock;
	dim3 numBlocks;
	TOut* downMixingKernelHost;
	Tensor<TOut>* downMixingKernelDevice = new Tensor<TOut>({bfPara_.sampleNum});


public:
	Demodulation(bfParameter bfPara);
	~Demodulation();

	void downMixing(const Tensor<TIn>* RF, Tensor<TOut>* IQ);
	void lpfiltering(const Tensor <float>* lpFilter, const Tensor<TOut>* IQDownMixed, Tensor<TOut>* IQFiltered);
	void demodulateNS200BW(const Tensor<TIn>* RF, Tensor<TOut>* IQ);

};

namespace kernels::Demodulation {
	template<typename TIn, typename TOut>
	__global__ void downMixing(TIn* RF, TOut* IQ, TOut* downMixingKernel, bfParameter bfPara_);

	template<typename TOut>
	__global__ void lpfiltering(float* lpFilter, TOut* IQDownMixed, TOut* IQFiltered, bfParameter bfPara_);

	template<typename TIn, typename TOut>
	__global__ void demodulateNS200BW(TIn* RF, TOut* IQ, bfParameter bfPara_);


}



#endif // !DEMODULATION_H
