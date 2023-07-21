#include "demodulation.cuh"
#include <cuda_runtime.h>

template<typename TIn, typename TOut>
Demodulation<TIn, TOut>::Demodulation(bfParameter bfPara) : bfPara_(bfPara) {
	threadsPerBlock = dim3(2, 256);
	numBlocks = dim3(ceil(double(bfPara_.sampleNum) / threadsPerBlock.x), ceil(double(bfPara_.channelNum * bfPara_.angleNum * bfPara_.frameNum) / threadsPerBlock.y));

	downMixingKernelHost = (TOut*)malloc(bfPara_.sampleNum * sizeof(TOut));
	for (int i = 0; i < bfPara_.sampleNum; i++) {
		float t = i / (bfPara_.sampleRate * bfPara_.demoFrequency);
		downMixingKernelHost[i].x = cosf(-2 * pi * bfPara_.demoFrequency * t);
		downMixingKernelHost[i].y = sinf(-2 * pi * bfPara_.demoFrequency * t);
	
	}
	
	downMixingKernelDevice->setData(downMixingKernelHost);
	//printf("%f\n", downMixingKernelHost[0].y);
	//printf("%f\n", downMixingKernelDevice->get({ 0 }).y);  

}

template<typename TIn, typename TOut>
Demodulation<TIn, TOut>::~Demodulation() {
	free(downMixingKernelHost);
	cudaFree(downMixingKernelDevice);
}

template<typename TIn, typename TOut>
void Demodulation<TIn, TOut>::downMixing(const Tensor<TIn>* RF, Tensor<TOut>* IQ) {

	kernels::Demodulation::downMixing << <numBlocks, threadsPerBlock >> > (RF->data(), IQ->data(), downMixingKernelDevice->data(), bfPara_);
}

template<typename TIn, typename TOut>
void Demodulation<TIn, TOut>::lpfiltering(const Tensor <float>* lpFilter, const Tensor<TOut>* IQDownMixed, Tensor<TOut>* IQFiltered) {
	
	kernels::Demodulation::lpfiltering << <numBlocks, threadsPerBlock >> > (lpFilter->data(), IQDownMixed->data(), IQFiltered->data(), bfPara_);

}

template<typename TIn, typename TOut>
void Demodulation<TIn, TOut>::demodulateNS200BW(const Tensor<TIn>* RF, Tensor<TOut>* IQ) {


}


template<typename TIn, typename TOut>
__global__ void kernels::Demodulation::downMixing(TIn* RF, TOut* IQ, TOut* downMixingKernel, bfParameter bfPara_) {

	size_t sampleIdx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t channelAngleFrameIdx = blockIdx.y * blockDim.y + threadIdx.y;

	size_t packetIdx = channelAngleFrameIdx / (bfPara_.angleNum * bfPara_.frameSize.framesPerPacket * bfPara_.channelNum);
	size_t channelNumIdx = (channelAngleFrameIdx - packetIdx * (bfPara_.angleNum * bfPara_.frameSize.framesPerPacket * bfPara_.channelNum)) / (bfPara_.angleNum * bfPara_.frameSize.framesPerPacket);
	size_t frameIdx = (channelAngleFrameIdx - packetIdx * (bfPara_.angleNum * bfPara_.frameSize.framesPerPacket * bfPara_.channelNum) - channelNumIdx * (bfPara_.angleNum * bfPara_.frameSize.framesPerPacket))/ bfPara_.angleNum;
	size_t angleNumIdx = channelAngleFrameIdx - packetIdx * (bfPara_.angleNum * bfPara_.frameSize.framesPerPacket * bfPara_.channelNum) - channelNumIdx * (bfPara_.angleNum * bfPara_.frameSize.framesPerPacket) - frameIdx * bfPara_.angleNum;
	size_t RFIdx;

	if (sampleIdx < bfPara_.sampleNum && channelAngleFrameIdx < bfPara_.channelNum * bfPara_.angleNum * bfPara_.frameNum) {
		RFIdx = sampleIdx + angleNumIdx * bfPara_.sampleNum + frameIdx * bfPara_.angleNum * bfPara_.sampleNum + channelNumIdx * bfPara_.frameSize.framesPerPacket * bfPara_.angleNum * bfPara_.sampleNum + packetIdx * bfPara_.channelNum * bfPara_.frameSize.framesPerPacket * bfPara_.angleNum * bfPara_.sampleNum;
		IQ[RFIdx].x = float(RF[RFIdx]) * downMixingKernel[sampleIdx].x;
		IQ[RFIdx].y = float(RF[RFIdx]) * downMixingKernel[sampleIdx].y;
	}
}

template<typename TOut>
__global__ void kernels::Demodulation::lpfiltering(float* lpFilter, TOut* IQDownMixed, TOut* IQFiltered, bfParameter bfPara_) {
	size_t sampleIdx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t channelAngleFrameIdx = blockIdx.y * blockDim.y + threadIdx.y;

	size_t packetIdx = channelAngleFrameIdx / (bfPara_.angleNum * bfPara_.frameSize.framesPerPacket * bfPara_.channelNum);
	size_t channelNumIdx = (channelAngleFrameIdx - packetIdx * (bfPara_.angleNum * bfPara_.frameSize.framesPerPacket * bfPara_.channelNum)) / (bfPara_.angleNum * bfPara_.frameSize.framesPerPacket);
	size_t frameIdx = (channelAngleFrameIdx - packetIdx * (bfPara_.angleNum * bfPara_.frameSize.framesPerPacket * bfPara_.channelNum) - channelNumIdx * (bfPara_.angleNum * bfPara_.frameSize.framesPerPacket)) / bfPara_.angleNum;
	size_t angleNumIdx = channelAngleFrameIdx - packetIdx * (bfPara_.angleNum * bfPara_.frameSize.framesPerPacket * bfPara_.channelNum) - channelNumIdx * (bfPara_.angleNum * bfPara_.frameSize.framesPerPacket) - frameIdx * bfPara_.angleNum;
	size_t RFIdx;

	if (sampleIdx < bfPara_.sampleNum && channelAngleFrameIdx < bfPara_.channelNum * bfPara_.angleNum * bfPara_.frameNum) {
		RFIdx = sampleIdx + angleNumIdx * bfPara_.sampleNum + frameIdx * bfPara_.angleNum * bfPara_.sampleNum + channelNumIdx * bfPara_.frameSize.framesPerPacket * bfPara_.angleNum * bfPara_.sampleNum + packetIdx * bfPara_.channelNum * bfPara_.frameSize.framesPerPacket * bfPara_.angleNum * bfPara_.sampleNum;
		
		if (RFIdx < bfPara_.lpLength-1) {
			for (int i = bfPara_.lpLength - RFIdx; i <= bfPara_.lpLength; i++) {
				IQFiltered[RFIdx].x += lpFilter[i-1] * IQDownMixed[i + RFIdx - bfPara_.lpLength].x;
				IQFiltered[RFIdx].y += lpFilter[i-1] * IQDownMixed[i + RFIdx - bfPara_.lpLength].y;
			}
		}
		else {
			for (int i = 0; i < bfPara_.lpLength; i++) {
				IQFiltered[RFIdx].x += lpFilter[i] * IQDownMixed[i + RFIdx - bfPara_.lpLength + 1].x;
				IQFiltered[RFIdx].y += lpFilter[i] * IQDownMixed[i + RFIdx - bfPara_.lpLength + 1].y;
			}
		}

	}

}

template<typename TIn, typename TOut>
__global__ void kernels::Demodulation::demodulateNS200BW(TIn* RF, TOut* IQ, bfParameter bfPara_) {

	size_t sampleIdx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t channelAngleFrameIdx = blockIdx.y * blockDim.y + threadIdx.y;

	size_t packetIdx = channelAngleFrameIdx / (bfPara_.angleNum * bfPara_.frameSize.framesPerPacket * bfPara_.channelNum);
	size_t channelNumIdx = (channelAngleFrameIdx - packetIdx * (bfPara_.angleNum * bfPara_.frameSize.framesPerPacket * bfPara_.channelNum)) / (bfPara_.angleNum * bfPara_.frameSize.framesPerPacket);
	size_t frameIdx = (channelAngleFrameIdx - packetIdx * (bfPara_.angleNum * bfPara_.frameSize.framesPerPacket * bfPara_.channelNum) - channelNumIdx * (bfPara_.angleNum * bfPara_.frameSize.framesPerPacket)) / bfPara_.angleNum;
	size_t angleNumIdx = channelAngleFrameIdx - packetIdx * (bfPara_.angleNum * bfPara_.frameSize.framesPerPacket * bfPara_.channelNum) - channelNumIdx * (bfPara_.angleNum * bfPara_.frameSize.framesPerPacket) - frameIdx * bfPara_.angleNum;
	size_t RFIdx;

	if (sampleIdx < bfPara_.sampleNum && channelAngleFrameIdx < bfPara_.channelNum * bfPara_.angleNum * bfPara_.frameNum) {
		RFIdx = sampleIdx + angleNumIdx * bfPara_.sampleNum + frameIdx * bfPara_.angleNum * bfPara_.sampleNum + channelNumIdx * bfPara_.frameSize.framesPerPacket * bfPara_.angleNum * bfPara_.sampleNum + packetIdx * bfPara_.channelNum * bfPara_.frameSize.framesPerPacket * bfPara_.angleNum * bfPara_.sampleNum;
		switch (sampleIdx % 4) {
		case 0:
			IQ[RFIdx].y = -float(RF[RFIdx]);
			break;
		case 1:
			IQ[RFIdx-1].x = -float(RF[RFIdx]);
			break;
		case 2:
			IQ[RFIdx].y = float(RF[RFIdx]);
			break;
		case 3:
			IQ[RFIdx-1].x = float(RF[RFIdx]);
			break;
		}

	}
}

template class Demodulation<int16_t, float2>;
