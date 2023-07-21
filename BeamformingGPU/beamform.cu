#include "beamform.cuh"
#include "tensor.cuh"
#include "kernel.cuh"
#include <cassert>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cufft.h>

Beamform::Beamform(bfParameter bfPara): bfPara_(bfPara){
	
	threadsPerBlock = dim3(1024, 1);
	//printf("%d %d %d\n", threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);
	numBlocks = dim3(ceil(double(bfPara_.pixelSize) / threadsPerBlock.x), ceil(double(bfPara_.frameNum) / threadsPerBlock.y));
	printf("%d %d\n", numBlocks.x, numBlocks.y);
}
Beamform::~Beamform(){}

void Beamform::CPW(const Tensor<int16_t>* RF, const Tensor<float>* delay, Tensor<float2>* IQ) {

	size_t axial = bfPara_.axialDim;
	size_t lateral = bfPara_.lateralDim;
	size_t frame = bfPara_.frameNum;
	
	size_t axialIQ = IQ->shape()[0];
	size_t lateralIQ = IQ->shape()[1];
	size_t frameIQ = IQ->shape()[2];
	
	kernels::Beamform::DAS << <numBlocks, threadsPerBlock >> > (RF->data(), delay->data(), IQ->data(), bfPara_);

}

void Beamform::CPW(const Tensor<int16_t>* RF, const Tensor<float>* txDelay, const Tensor<float>* rcvDelay, const Tensor<float>* eleSens, Tensor<float2>* IQ) {

	size_t axial = bfPara_.axialDim;
	size_t lateral = bfPara_.lateralDim;
	size_t frame = bfPara_.frameNum;

	size_t axialIQ = IQ->shape()[0];
	size_t lateralIQ = IQ->shape()[1];
	size_t frameIQ = IQ->shape()[2];

	kernels::Beamform::DAS << <numBlocks, threadsPerBlock >> > (RF->data(), txDelay->data(), rcvDelay->data(), eleSens->data(), IQ->data(), bfPara_);

}


void Beamform::CPW3D(const Tensor<int16_t>* RF, const Tensor<float>* txDelay, const Tensor<float>* rcvDelay, const Tensor<float>* eleSens, const Tensor<size_t>* aperture, Tensor<float2>* IQ) {

	size_t axial = bfPara_.axialDim;
	size_t lateral = bfPara_.lateralDim;
	size_t frame = bfPara_.frameNum;

	size_t axialIQ = IQ->shape()[0];
	size_t lateralIQ = IQ->shape()[1];
	size_t frameIQ = IQ->shape()[2];

	kernels::Beamform::DAS3D << <numBlocks, threadsPerBlock >> > (RF->data(), txDelay->data(), rcvDelay->data(), eleSens->data(), aperture->data(), IQ->data(), bfPara_);
	//kernels::Beamform::DAS3D << <dim3(200, 200), threadsPerBlock >> > (RF->data(), txDelay->data(), rcvDelay->data(), eleSens->data(), aperture->data(), IQ->data(), bfPara_);
}


void Beamform::envDetect(Tensor<float2>* IQ) {

	cufftHandle plan;
	cufftPlan1d(&plan, IQ->shape()[0], CUFFT_C2C, IQ->shape()[1]*IQ->shape()[2]*IQ->shape()[3]);
	cufftExecC2C(plan, IQ->data(), IQ->data(), CUFFT_FORWARD);

	//kernels::Beamform::hilbertKernelMultiply << <numBlocks, threadsPerBlock >> > (IQ->data(), bfPara_);
	cufftExecC2C(plan, IQ->data(), IQ->data(), CUFFT_INVERSE);
	kernels::Beamform::abs << <numBlocks, threadsPerBlock >> > (IQ->data(), bfPara_);

}



__global__ void kernels::Beamform::DAS(int16_t* RF, float* delay, float2* IQ, bfParameter bfPara_) {
	size_t pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t frameNumIdx = blockIdx.y * blockDim.y + threadIdx.y;
	size_t idx;

	size_t elevationalDimIdx = pixelIdx / (bfPara_.axialDim * bfPara_.lateralDim);
	size_t lateralDimIdx = (pixelIdx - elevationalDimIdx * (bfPara_.axialDim * bfPara_.lateralDim)) / bfPara_.axialDim;
	size_t axialDimIdx = pixelIdx - elevationalDimIdx * (bfPara_.axialDim * bfPara_.lateralDim) - lateralDimIdx * bfPara_.axialDim;
	
	size_t packetIdx = frameNumIdx / bfPara_.frameSize.framesPerPacket;
	size_t frameIdx = frameNumIdx - packetIdx * bfPara_.frameSize.framesPerPacket;

	if (pixelIdx < bfPara_.pixelSize && frameNumIdx < bfPara_.frameNum) {

		idx = pixelIdx + frameNumIdx * bfPara_.pixelSize;
		for (size_t channelNumIdx = 0; channelNumIdx < bfPara_.channelNum; channelNumIdx++) {
			for (size_t angleNumIdx = 0; angleNumIdx < bfPara_.angleNum; angleNumIdx++) {
				float delayIdx = delay[pixelIdx + channelNumIdx * bfPara_.pixelSize + angleNumIdx * bfPara_.pixelSize * bfPara_.channelNum];
				size_t delayTop = size_t(ceilf(delayIdx)) - 1;
				size_t delayBot = size_t(floorf(delayIdx)) - 1;
				float interpRate = delayIdx - floorf(delayIdx);

				size_t RFIdx = angleNumIdx * bfPara_.sampleNum + frameIdx * bfPara_.angleNum * bfPara_.sampleNum + channelNumIdx * bfPara_.frameSize.framesPerPacket * bfPara_.angleNum * bfPara_.sampleNum + packetIdx * bfPara_.channelNum * bfPara_.frameSize.framesPerPacket * bfPara_.angleNum * bfPara_.sampleNum;
				float RFBot = float(RF[delayBot + RFIdx]);
				float RFTop = float(RF[delayTop + RFIdx]);

				IQ[idx].x += RFBot + interpRate * (RFTop - RFBot);

			}
		}

	}
}

__global__ void kernels::Beamform::DAS(int16_t* RF, float* txDelay, float* rcvDelay, float* eleSens, float2* IQ, bfParameter bfPara_) {
	size_t pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t frameNumIdx = blockIdx.y * blockDim.y + threadIdx.y;
	size_t idx;

	size_t elevationalDimIdx = pixelIdx / (bfPara_.axialDim * bfPara_.lateralDim);
	size_t lateralDimIdx = (pixelIdx - elevationalDimIdx * (bfPara_.axialDim * bfPara_.lateralDim)) / bfPara_.axialDim;
	size_t axialDimIdx = pixelIdx - elevationalDimIdx * (bfPara_.axialDim * bfPara_.lateralDim) - lateralDimIdx * bfPara_.axialDim;

	size_t packetIdx = frameNumIdx / bfPara_.frameSize.framesPerPacket;
	size_t frameIdx = frameNumIdx - packetIdx * bfPara_.frameSize.framesPerPacket;

	if (pixelIdx < bfPara_.pixelSize && frameNumIdx < bfPara_.frameNum) {

		idx = pixelIdx + frameNumIdx * bfPara_.pixelSize;
		for (size_t channelNumIdx = 0; channelNumIdx < bfPara_.channelNum; channelNumIdx++) {
			for (size_t angleNumIdx = 0; angleNumIdx < bfPara_.angleNum; angleNumIdx++) {
				size_t RFIdx = angleNumIdx * bfPara_.sampleNum + frameIdx * bfPara_.angleNum * bfPara_.sampleNum + channelNumIdx * bfPara_.frameSize.framesPerPacket * bfPara_.angleNum * bfPara_.sampleNum + packetIdx * bfPara_.channelNum * bfPara_.frameSize.framesPerPacket * bfPara_.angleNum * bfPara_.sampleNum;
				float txDelayIdx = txDelay[pixelIdx + angleNumIdx * bfPara_.pixelSize];
				float rcvDelayIdx = rcvDelay[pixelIdx + channelNumIdx * bfPara_.pixelSize];

				float delayIdxI = txDelayIdx + rcvDelayIdx;
				size_t delayTopI = size_t(ceilf(delayIdxI)) - 1;
				size_t delayBotI = size_t(floorf(delayIdxI)) - 1;
				float interpRateI = delayIdxI - floorf(delayIdxI);
				float RFBotI = float(RF[delayBotI + RFIdx]);
				float RFTopI = float(RF[delayTopI + RFIdx]);
				IQ[idx].x += (RFBotI + interpRateI * (RFTopI - RFBotI)) * eleSens[pixelIdx + channelNumIdx * bfPara_.pixelSize];


				float delayIdxQ = txDelayIdx + rcvDelayIdx - bfPara_.sampleRate / 4;
				size_t delayTopQ = size_t(ceilf(delayIdxQ)) - 1;
				size_t delayBotQ = size_t(floorf(delayIdxQ)) - 1;
				float interpRateQ = delayIdxQ - floorf(delayIdxQ);
				float RFBotQ = float(RF[delayBotQ + RFIdx]);
				float RFTopQ = float(RF[delayTopQ + RFIdx]);
				IQ[idx].y += (RFBotQ + interpRateQ * (RFTopQ - RFBotQ)) * eleSens[pixelIdx + channelNumIdx * bfPara_.pixelSize];
			}
		}

	}
}


__global__ void kernels::Beamform::DAS3D(int16_t* RF, float* txDelay, float* rcvDelay, float* eleSens, size_t* aperture, float2* IQ, bfParameter bfPara_) {
	size_t pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t frameNumIdx = blockIdx.y * blockDim.y + threadIdx.y;
	size_t idx;

	size_t elevationalDimIdx = pixelIdx / (bfPara_.axialDim * bfPara_.lateralDim);
	size_t lateralDimIdx = (pixelIdx - elevationalDimIdx * (bfPara_.axialDim * bfPara_.lateralDim)) / bfPara_.axialDim;
	size_t axialDimIdx = pixelIdx - elevationalDimIdx * (bfPara_.axialDim * bfPara_.lateralDim) - lateralDimIdx * bfPara_.axialDim;

	size_t packetIdx = frameNumIdx / bfPara_.frameSize.framesPerPacket;
	size_t frameIdx = frameNumIdx - packetIdx * bfPara_.frameSize.framesPerPacket;
	
		
	if (pixelIdx < bfPara_.pixelSize && frameNumIdx < bfPara_.frameNum) {

		idx = pixelIdx + frameNumIdx * bfPara_.pixelSize;
		IQ[idx].x = float(idx / 12.0);
		
		/*
		for (size_t panelIdx = 0; panelIdx < bfPara_.channelSize.panelNum; panelIdx++) {
			for (size_t channelNumIdx = 0; channelNumIdx < bfPara_.channelSize.channelsPerPanel; channelNumIdx++) {
				for (size_t angleNumIdx = 0; angleNumIdx < bfPara_.angleNum; angleNumIdx++) {

					
					
					size_t channelIdx = channelNumIdx + panelIdx * bfPara_.channelSize.channelsPerPanel;
					
					size_t apertureIdx = aperture[channelIdx];
					
					
					//IQ[channelIdx].x = float(channelIdx / 12.0);
					//IQ[channelIdx].y = float(bfPara_.channelSize.channelsPerPanel);
					
					size_t aperturePanelIdx = apertureIdx / bfPara_.channelSize.channelsPerPanel;

					
					
					size_t apertureChannelIdx = apertureIdx - aperturePanelIdx * bfPara_.channelSize.channelsPerPanel;
					
					
					
					size_t RFIdx = angleNumIdx * bfPara_.sampleNum 
						+ aperturePanelIdx * bfPara_.sampleNum * bfPara_.angleNum
						+ frameIdx * bfPara_.angleNum * bfPara_.sampleNum * bfPara_.channelSize.panelNum 
						+ apertureChannelIdx * bfPara_.frameSize.framesPerPacket * bfPara_.angleNum * bfPara_.sampleNum * bfPara_.channelSize.panelNum
						+ packetIdx * bfPara_.channelNum * bfPara_.frameSize.framesPerPacket * bfPara_.angleNum * bfPara_.sampleNum * bfPara_.channelSize.panelNum;

					float txDelayIdx = txDelay[pixelIdx + angleNumIdx * bfPara_.pixelSize];
					float rcvDelayIdx = rcvDelay[pixelIdx + channelNumIdx * bfPara_.pixelSize + panelIdx * bfPara_.channelSize.channelsPerPanel * bfPara_.pixelSize];
					
					
					
					
					float delayIdxI = txDelayIdx + rcvDelayIdx;
					size_t delayTopI = size_t(ceilf(delayIdxI)) - 1;
					size_t delayBotI = size_t(floorf(delayIdxI)) - 1;
					float interpRateI = delayIdxI - floorf(delayIdxI);
					float RFBotI = float(RF[delayBotI + RFIdx]);
					float RFTopI = float(RF[delayTopI + RFIdx]);
					IQ[idx].x += (RFBotI + interpRateI * (RFTopI - RFBotI)) * eleSens[pixelIdx + channelNumIdx * bfPara_.pixelSize + panelIdx * bfPara_.channelSize.channelsPerPanel * bfPara_.pixelSize];


					float delayIdxQ = txDelayIdx + rcvDelayIdx - bfPara_.sampleRate / 4;
					size_t delayTopQ = size_t(ceilf(delayIdxQ)) - 1;
					size_t delayBotQ = size_t(floorf(delayIdxQ)) - 1;
					float interpRateQ = delayIdxQ - floorf(delayIdxQ);
					float RFBotQ = float(RF[delayBotQ + RFIdx]);
					float RFTopQ = float(RF[delayTopQ + RFIdx]);
					IQ[idx].y += (RFBotQ + interpRateQ * (RFTopQ - RFBotQ)) * eleSens[pixelIdx + channelNumIdx * bfPara_.pixelSize + panelIdx * bfPara_.channelSize.channelsPerPanel * bfPara_.pixelSize];
					
					

				}
			}
		}
		*/
	}

	
}


__global__ void kernels::Beamform::hilbertKernelMultiply(float2* IQ, bfParameter bfPara_) {
	size_t pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t frameNumIdx = blockIdx.y * blockDim.y + threadIdx.y;
	size_t idx;

	size_t elevationalDimIdx = pixelIdx / (bfPara_.axialDim * bfPara_.lateralDim);
	size_t lateralDimIdx = (pixelIdx - elevationalDimIdx * (bfPara_.axialDim * bfPara_.lateralDim)) / bfPara_.axialDim;
	size_t axialDimIdx = pixelIdx - elevationalDimIdx * (bfPara_.axialDim * bfPara_.lateralDim) - lateralDimIdx * bfPara_.axialDim;

	if (pixelIdx < bfPara_.pixelSize && frameNumIdx < bfPara_.frameNum) {
	
		idx = pixelIdx + frameNumIdx * bfPara_.pixelSize;

		if (axialDimIdx > 0 && axialDimIdx < bfPara_.axialDim / 2) {
			IQ[idx].x *= 2;
			IQ[idx].y *= 2;
		}
		if (axialDimIdx > bfPara_.axialDim / 2) {
			IQ[idx].x = 0;
			IQ[idx].y = 0;
		}

	}
}

__global__ void kernels::Beamform::abs(float2* IQ, bfParameter bfPara_) {
	size_t pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t frameNumIdx = blockIdx.y * blockDim.y + threadIdx.y;
	size_t idx;

	if (pixelIdx < bfPara_.pixelSize && frameNumIdx < bfPara_.frameNum) {

		idx = pixelIdx + frameNumIdx * bfPara_.pixelSize;

		IQ[idx].x = sqrtf(powf(IQ[idx].x / bfPara_.axialDim, 2.0) + powf(IQ[idx].y / bfPara_.axialDim, 2.0));



	}
}