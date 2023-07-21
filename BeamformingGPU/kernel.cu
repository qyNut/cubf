
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "tensor.cuh"
#include "beamform.cuh"
#include "loadDat.cuh"
#include "kernel.cuh"
#include "Demodulation.cuh"
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <string>
#include <filesystem>
#include <cufft.h>

void kernelExample(Tensor<float2> tensor) {
    // Accessing and modifying tensor elements
    //tensor.set({ 0, 0, 0 }, 1.0f);
    //tensor.set({ 1, 2, 0 }, 2.0f);
    //tensor.set({ 2, 3, 1 }, 3.0f);

    // Accessing tensor elements
    printf("%f\n", tensor.get({ 600, 0, 0, 0}).x);  // Output: 1.0
    printf("%f\n", tensor.get({ 600, 0, 0, 0 }).y);  // Output: 2.0
   // printf("%f\n", tensor.get({ 2, 0, 0, 0}));  // Output: 3.0
}


int main(int argc, char* argv[]) {
    cudaError_t cudaStatus;
    
    int driverVersion;
    cudaStatus = cudaDriverGetVersion(&driverVersion);

    if (cudaStatus != cudaSuccess) {
        printf("No Cuda Driver found!\n");
        return 1;
    }
    else {
        printf("Cuda Driver Version: %d\n", driverVersion);
    }

    int deviceCount;
    cudaStatus = cudaGetDeviceCount(&deviceCount);
    if (cudaStatus != cudaSuccess) {
        printf("No Cuda Device found! Return Error Type: %d\n", cudaStatus);
        return 1;
    }
    else {
        printf("Cuda Device Count: %d\n", deviceCount);
    }

    cudaStatus = cudaSetDevice(1);
    if (cudaStatus != cudaSuccess) {
        printf("Set Cuda Device Error!\n");
        return 1;
    }
    else {
        printf("Set Cuda Device Index 0\n");
    }
    cudaDeviceReset();

#if 0  
    const std::chrono::zoned_time cur_time{ std::chrono::current_zone(), std::chrono::time_point_cast<std::chrono::seconds>(std::chrono::system_clock::now()) };
    std::string IQPath = "\\IQRecon" + std::format("{:%Y%m%d_%H%M%S}", cur_time);

    std::string RFPath = argv[1];
    std::filesystem::create_directory(RFPath + IQPath);


    bfParameter bfPara_ = { .demoFrequency = 15.625e6, .speedOfSound = 1540.0,.sampleRate = 4.0, .sampleNum = 1536,.angleNum = 5,.channelNum = 128,.frameSize = frameSize(400, 2),.lateralDim = 129,.axialDim = 244,.elevationalDim = 1,.lpLength = 41 };
    bfPara_.pixelSize = bfPara_.axialDim * bfPara_.lateralDim * bfPara_.elevationalDim;
    bfPara_.frameNum = bfPara_.frameSize.framesPerPacket * bfPara_.frameSize.packetNum;
    bfPara_.RFSize = bfPara_.sampleNum * bfPara_.angleNum * bfPara_.channelNum * bfPara_.frameNum;
    bfPara_.IQSize = bfPara_.pixelSize * bfPara_.frameNum;
    bfPara_.txDelaySize = bfPara_.pixelSize * bfPara_.angleNum;
    bfPara_.rcvDelaySize = bfPara_.pixelSize * bfPara_.channelNum;
    Beamform bf(bfPara_);

    loadDat<float> loadtxDelay("L22_14vX/txDelay.dat", bfPara_.txDelaySize);
    loadDat<float> loadrcvDelay("L22_14vX/rcvDelay.dat", bfPara_.rcvDelaySize);
    loadDat<float> loadEleSens("L22_14vX/elementSens.dat", bfPara_.rcvDelaySize);


    Tensor<float> txDelay({ bfPara_.axialDim, bfPara_.lateralDim, bfPara_.elevationalDim, bfPara_.angleNum }, loadtxDelay.data());
    Tensor<float> rcvDelay({ bfPara_.axialDim, bfPara_.lateralDim, bfPara_.elevationalDim, bfPara_.channelNum }, loadrcvDelay.data());
    Tensor<float> eleSens({ bfPara_.axialDim, bfPara_.lateralDim, bfPara_.elevationalDim, bfPara_.channelNum }, loadEleSens.data());


    Tensor<float2> IQ({ bfPara_.axialDim, bfPara_.lateralDim, bfPara_.elevationalDim, bfPara_.frameNum });
    Tensor<int16_t> RF({ bfPara_.sampleNum, bfPara_.angleNum, bfPara_.channelNum, bfPara_.frameNum });
    
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;
    for (const auto& entry : std::filesystem::directory_iterator(RFPath)) {

        if (entry.path().extension() == ".dat") {

            std::cout << "Processing " + entry.path().filename().string();

            begin = std::chrono::steady_clock::now();

            loadDat<int16_t> loadRF(entry.path().string().c_str(), bfPara_.RFSize);
            RF.loadData(loadRF.data());
            end = std::chrono::steady_clock::now();
            std::cout <<"   Read Data Time used: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]";

            begin = std::chrono::steady_clock::now();
            bf.CPW(&RF, &txDelay, &rcvDelay, &eleSens, &IQ);
            //bf.envDetect(&IQ);
            end = std::chrono::steady_clock::now();
            std::cout << "   Beamform Time used: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[us]";

            begin = std::chrono::steady_clock::now();
            std::string IQFilename = "\\IQData" + entry.path().filename().string().substr(6);
            IQ.write((RFPath + IQPath + IQFilename).c_str());
            end = std::chrono::steady_clock::now();
            std::cout << "   Write Data used: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]"<<std::endl;
        }

    }

    cudaDeviceSynchronize();
        //std::cout << entry.path().filename() << std::endl;
#endif

    // 2D Beamforming
#if 1
    bfParameter bfPara_ = { .demoFrequency = 15.625e6, .speedOfSound = 1540.0,.sampleRate = 4.0, .sampleNum = 1536,.angleNum = 5,.channelNum = 128,.frameSize = frameSize(400, 2),.channelSize = channelSize(128, 1),.lateralDim = 129,.axialDim = 244,.elevationalDim = 1,.lpLength = 41 };
    bfPara_.pixelSize = bfPara_.axialDim * bfPara_.lateralDim * bfPara_.elevationalDim;
    bfPara_.frameNum = bfPara_.frameSize.framesPerPacket * bfPara_.frameSize.packetNum;
    bfPara_.channelNum = bfPara_.channelSize.channelsPerPanel * bfPara_.channelSize.panelNum;
    bfPara_.RFSize = bfPara_.sampleNum * bfPara_.angleNum * bfPara_.channelNum * bfPara_.frameNum;
    bfPara_.IQSize = bfPara_.pixelSize * bfPara_.frameNum;
    bfPara_.txDelaySize = bfPara_.pixelSize * bfPara_.angleNum;
    bfPara_.rcvDelaySize = bfPara_.pixelSize * bfPara_.channelNum;
    printf("%d %d %d\n", bfPara_.axialDim, bfPara_.lateralDim, bfPara_.elevationalDim, bfPara_.frameNum);
    
    /*
    loadDat<int16_t> loadRF("D:/project/GPUBeamforming/BeamformingGPU/RF_2D.dat", bfPara_.RFSize);
    loadDat<float> loadLP("D:/project/GPUBeamforming/BeamformingGPU/lpFilter.dat", bfPara_.lpLength);
    Tensor<float> lpFilter({ bfPara_.lpLength }, loadLP.data());



    Demodulation<int16_t, float2> DemodulationRF(bfPara_);

    Tensor<int16_t> RF({ bfPara_.sampleNum, bfPara_.angleNum, bfPara_.channelNum, bfPara_.frameNum }, loadRF.data());
    Tensor<float2> IQ_DownMixed({ bfPara_.sampleNum, bfPara_.angleNum, bfPara_.channelNum, bfPara_.frameNum });
    Tensor<float2> IQ_Filtered({ bfPara_.sampleNum, bfPara_.angleNum, bfPara_.channelNum, bfPara_.frameNum });
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    DemodulationRF.downMixing(&RF, &IQ_DownMixed);
    DemodulationRF.lpfiltering(&lpFilter, &IQ_DownMixed, &IQ_Filtered);

    
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[us]" << std::endl;
    kernelExample(IQ_Filtered);

    */
    loadDat<int16_t> loadRF("D:/project/GPUBeamforming/BeamformingGPU/RF_2D.dat", bfPara_.RFSize);

    loadDat<float> loadtxDelay("D:/project/GPUBeamforming/BeamformingGPU/2D/txDelay.dat", bfPara_.txDelaySize);
    loadDat<float> loadrcvDelay("D:/project/GPUBeamforming/BeamformingGPU/2D/rcvDelay.dat", bfPara_.rcvDelaySize);
    loadDat<float> loadEleSens("D:/project/GPUBeamforming/BeamformingGPU/2D/elementSens.dat", bfPara_.rcvDelaySize);


    
    Tensor<int16_t> RF({ bfPara_.sampleNum, bfPara_.angleNum, bfPara_.channelNum, bfPara_.frameNum }, loadRF.data());
    Tensor<float> txDelay({ bfPara_.axialDim, bfPara_.lateralDim, bfPara_.elevationalDim, bfPara_.angleNum }, loadtxDelay.data());
    Tensor<float> rcvDelay({ bfPara_.axialDim, bfPara_.lateralDim, bfPara_.elevationalDim, bfPara_.channelNum }, loadrcvDelay.data());
    Tensor<float> eleSens({ bfPara_.axialDim, bfPara_.lateralDim, bfPara_.elevationalDim, bfPara_.channelNum }, loadEleSens.data());
    Tensor<float2> IQ({bfPara_.axialDim, bfPara_.lateralDim, bfPara_.elevationalDim, bfPara_.frameNum});


    std::vector<size_t> shape = IQ.shape();
    printf("%d %d %d\n", shape[0], shape[1], shape[2], shape[3]);

    Beamform bf(bfPara_);
    
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    bf.CPW(&RF, &txDelay, &rcvDelay ,&eleSens, &IQ);
    cudaDeviceSynchronize();

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[us]" << std::endl;


    IQ.write("D:/project/GPUBeamforming/matlab2D/IQ.dat");
    

    // Launch CUDA kernel
    //kernelExample(IQ);
    cudaDeviceSynchronize();
#endif

    // 3D Beamforming
#if 0

    /*
    size_t size = 1024ULL * 1024ULL * 1024ULL * 36ULL;

    cudaStatus = cudaDeviceSetLimit(cudaLimitMallocHeapSize, size);
    if (cudaStatus != cudaSuccess) {
        printf("Increase Cuda Malloc HeapSize Limit Error!\n");
        return 1;
    }
    else {
        printf("Set Cuda Malloc HeapSize Limit: %lld\n", size / 1024 / 1024 / 1024);
    }

    cudaStatus = cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
    if (cudaStatus != cudaSuccess) {
        printf("Get Cuda Malloc HeapSize Limit Error!\n");
        return 1;
    }
    else {
        printf("Cuda Malloc HeapSize Limit: %lld\n", size / 1024 / 1024 / 1024);
    }
    */

    size_t size = 1024ULL * 8ULL;

    cudaStatus = cudaDeviceSetLimit(cudaLimitStackSize, size);
    if (cudaStatus != cudaSuccess) {
        printf("Increase Cuda Malloc StackSize Limit Error!\n");
        return 1;
    }
    else {
        printf("Set Cuda Malloc StackSize Limit: %lld\n", size);
    }

    cudaStatus = cudaDeviceGetLimit(&size, cudaLimitStackSize);
    if (cudaStatus != cudaSuccess) {
        printf("Get Cuda Malloc HeapSize Limit Error!\n");
        return 1;
    }
    else {
        printf("Cuda Malloc StackSize Limit: %lld\n", size);
    }

    


    float x = 2.0;
    size_t y = size_t(x);
    printf("%d", y);


    size_t freeMem, totMem;

    cudaMemGetInfo(&freeMem, &totMem);
    printf("Free memory: %lld  Total memory:%lld\n", freeMem / 1024 / 1024 / 1024, totMem / 1024 / 1024 / 1024);
    
    bfParameter bfPara_ = { .demoFrequency = 7.81e6, .speedOfSound = 1540.0,.sampleRate = 2.0, .sampleNum = 384,.angleNum = 5,.frameSize = frameSize(200, 2), .channelSize = channelSize(256, 4), .lateralDim = 98,.axialDim = 128,.elevationalDim = 107,.lpLength = 41 };
    bfPara_.pixelSize = bfPara_.axialDim * bfPara_.lateralDim * bfPara_.elevationalDim;
    bfPara_.frameNum = bfPara_.frameSize.framesPerPacket * bfPara_.frameSize.packetNum;
    bfPara_.channelNum = bfPara_.channelSize.channelsPerPanel * bfPara_.channelSize.panelNum;
    bfPara_.RFSize = bfPara_.sampleNum * bfPara_.angleNum * bfPara_.channelNum * bfPara_.frameNum;
    bfPara_.IQSize = bfPara_.pixelSize * bfPara_.frameNum;
    bfPara_.txDelaySize = bfPara_.pixelSize * bfPara_.angleNum;
    bfPara_.rcvDelaySize = bfPara_.pixelSize * bfPara_.channelNum;
    printf("%d %d %d\n", bfPara_.axialDim, bfPara_.lateralDim, bfPara_.elevationalDim, bfPara_.frameNum);

    
    loadDat<int16_t> *loadRF = new loadDat<int16_t>("D:/project/GPUBeamforming/BeamformingGPU/RF_3D.dat", bfPara_.RFSize);
    Tensor<int16_t> *RF = new Tensor<int16_t>({ bfPara_.sampleNum, bfPara_.angleNum, bfPara_.channelSize.panelNum, bfPara_.frameSize.framesPerPacket, bfPara_.channelSize.channelsPerPanel, bfPara_.frameSize.packetNum }, loadRF->data());

    loadDat<float> *loadtxDelay = new loadDat<float>("D:/project/GPUBeamforming/BeamformingGPU/3D/txDelay.dat", bfPara_.txDelaySize);
    Tensor<float> *txDelay = new Tensor<float>({ bfPara_.axialDim, bfPara_.lateralDim, bfPara_.elevationalDim, bfPara_.angleNum }, loadtxDelay->data());

    loadDat<float> *loadrcvDelay = new loadDat<float>("D:/project/GPUBeamforming/BeamformingGPU/3D/rcvDelay.dat", bfPara_.rcvDelaySize);
    Tensor<float> *rcvDelay = new Tensor<float>({ bfPara_.axialDim, bfPara_.lateralDim, bfPara_.elevationalDim, bfPara_.channelNum }, loadrcvDelay->data());

    loadDat<float> *loadEleSens = new loadDat<float>("D:/project/GPUBeamforming/BeamformingGPU/3D/elementSens.dat", bfPara_.rcvDelaySize);
    Tensor<float> *eleSens = new Tensor<float>({ bfPara_.axialDim, bfPara_.lateralDim, bfPara_.elevationalDim, bfPara_.channelNum }, loadEleSens->data());

    loadDat<size_t> *loadApertureInfo = new loadDat<size_t>("D:/project/GPUBeamforming/BeamformingGPU/3D/apertureInfo.dat", bfPara_.channelNum);
    Tensor<size_t> *apertureInfo = new Tensor<size_t>({ bfPara_.channelNum }, loadApertureInfo->data());

    Tensor<float2> *IQ = new Tensor<float2>({ bfPara_.axialDim, bfPara_.lateralDim, bfPara_.elevationalDim, bfPara_.frameNum });
    
    std::vector<size_t> shape = IQ->shape();
    printf("%d %d %d %d\n", shape[0], shape[1], shape[2], shape[3]);

    Beamform bf(bfPara_);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    bf.CPW3D(RF, txDelay, rcvDelay, eleSens, apertureInfo, IQ);
    cudaDeviceSynchronize();

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[us]" << std::endl;

    
    float2* IQHost = (float2*)malloc(bfPara_.axialDim * bfPara_.lateralDim * bfPara_.elevationalDim * bfPara_.frameNum * sizeof(float2));
    cudaMemcpy(IQHost, IQ->data(), bfPara_.axialDim* bfPara_.lateralDim* bfPara_.elevationalDim* bfPara_.frameNum * sizeof(float2), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 1024; i++) {
        printf("%f   %f   %f\n", IQHost[i].x, IQHost[i].y, IQHost[i].x / IQHost[i].y);
    }
    


    //IQ->write("D:/project/GPUBeamforming/matlab2D/IQ_3D.dat");
    

    // Launch CUDA kernel
    //kernelExample(IQ);

    cudaDeviceSynchronize();
#endif

#if 0
    cufftHandle plan;
    cufftComplex* dataHost;

    dataHost = (cufftComplex*)malloc(sizeof(cufftComplex) * 3 * 3);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            dataHost[j + i * 3].x = j + i * 3;
            dataHost[j + i * 3].y = 8 - j - i * 3;
            printf("%1.0f+%1.0fj   ", dataHost[j + i * 3].x, dataHost[j + i * 3].y);
        }
        printf("\n");
    }


    cufftComplex* dataDevice;
    cudaMalloc((void**)&dataDevice, sizeof(cufftComplex) * 3 * 3);
    cudaMemcpy(dataDevice, dataHost, sizeof(cufftComplex) * 3 * 3, cudaMemcpyHostToDevice);


    cufftPlan1d(&plan, 3, CUFFT_C2C, 3);
    cufftExecC2C(plan, dataDevice, dataDevice, CUFFT_FORWARD);
    cufftExecC2C(plan, dataDevice, dataDevice, CUFFT_INVERSE);
    cudaMemcpy(dataHost, dataDevice, sizeof(cufftComplex) * 3 * 3, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%1.3f+%1.3fj   ", dataHost[j + i * 3].x, dataHost[j + i * 3].y);
        }
        printf("\n");
    }
#endif

    return 0;
}

#if 0
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
#endif