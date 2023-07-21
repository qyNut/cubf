#ifndef KERNEL_H
#define KERNEL_H
#include <numbers>

constexpr float pi = std::numbers::pi;

struct frameSize
{
    size_t framesPerPacket;
    size_t packetNum;
};

struct channelSize
{
    size_t channelsPerPanel;
    size_t panelNum;
};

struct bfParameter
{
    float demoFrequency;
    float speedOfSound;
    float sampleRate;
    size_t sampleNum;
    size_t angleNum;
    size_t channelNum;
    frameSize frameSize;
    channelSize channelSize;
    size_t lateralDim;
    size_t axialDim;
    size_t elevationalDim;
    size_t pixelSize;
    size_t RFSize;
    size_t IQSize;
    size_t txDelaySize;
    size_t rcvDelaySize;
    size_t frameNum;
    size_t lpLength;
};



#endif // !KERNEL_H
