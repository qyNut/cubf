# Reconstruction of Ultrasound Bmode Image using CUDA Library
This program is designed for delay-and-sum reconstruction using the ultrasound raw RF data acquired from Verasonics Ultrasound Platform.

# Usage
- In command window, type in "BeamformingGPU.exe [RF Data Folder Path]"

 <img src="examples/CommandWin.PNG" width="800px"/>

The program will reconstruct all the RF Data saved with .dat suffix. The reconstructed IQ Data will be saved in a separate folder named with the current time stamp.

The RF Data needs to be saved in a one-dimensional vector in the order of [Samples * NumofTransmissions * NumofChannels * NumofFrames]. 

- Before running exe file, the reconstruction program needs the information of transmit and receive time delay and transducer element sensitivity. In the utils folder, genPara.m gives an example of matlab script to generate the required files.

# Example of Reconstructed Image
- Signal acquired from a Rat Brain

Reconstructed Image 

<img src="examples/GPUBF2.PNG" width="800px"/>

Reconstructed Microbubble Movie

https://github.com/qyNut/cubf/assets/136265803/793d6afc-4c84-4fa3-96af-f64cab7af0cd

# Optimization
- Transmit and receive delays are pre-calculated and loaded as look-up-table for finding the correct sample index.
- IQ demodulation are done in the raw RF channel data to reduce the data size of the follow up processing. Refer to [this paper](https://www.sciencedirect.com/science/article/pii/S0041624X20302444).









