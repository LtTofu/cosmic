// hashburner.cuh : header for the CUDA solver  (hashburner.cu)
// -2021 LtTofu
#pragma once
#ifndef HASHBURNER_CUH
#define HASHBURNER_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
//#include <thread>
#include <sys/timeb.h>	// time.h
//#include <cinttypes>	// inttypes.h
//#include <chrono>

//extern bool b_cudaDevices[CUDA_MAX_DEVICES];
//extern bool b_genericSolvers[MAX_SOLVERS];

#define PASS_MIDSTATE_TARGET_TO_KERNEL 0	// [TODO]. They should be automatically cached, if stored in constant memory

//class genericSolver;
#ifndef GPUSOLVER
//#include "generic_solver.hpp"	//[new]<--


#endif

#include "defs.hpp"
// for h_solutions/d_solutions:
#define CUDA_MAX_DEVICES 19					//<--- [OLD]

constexpr auto MAX_SOLUTIONS = 128;			//was max of 256 sol'ns = 2048 bytes (1D array size). 128 sol'ns = 1024 bytes.	*128 for debugging only!!*
constexpr auto SOLUTIONS_COUNT_SIZE = sizeof(uint32_t);	//4	bytes			// [CHECK THIS] <--
//constexpr size_t SOLUTIONS_SIZE = MAX_SOLUTIONS * sizeof(uint64_t);		// in bytes
#define SOLUTIONS_SIZE MAX_SOLUTIONS * sizeof(uint64_t)						// in bytes

//constexpr uint64_t COUNT_OFFSET = 35000000000000;
//constexpr auto TPB50 = 1024u;				// compute_50 and up
//constexpr auto TPB35 = 384u;				// compute_35 and down
//constexpr auto NPT = 2;

//extern __constant__ /*__align__(32)*/ uint64_t d_midstate[25];	// device mid-state input  (200 bytes or 1600 bits)
//extern __constant__ /*__align__(32)*/ uint64_t d_target/*[1]*/;	// device target. most-significant 64 bits


// === macros ===
//LAUNCH_BOUNDS __launch_bounds__(TPB50, 1)				[compute capability 5.0 and up]
//LAUNCH_BOUNDS __launch_bounds__(TPB35, 2)				[old. cc 3.5 and lower, unsupported.]
#define ROTL64(x, y) (((x) << (y)) ^ ((x) >> (64 - (y))))   // 64-bit rotate x left by y
#define ROTR64(x, y) (((x) >> (y)) ^ ((x) << (64 - (y))))   // 64-bit rotate x right by y


extern uint64_t	cnt[CUDA_MAX_DEVICES];	// for striding across search space. counter becomes `nonce` input to kernel.
										// TODO: multiple devices now implemented, independent or unified counter?

// === externals ===	(double-check for duplicates!)
//extern unsigned int gCudaDeviceIntensities[CUDA_MAX_DEVICES];

extern uint64_t gNum_SolutionCount[CUDA_MAX_DEVICES];
extern uint64_t gNum_InvalidSolutionCount[CUDA_MAX_DEVICES];
extern double gNum_Hashrate[CUDA_MAX_DEVICES];
extern ushort gApplicationStatus;
//extern double cuda_solvetime[CUDA_MAX_DEVICES];
//extern uint8_t solution[32];

extern int gCudaDevicePciBusIDs[CUDA_MAX_DEVICES];		// see WatchQat.cpp
extern unsigned int gCudaDevicesStarted;				// CosmicWind.h
extern std::string gpuName[CUDA_MAX_DEVICES];			// CUDA device names (Cosmic.cpp)

extern ushort gVerbosity;
extern bool gCudaSolving;								// CosmicWind.h
//extern bool	gCudaDeviceEnabled[CUDA_MAX_DEVICES];
extern bool	gSolving[CUDA_MAX_DEVICES];

//extern uint32_t gCuda_Threads[CUDA_MAX_DEVICES];		// based on intensity setting
extern uint8_t gCuda_Engine[CUDA_MAX_DEVICES];			// hashing function selection

extern bool gNetPause;									// pause if network unavailable

// === function prototypes ===
std::string Cuda_GetDeviceNames(int devIndex);
std::string Cuda_GetDeviceBusInfo(int devIndex);
void Cuda_StoreDevicePciBusIDs(void);
int Cuda_GetNumberOfDevices(void);
// condense into cudaDevice class ^

//void Cuda_UpdateDeviceIntensity(const unsigned short deviceID);
//void Cuda_ResetHashrateCalc(const unsigned short deviceID);		//see	genericSolver::ResetHashrateCalc() <---
void Cuda_PrintDeviceCounters();	// Debug
int Cuda_GetNumberOfDevices(void);

void domesg_verb(const std::string& to_print, const bool make_event, const unsigned short req_verbosity);
void print_bytes(const uint8_t inArray[], const uint8_t len, const std::string desc);	 // net_pool.cpp
void AddEventToQueue(const std::string theText);							 // CosmicWind.cpp

//kernLaunchResult cudaDevice::launch_kernels(/*const unsigned short dev_no*/, uint64_t* host_ptr /*, uint64_t* device_ptr*/);
//extern int gTxView_WaitForTx;							// -1 = not waiting.  >=0 = a solution item in TXview


// #include "cuda_device.hpp"
// extern std::vector<cudaDevice*> CudaDevices;

bool Cuda_Call(const cudaError_t apiCallResult, const std::string theTask, const int deviceID /*, const bool trivial*/);

#endif	//HASHBURNER_CUH
