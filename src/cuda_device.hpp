
#pragma once
#ifndef CUDADEVICE
#define CUDADEVICE

#include "defs.hpp"			//for: extern unsigned short gVerbosity and #definitions: DEBUGMODE, HIGHVERBOSITY.
#include "hashburner.cuh"	//<---

#include <cuda.h>
#include <cuda_runtime.h>

//#include "generic_solver.hpp"
class genericSolver;


class cudaDevice
{
public:
	cudaDevice::cudaDevice(const int cuda_device_no, genericSolver* for_solver);
	cudaDevice::~cudaDevice();

public:
	int dev_no;							// as numbered by CUDA
	genericSolver* solver;				// genericSolver-class object associated with device

	DeviceStatus status;
	bool is_initialized;
	bool is_cleaned_up;

	cudaDeviceProp deviceProperties;	// struct cudaDeviceProp ?

// === host/device i/o ===
	uint64_t* h_solutions;				// host-side array			uint64_t** h_solutions[MAX_SOLUTIONS];
	uint32_t* h_solutions_count;		// host-side				unsigned short** h_solutions_count;

	uint64_t* d_solutions;			//	device side: points to 
	uint32_t* d_solutions_count;		//	device side: points to 

private:
	//grid, block, threads ? <--- see old gCuda_... vars <--- [WIP] / [TODO].
//	struct timeb tStart, tEnd;	//moved to genericSolver class

public:
	bool cudaDevice::find_solutions (unsigned short* solns_found);	//hashburner.cu	
	bool cudaDevice::Init_Device (void);
	bool cudaDevice::Allocate (void);
	bool cudaDevice::Clean_Up (void);		//combine these [WIP]
//	bool cudaDevice::update_cuda_device ( ARGS? );	//<--- OR , pass the midstate and target into the kernel.
													//		If no performance hit, obsolete this. [TODO] / [WIP]

private:
	bool cudaDevice::Check (void);	//checks host/device i/o pointers. [WIP]
	bool cudaDevice::SetStatus(const DeviceStatus new_status);
	bool cudaDevice::enqueue_solutions (/*unsigned short* solns_found*/);
	bool cudaDevice::clear_solutions (void);
};

// ==== cudaDevice implementation in generic_solver.cpp and hashburner.cu ====


#include <vector>
extern std::vector<cudaDevice*> CudaDevices;

extern uint8_t	gCuda_Engine[CUDA_MAX_DEVICES];		//<-- [OLD]
extern uint64_t	cnt[CUDA_MAX_DEVICES];				//<--

#else
#pragma message( "not including " __FILE__ ", CUDADEVICE is already defined." )
#endif	//CUDADEVICE
