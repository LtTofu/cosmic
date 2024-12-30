#pragma once
// GPU Solver - GpuSolver.hpp
// 2020 LtTofu
#ifndef GENERICSOLVER
#define GENERICSOLVER

#include "defs.hpp"
#include "types.hpp"
#define ushort unsigned short
extern ushort gVerbosity;

#include <stdint.h>
#include <sys/timeb.h>
#include <mutex>	//New <----
//#include <loguru/loguru.hpp>
//#include "network.hpp"
//#include "hashburner.cuh"

//skip includes if this file is being included from CUDA files:
#ifdef __CUDA_CC__
//#include "network.hpp"
//#include <stdint.h>

// #include <driver_types.h>				//<--redundant?
//#include <cuda.h>
//#include <cuda_runtime.h>					//<--
//#include <device_launch_parameters.h>
#endif

bool allocate_gpusolvers_cuda(const ushort howmany);		//elsewhere

class cudaDevice;
// note: there is still some overlap between functionalities of gpusolver and cudadevice.

class genericSolver
{
public:
	bool enabled;
//	bool thread_running ?
	uint8_t solver_no;				// as shown in COSMiC's Device List, from 0.
	ushort api_device_no;			// as numbered by cuda/etc. can also be cpu thread#		// [WIP]
	DeviceType device_type;

	cudaDevice *cuda_device;		//the cuda device this genericSolver uses, allocated in constructor
//	otherDevice other_device;		//other device types [todo].
//	gpuMonDevice gpumon_device;		// [todo].

// === status and async ===
	SolverStatus solver_status;
	DeviceStatus device_status;
	PauseType pause;
//	bool resuming;					// redundant?

// === device type-specific (just cuda, for now) ===
	std::string gpuName;
//	unsigned short cuda_pause;				// if paused, the reason	(enum?)

	unsigned int	intensity;
	uint32_t		threads;
	uint64_t		hash_count;				// total hashes (this solve)
	double			hash_rate;				// Megahashes/sec. (current)
	struct timeb	tStart, tEnd;			// mining loop timing, hashrate calc.
	double			solve_time;				//<-- moved from cudaDevice class

// pointer to the params in the Network BGworker? or global?	<--- [wip]
	bool new_params_available;		// "params_changing"	//solver should -Wait- til it's supplied with valid params by net thread [WIP]. <---
	bool params_changing;			// <---

// mining parameters and actual kernel input to hash (only missing the uint64 counter 'nonce')
// ---> get mutex first, or otherwise ensure the parms won't change while they are being read! <---
	uint8_t initial_message[84];	//<--- INITIALIZE THIS IN CTOR
	uint8_t hash_prefix[52];		//<--- INITIALIZE THIS IN CTOR
	//
	uint64_t midstate[25];			//<----	
	uint64_t target;				//<-----
	uint64_t difficulty;				//<----- INITIALIZE THIS IN CONSTRUCTOR <-----------
	//
	uint64_t valid_solutions;		//<-- init!		//passed CPU verification
	uint64_t invalid_solutions;		//<-- init!		[idea]: also could count known stale sol'ns separate from invalids (failed verification).
//	uint64_t stale_solutions;		//<-- init!
//	uint64_t solutions_found;		//<-- init!		//solutions found in total, valid or otherwise

	genericSolver(const DeviceType deviceType, const ushort deviceID, const ushort solverNo /*, const miningParameters* init_params*/);
	~genericSolver();

public:
	int genericSolver::SpawnThread(void);
	bool genericSolver::GetSolverStatus (double* hashrate, uint64_t* hashcount);	// for the UI
	bool genericSolver::InitGenericDevice(void);
	bool genericSolver::Start(void/*miningParameters* init_params*/);
	//bool genericSolver::EnqueueSolutions_CUDA(/*const unsigned short howmany*/ void);	//<--combine these
	bool genericSolver::ClearSolutions(void);											//<--
	void genericSolver::ResetHashrateCalc(void);	// was private
	void genericSolver::SetIntensity();		//kludge

private:
	bool genericSolver::SendToDevice(void);
	bool genericSolver::NewParameters(const miningParameters* new_params);
	ushort genericSolver::CudaSolve (void /*miningParameters params*/);					// [WIP].
	bool genericSolver::Shutdown (void);	// WIP (anything else needed on device-specific shutdown? vars reset?)

//	void genericSolver::SetTime (const CudaSolverTimes which);
};


#include <vector>
extern std::vector<genericSolver*> Solvers;

#else
#pragma message( "not including " __FILE__ ", GENERICSOLVER is already included." )
#endif	//GENERICSOLVER


// __FILE__ __TIMESTAMP__ <--- for `pragma message` strings

// [old stuff]:
// [todo]: solution/invalid count etc. ^		<---
// uint8_t cuda_computeCapability;
