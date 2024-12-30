//#pragma once

#include <string>
#include <inttypes.h>	//only for debugging
/*_inline*/ void domesg_verb(const std::string& to_print, const bool make_event, const unsigned short req_verbosity);	//util.hpp

#include "cuda_device.hpp"		// includes hashburner.cuh

//#define ushort unsigned short

//#include <inttypes.h>	// inttypes.h. Includes stdint/cstdint? <---
#include <loguru/loguru.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
//#include <cuda_runtime_api.h>
//#include <driver_types.h>
//#include <device_launch_parameters.h>
//#include <device_functions.h>

std::vector<cudaDevice*> CudaDevices;

#include "defs.hpp"

extern /* __align__(32) */ dim3 gCuda_Grid[CUDA_MAX_DEVICES];	//hashburner.cu
extern /* __align__(32) */ dim3 gCuda_Block[CUDA_MAX_DEVICES];	//hashburner.cu
#include "generic_solver.hpp"	//<---
//class genericSolver;

uint8_t gCuda_Engine[CUDA_MAX_DEVICES]{};		// [OLD] <---


#define DEBUG_PRINT_SUCCESSFUL_CUDA_CALLS
bool Cuda_Call(const cudaError_t apiCallResult, const std::string theTask, const int deviceID /*, const bool trivial*/)
{ // returns True If CUDA call successful, False otherwise. (if false, handles err. string.)
	if (apiCallResult == cudaSuccess) {
#ifdef DEBUG_PRINT_SUCCESSFUL_CUDA_CALLS
		if (DEBUGMODE) { printf("cuda call [dev.# %d]:	%s...\n", deviceID, theTask.c_str()); }
#endif
		return true;
	}
	else { /* err: */
		const std::string errString(cudaGetErrorString(apiCallResult));
		LOG_IF_F(ERROR, NORMALVERBOSITY, "[cuda dev.# %d] while %s: %s", deviceID, theTask.c_str(), errString.c_str());		//WARNING ?
	//	LOG_IF_F(trivial ? WARNING : ERROR, NORMALVERBOSITY, "[cuda dev. #%d] while %s: %s", deviceID, theTask.c_str(), errString.c_str());		//WARNING ?
		return false;
	}
}

//
// === cudaDevice class ===
//

cudaDevice::cudaDevice(const int cuda_device_no, genericSolver* for_solver)
{ // === constructor: ===
	if (cuda_device_no >= 0 || cuda_device_no < CUDA_MAX_DEVICES) {
		LOG_IF_F(INFO, DEBUGMODE, "Instantiating cudaDevice # %d: ", cuda_device_no);
		dev_no = cuda_device_no;
		solver = for_solver;		//cuda device object is assigned to a solver object.
		this->SetStatus(DeviceStatus::Null);
		is_initialized = false;		//<--
		is_cleaned_up = false;		//<-- condense with "status"?
	}
	else throw(ExceptionType::CtorFailed);	// [TODO] / [WIP]. <--

	if (Cuda_Call(cudaGetDeviceProperties(&deviceProperties, dev_no), "cudaGetDeviceProperties", dev_no))
		LOG_IF_F(INFO, DEBUGMODE, "new cudaDevice: got device# %d", cuda_device_no);
	else throw(ExceptionType::NoDevice);	//throw a more specific exception? catch in calling func.! [TODO / FIXME] <--

//	const bool init_success = this->Init_Device();	// initing when mining starts instead. <--
	h_solutions			= nullptr;		//<-
	h_solutions_count	= nullptr;		//<-
	d_solutions		= nullptr;
	d_solutions_count	= nullptr;

//	...
} //everything inited? check all members. <-- [TODO] / [WIP].



cudaDevice::~cudaDevice()
{ // === destructor: ===
	LOG_IF_F(INFO, DEBUGMODE, "cudaDevice (API dev.# %d) destructor", dev_no);
	this->status = DeviceStatus::Null;
	//	this->solver->status = SolverStatus::NotSolving;
	
	if (!this->is_cleaned_up) {
		if (this->Clean_Up()) {
			LOG_IF_F(INFO, HIGHVERBOSITY, "Cleaned up cudaDevice# %d: ", this->dev_no);
			return;
		}
		else LOG_F(WARNING, "CUDA device# %d cleanup encountered error(s)!");
	} else LOG_IF_F(INFO, DEBUGMODE, "Clean-up of CUDA device# %d already done!", dev_no);

	//	anything else to clean up?	[TODO] [WIP] <----
}


bool cudaDevice::Clean_Up(void)						// <--- [MOVEME]. Destructor instead <---
{
	LOG_IF_F(INFO, DEBUGMODE, "cudaDevice (API dev.# %d) Clean_Up():", dev_no);
	this->is_initialized = false;	//<--

	unsigned short errs{ 0 };
	h_solutions ? Cuda_Call(cudaFreeHost(h_solutions), "freeing h_solutions", dev_no) : ++errs;
	h_solutions_count ? Cuda_Call(cudaFreeHost(h_solutions_count), "freeing h_solutions_count", dev_no) : ++errs;
	d_solutions ?		Cuda_Call(cudaFree(d_solutions), "freeing d_solutions", dev_no) : ++errs;
	d_solutions_count ? Cuda_Call(cudaFree(d_solutions_count), "freeing d_solutions_count", dev_no) : ++errs;

	if (!Cuda_Call(cudaDeviceReset(), "resetting device", dev_no))	//<---
		return false;	//Error! ++err ? <----

	if (errs) {
		LOG_F(WARNING, "Error(s): %u", errs);
		return false;
	}

	this->is_cleaned_up = true;	//<-- any other cleanup?
	return true;	//OK
}


bool cudaDevice::Init_Device(void)
{ // [TODO] [Testing]:	abort mining start if a Device fails to Init? should try to init next anyway.
	LOG_IF_F(INFO, DEBUGMODE, "CUDA device# %d: Init_Device():", dev_no);
	if (!Cuda_Call(cudaSetDevice(dev_no), "setting device", dev_no)) {
		LOG_F(WARNING, "Couldn't set CUDA device #%d. Not initializing it.", dev_no);
		status = DeviceStatus::Unavailable;
		return false;	//err, mining will not start
	}

	status = DeviceStatus::Initializing;
	if (Cuda_Call(cudaGetDeviceProperties(&deviceProperties, dev_no), "getting device properties", dev_no)) {
		solver->gpuName = std::string(deviceProperties.name);	// saving device name in solver class (not cudaDevice).
		LOG_IF_F(INFO, NORMALVERBOSITY, "Got CUDA device# %d [%s] properties: using Intensity %u.", dev_no, deviceProperties.name, Solvers[dev_no]->intensity);
		domesg_verb("Got CUDA Device# " + std::to_string(dev_no) + " properties:	" + solver->gpuName + ".	Using intensity: " +
			std::to_string(solver->intensity), true, V_NORM);	//reminder for user to set intensity. <--
	} else { //if error:
		domesg_verb("Couldn't get CUDA device# " + std::to_string(dev_no) + " properties. Busy? Intensity setting too high?", true, V_NORM);
		status = DeviceStatus::Unavailable;
		return false;	//err: mining will not start
	}
//
	solver->threads = static_cast<uint32_t>(1u << solver->intensity);
// set grid size/threads-per-block for this device:
	if (deviceProperties.major >= 5) { /* compute capability >=500 control path (500) */
if (DEBUGMODE) printf("setting grid: %" PRIu32 " + %u - 1  /  %" PRIu32 " ", solver->threads, TPB50, TPB50);
		gCuda_Grid[dev_no] = (solver->threads + TPB50 - 1) / TPB50;  // <--- adjust for "compatibility" engine [todo].						//GRID
LOG_IF_F(INFO, DEBUGMODE, "grid: %d(x), %d(y), %d(z) \n", gCuda_Grid[dev_no].x, gCuda_Grid[dev_no].y, gCuda_Grid[dev_no].z);
if (DEBUGMODE) printf("setting block: %" PRIu32 " ", TPB50);
		gCuda_Block[dev_no] = TPB50;																												//BLOCK
LOG_IF_F(INFO, DEBUGMODE, "block: %d(x), %d(y), %d(z) \n", gCuda_Block[dev_no].x, gCuda_Block[dev_no].y, gCuda_Block[dev_no].z);
	}
	else { //old architectures, not currently supported
		domesg_verb("CUDA Device# " + std::to_string(dev_no) + " not supported. Compute Capability of 5.x or higher needed.", true, V_LESS);	//<-- [TODO] / [FIXME]: use log callback.
		LOG_F(WARNING, "GPU not supported: compute capability of >=5.x is needed");
		//	gCuda_Grid[dev_no] = (gCuda_Threads[dev_no] + TPB35 - 1) / TPB35;
		//	gCuda_Block[dev_no] = TPB35;
		status = DeviceStatus::Unavailable;
		return false;	//err
	}

// Reset right before mining start? Only? <---- [WIP].
	LOG_IF_F(INFO, HIGHVERBOSITY, "Resetting CUDA GPU# %d (%s):", dev_no, solver->gpuName.c_str());
	if (!Cuda_Call(cudaDeviceReset(), "resetting device", dev_no)) {
		status = DeviceStatus::Unavailable;
		return false;	//err
	}

	// Warning: these flags must be supported. If this call fails, mining will not start on the device. [WIP]
	if (!Cuda_Call(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync | cudaDeviceLmemResizeToMax/* | cudaDeviceMapHost*/), "setting device flags", dev_no)) {
		LOG_F(ERROR, "Couldn't set CUDA device# %d flags. Device not supported?", dev_no);
		status = DeviceStatus::Unavailable;	//::NotSupported?
		return false;	//err
	}

	deviceProperties.major >= 7 ? Cuda_Call(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1), "setting func cache config to PreferL1", dev_no) :
		Cuda_Call(cudaDeviceSetCacheConfig(cudaFuncCachePreferNone), "setting func cache config to PreferNone", dev_no);	//faster on Turing?

// host<->device i/o memory is not allocated yet- do that at mining start. See genericSolver::Start() ! <--- [WIP]

// === initialize counter and time points ===
//	ftime(&solver->tStart);		//<-- time start/end wont be meaningful 'til the mining loop starts, see constructor
//	ftime(&solver->tEnd);		//<-- remove
//	solver->tStart = solver->tEnd = {};		//neater. valid to init this way?
	cnt[dev_no] = 0;	// [todo]: re-test separate init-messages per GPU w/ unified counter. was cnt[dev_no] = CNT_OFFSET * dev_no;

	this->SetStatus(DeviceStatus::Ready);	//<-----
	
	LOG_IF_F(INFO, DEBUGMODE, "-- for genericSolver #%u (as numbered by COSMiC) --", solver->solver_no);	//<--- Debug Only
	LOG_IF_F(INFO, NORMALVERBOSITY, "Available compute capability: %d.%d", deviceProperties.major, deviceProperties.minor);
	LOG_IF_F(INFO, NORMALVERBOSITY, "Initialized CUDA device %d : %s", dev_no, solver->gpuName.c_str());
	LOG_IF_F(INFO, DEBUGMODE, "CUDA Device# %d: %s is ready ", dev_no, solver->gpuName.c_str());
	return true;	//OK
}

extern __device__ uint32_t solns_count;
extern __device__ uint64_t solns_data;


// [WIP]: consider a generic allocate function in genericSolver.
//bool cudaDevice::Allocate(void) { ... }
// WAS HERE, see hashburner.cu



bool cudaDevice::clear_solutions(void)
{
	if (h_solutions != nullptr && h_solutions_count != nullptr)
	{
	// [TESTME].
		LOG_IF_F(INFO, DEBUGMODE, "Initializing solutions array (cuda device# %d). [TODO]: Make device type fully generic.", dev_no);
		*h_solutions_count = 0;	//added *
		memset(h_solutions, 0xFF, SOLUTIONS_SIZE);		// clear host-side solutions array for this device
		cudaMemset(d_solutions, 0xFF, SOLUTIONS_SIZE);			// legal?
		cudaMemset(d_solutions_count, 0, SOLUTIONS_COUNT_SIZE);	// <--- [WIP] / [FIXME]
	//...
	} else {
		LOG_F(ERROR, "Bad pointer- lost CUDA device# %d?", this->dev_no);
		return false;
	}
	// [FIXME] ! <-----

}



#include <mutex>		//<-- [MOVEME]
#include <queue>		//<-- [MOVEME]
#include "types.hpp"	//<--
#include "util.hpp"		//<-- just for uint8t_array_toHexString(). <--
#include "network.hpp"
//
#include "coredefs.hpp"	//<-- just for QueuedSolution. Move it to "types.hpp" <--
// [WIP]: passing pointer to a Solver's member Params passed to this function might remove need for mutex-lock while reading the Params
//			(intended to prevent the solution[32] "template" being read while it is changing.) _IF_ the Solvers pull their own params when
//			the global ones update (see bool genericSolver::new_params_available), they can't have _THEIR_ .Params won't be changed by another thread.
//

// multiple overloads for different device types [TODO].
bool cudaDevice::enqueue_solutions (/*unsigned short* solns_found*/)
{
// bool err{ false };	//bool success{ false };
	if (solver->params_changing) {
	LOG_F(WARNING, "Not enqueueing sol'ns- solver's mining params are changing");
	return false; }							// the sol'ns must be cleared! See calling function! <--

//alternately (lock sol'ns queue mutex, enqueue all sol'ns, unlock) <---
	//try {
	if (h_solutions == nullptr || h_solutions_count == nullptr) {
		LOG_F(ERROR, "Bad pointer- lost CUDA device# %d ?", this->dev_no);
		return false; }

	unsigned short solns_found{0};
	solns_found = static_cast<unsigned short>(*h_solutions_count);	//this->cuda_device->h_solutions_count
	if (solns_found < 1 || solns_found > MAX_SOLUTIONS) {
		LOG_F(WARNING, "No or bad # of solutions (%u) in cudaDevice::enqueue_solutions()!" BUGIFHAPPENED, solns_found);
		return false; }
	//catch (...) {
		//exception caught. device lost?	// [WIP] <---
		//Clear Solutions!					// [FIXME] Clear any sol'ns in cudaDevice::launch_kernels()? (if no error). <---
		//return false;
	//}

	LOG_IF_F(INFO, DEBUGMODE, "enqueueing %u solutions from solver# %u, cuda device# %d...", solns_found, solver->solver_no, dev_no);
	print_bytes(reinterpret_cast<uint8_t*>(h_solutions), static_cast<uint8_t>(solns_found) * 8, "h_solutions");	//<-- Debug Only

//	std::unique_lock<std::mutex> u_lockg(mtx_solutions);	// to protect q_solutions from async access by multiple solver threads.
	//offset pointer address to read, enqueue each uint64 value in the array:
	for (unsigned short s = 0; s < solns_found; ++s)
	{ 
		QueuedSolution temp_soln{};		//has default initializing values. {} redundant here?	[WIP]<---
		if (h_solutions + s > h_solutions + (MAX_SOLUTIONS-1)) {	//just in case.	was: >=
			LOG_F(ERROR, "Bad pointer to solution!" BUGIFHAPPENED);
			return false;	// [CHECKME] <--
			break;			//redundant
		}
		if (solver->params_changing) {
			LOG_IF_F(WARNING, HIGHVERBOSITY, "Not enqueueing %u sol'ns- the solver's params are changing.", solns_found);
			return false;
			break;
		}

		temp_soln = {};
		if (DEBUGMODE) { print_bytes(temp_soln.solution_bytes, 32, "TEMP_SOLN (before anything - should be zeroes)"); }	// [DEBUG], remove

//		std::unique_lock<std::mutex> ulock_params(mtx_coreparams);
		memcpy(temp_soln.solution_bytes, &solver->initial_message[52], 32);	// copy 32-byte solution "template" from the solver's message (after the challenge/mint address)
		if (DEBUGMODE) { print_bytes(temp_soln.solution_bytes, 32, "sol'n bytes (before copy)"); }	// [DEBUG], remove
		memcpy(&temp_soln.solution_bytes[12], &h_solutions[s], 8);				//copy solution bytes
	//	memcpy(&temp_soln.solution_bytes[12], h_solutions + s, 8);	//copy solution bytes (alt.)
		if (DEBUGMODE) { print_bytes(temp_soln.solution_bytes, 32, "sol'n bytes (after copy)"); }	// [DEBUG]<-- remove call

	// populate other req'd members of `temp_soln`, push into `q_solutions`.
		temp_soln.deviceOrigin = this->dev_no;
		temp_soln.deviceType = DeviceType::Type_CUDA;	//was `devType`;
		
		temp_soln.solver = this->solver;				//set solverOrigin in the new sol'n [WIP]<-----

		temp_soln.solution_string = "0x" + uint8t_array_toHexString(temp_soln.solution_bytes, 32);		// convert, store 32 bytes as hex in a std::string. (still used?) <-- [todo]
		LOG_IF_F(INFO, NORMALVERBOSITY, "Solution found by CUDA device# %d: %s ", dev_no, temp_soln.solution_string.c_str());	//<-- use log callback
	//	domesg_verb("Solution found by CUDA device#" + std::to_string(deviceID) + ": " + temp_soln.solution_string, true, V_NORM);		//<-- use log callback
		uint8_t challenge[32]{};
		memcpy(challenge, solver->initial_message, 32);
		temp_soln.challenge_string = uint8t_array_toHexString(challenge, 32); //gMiningParameters.challenge_str;	// save challenge it was solved for (this will be useful). [TODO]: pass major mining params as function args.
		temp_soln.poolDifficulty = solver->difficulty;	//gMiningParameters.difficultyNum;		// same for the difficulty #
	//
	//FIXME: The poolDifficulty must be set!!!!
	// 
		// if(!check_vs_prior_solutions(temp_soln.solution_string))	return false;	//<-- don't add it to the queue. chasing a bug. [wip]
		// fl_pastsolutions.push_front(temp_soln);									//<-- TESTME (debug use only!)
		LOG_IF_F(INFO, DEBUGMODE, "Storing solution in queue:	%s", temp_soln.solution_string.c_str());
		std::unique_lock<std::mutex> ulock_solns(mtx_solutions);	// to protect from async access to q_solutions
		q_solutions.push(temp_soln);								// push local temp. solution to queue
		ulock_solns.unlock();
	} //mutex?		consider Solver#X thread<->Network BGworker thread comm. [WIP]	<---
	//ulock_solns.unlock();

	this->solver->ResetHashrateCalc();			// reset hash count, rate, time start.		//<---- [WIP]: Move setting of tStart to correct location!! (tEnd also).
	//ftime(&this->tStart);						// [MOVEME] <--

	return true;	//if (err)	return false;
}

bool cudaDevice::SetStatus (const DeviceStatus new_status)
{
	status = new_status;
	solver->device_status = status;
	LOG_IF_F(INFO, HIGHVERBOSITY, "CUDA Device status change: %d", (int)status);	//<--

	if (status == DeviceStatus::Fault || status == DeviceStatus::MemoryFault) {
		LOG_F(WARNING, "CUDA Device has fault/memory fault status (%d) !", (int)status);	//<--
		solver->solver_status = SolverStatus::DeviceError;
		return false;
	}
	return true;
}

//}; //class cudaDevice


// by Init_Device(), cudaDevice::cudaDevice() also <--- [WIP].
	// - if (!gCudaDeviceEnabled[this->dev_no])	continue;		// skip if disabled. shouldn't be called.
	// - `dev_no` checked already in constructor? don't instantiate a cudaDevice with a bad device#. <---- [CHECKME].
	// ..... Move these functions to `generic_solver.cpp`? does NVCC need to compile them, or just the device code? <----- [WIP].

// [WIP]: CLEANUP IN WHICH DESTRUCTOR? <------ [FIXME] !
//	** Check for any occurrences of `dev_no` (CUDA device!) which should be Solver #. <----
