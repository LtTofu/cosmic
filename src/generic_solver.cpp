// generic_solver.cpp (.cpp/.hpp)
// GPU solver class for COSMiC V4.  2020 LtTofu
// 
// Goal:	A generic solver for GPUs, nVidia CUDA (and AMD?) with OOP approach.
//			Currently implemented only for CUDA. A work-in-progress!


#include <cstdint>				// <cinttypes> -or- <inttypes.h>...
#include <cuda.h>				// [TESTME]. Prefer to keep CUDA calls in the .cu file?
#include <cuda_runtime.h>
#include <loguru/loguru.hpp>

#include "defs.hpp"
//#include <string>				// included in `defs`
//#include "coredefs.hpp"
#include <net_solo.h>			// gTxView_WaitForTx, etc.
#include "hashburner.cuh"		// cudaDevice class
#include "generic_solver.hpp"
#include "cuda_device.hpp"		//<---

#include <sys/timeb.h>

//#include <chrono>
//std::chrono::steady_clock
//std::chrono::high_resolution_clock

//#define DEBUG_ALLOCATE_IN_CONSTRUCTOR
#define DEBUG_ALLOCATE_ON_INIT

constexpr size_t MAX_SOLUTIONS_SIZE = MAX_SOLUTIONS * sizeof(uint64_t);
constexpr size_t MAX_SOLUTIONS_COUNT_SIZE = sizeof(uint8_t);

//bool update_device_symbols(const miningParameters* params);		//hashburner.cu

#include "types.hpp"
std::vector<genericSolver*> Solvers;

//constructor for generic solver
genericSolver::genericSolver(const DeviceType deviceType, const ushort deviceID, const ushort solverNo /*, const miningParameters* init_params*/)
{ // parameterized constructor
	LOG_IF_F(INFO, DEBUGMODE, "new genericSolver: size of `Solvers` was %zu <--", Solvers.size());	//<--- REMOVE
	enabled = false;	// must be enabled!
	//
	solver_no = static_cast<uint8_t>(solverNo);	// from 0 in the order they're instantiated (unrelated to numbering assigned by any device API)
	api_device_no = deviceID;
	device_type = deviceType;
	//this->cuda_device (later this func.)
	//this->other_device...		//support other types with a variation on this func.	<-----
	//this->gpumon_device...	//pointer to the hardware monitoring device				<----- [TODO]

	solver_status = SolverStatus::DeviceNotInited;	//::Null
	device_status = DeviceStatus::Null;
	pause = PauseType::WaitingForDevice;	//cuda device not initialized yet
	new_params_available = false;			//will pull from global mining parameters
	//resuming = false;

	//solutions_found = 0;
	valid_solutions = 0;
	invalid_solutions = 0;
	//stale_solutions = 0;
	
// === device type-specific (cuda) ===
	gpuName = "";	// to be set from device properties later this function		<-----
	intensity = DEFAULT_CUDA_INTENSITY;
	threads = 0;						//set from intensity later

	hash_count = 0;		//uint64
	hash_rate = 0.0;	//double
	ftime(&this->tStart);
	ftime(&this->tEnd);
	solve_time = 0.0;

	//memset(&this->params, 0, sizeof(miningParameters));	//<-- incorrect, see default initialization of objects of type `miningParameters` (types.hpp)
//	this->params should initialize itself. See default initialization in `miningParameters` definition (types.hpp).
	//uint8_t* DELETETHIS = (uint8_t*)&this->params;		//<------------ ^ CHECK THAT IT'S INITIALIZED PROPERLY, THEN REMOVE THIS
	memset(midstate, 0, 200);	//<-- 
	target = 0;					//<--the 64-bit uint target for a GPU. [redundant?]
	difficulty = 0;
	// <---- INITIALIZE HASH_PREFIX
	// <---- INITIALIZE INITIAL_MESSAGE
	//


	// [TODO]: This is clunky. Use multiple overloads of genericSolver() to construct a genericSolver set up for a specific device type.
	//			Or, objects of class type "cudaSolver", "cpuSolver", etc., which inherit from the genericSolver base class. <--
	if (deviceType == DeviceType::Type_CUDA) {
		//this->cuda_device->Init_Device();	//wait til mining start

		try {
			this->cuda_device = new cudaDevice(static_cast<int>(deviceID), this);		// pass ref. to this genericSolver to the cudaDevice constructor	[WIP].
			CudaDevices.push_back(this->cuda_device);
		}
		catch (ExceptionType e) {
			if (e == ExceptionType::CtorFailed) LOG_F(ERROR, "Exception caught while instantiating cudaDevice object- constructor failed");
			 else if (e == ExceptionType::CtorFailed) LOG_F(ERROR, "Exception caught while instantiating cudaDevice object- didn't get Device");
			 else LOG_F(ERROR, "Unknown Exception caught while instantiating cudaDevice object- out of memory?");

			throw ExceptionType::CtorFailed;	//solver's constructor failed
		}
		// TODO: exception if device properties can't be retrieved <--

		//got object?
		if (!cuda_device || !CudaDevices.back()) {		// [CHECKME]: if nullptr
		//	LOG_F(ERROR, "Couldn't allocate system memory for cudaDevice!");
			throw ExceptionType::CtorFailed;										//catch in instantiating function. <----
		}else
			LOG_IF_F(INFO, DEBUGMODE, "Instantiated cudaDevice, CudaDevices size: %zu", CudaDevices.size());

		intensity = 24;					//<-- default
		gpuName = "";						//<-- [fixme]: get the GPU name!
		cuda_device->solver = this;		// so genericSolver, cudaDevice can refer to each other
//...any other members to init?
	}
	else if (deviceType == DeviceType::Type_CPU) {
		/* ... [TODO]. */
	}
	else if (deviceType == DeviceType::Type_VK) {
		/* ... [TODO]. */
	}
	else {
		// unknown type. error!		(throw?)
	}

	solver_status = SolverStatus::DeviceNotInited;	//::Ready;
	LOG_IF_F(INFO, DEBUGMODE, "GenericSolver# %u constructed- Solver has device object.", this->solver_no);
}


genericSolver::~genericSolver()
{ // destructor: do per-device cleanup here instead?	[todo]. <--- MOVE THIS
	LOG_IF_F(WARNING, DEBUGMODE, "genericSolver destructor: cleaning up CUDA device# %d", this->cuda_device->dev_no);	//DBG
	try {
		if (this->device_type == DeviceType::Type_CUDA) {
			if (this->cuda_device != nullptr) {
				LOG_IF_F(INFO, DEBUGMODE, "deleting cudaDevice object # %d", this->cuda_device->dev_no);	//DBG
				delete this->cuda_device;
			} else LOG_F(WARNING, "cuda_device object does not exist! cannot clean it up");
		}
		// other device types [TODO] <--
	}
	catch (...) {	//the type of exception [FIXME]. <--
		LOG_F(WARNING, "Error(s) while freeing resources for CUDA device# %d!", this->cuda_device->dev_no);
		// ... [TODO]
	}
}


bool genericSolver::Start (/*miningParameters init_params*/)
{
	this->solver_status = SolverStatus::Starting;

// === Solver Loop ===
	const bool success = this->CudaSolve( /*init_params*/ );	//<--- pass by reference instead?
	if (success) {
		this->solver_status = SolverStatus::NotSolving;		// _STOPPING
		LOG_IF_F(INFO, /*HIGH*/NORMALVERBOSITY, "Solver stopping normally.");
	} else {
		this->solver_status = SolverStatus::DeviceError;
		LOG_F(INFO, "Solver# %u stopping because of a device error");
	}
// ===
//	this->ResetHashrateCalc();
//	this->hash_rate = 0.0;
//	this->hash_count = 0;							// here? Init_Device() should already be doing this <---
//	gSolving[this->cuda_device->dev_no] = false;	//redundant?
// ...

	return success;
}





//constexpr auto PARAMSCHANGE_WAIT_MS = 1000;			// if the solver's params were being updated from the main or network thread
constexpr auto NETWORK_WAIT_MS = 2000;
constexpr auto PARAMS_WAIT_MS = 1000;
void Pause_ms(const unsigned long ms);	// CosmicWind.cpp


// [FIXME]: `this->new_params_available` must be set to TRUE when new mining parameters are retrieved !! <---- [WIP / FIXME]

ushort genericSolver::CudaSolve(/*miningParameters params*/)	//<-- Finish converting to a member func. of genericSolver class!
{
	bool end{ false };			//<--
	bool err{ false };			//<--
	bool have_params{ false };	//<-- [WIP]: messy
	gSolving[cuda_device->dev_no] = true;			// for UI: solver thread started
//	this->solver_status = SolverStatus::Solving;
	ftime(&tStart);	//<---

//	=== Mining Loop: ===
	do {
		//if(err) break;
		if (!gCudaSolving || gApplicationStatus == APPLICATION_STATUS_CLOSING)		/* if mining is stopped by the user, or system */
			break;	// mining should gracefully stop

		// [todo] / [wip]: reset hashrate calc? when unpausing. <---
		if (gNetPause || gTxView_WaitForTx != NOT_WAITING_FOR_TX) {		// || this->params_changing
			this->solver_status = SolverStatus::WaitingForNetwork;
			Pause_ms(NETWORK_WAIT_MS);
			continue;						//<--- unnecessary?
		}
		else if (g_params_changing || gMiningParameters.params_changing) {	//<--- One or the other! <--- [WIP]
			this->solver_status = SolverStatus::WaitingForParams;
			Pause_ms(PARAMS_WAIT_MS);
			continue;						//<--- unnecessary?
		}
		else { //not waiting for parameters or the network:
			if (solver_status == SolverStatus::WaitingForNetwork || solver_status == SolverStatus::WaitingForParams) {
				LOG_IF_F(INFO, HIGHVERBOSITY, "Solver# %u is resuming!", this->solver_no);
				solver_status = SolverStatus::Resuming;
			}
		}

	//	if (g_params_changing){} [old]		// don't solve with parameters that are changing (would make the solutions invalid) [WIP] <------- check in this->NewParameters()?

		if (this->new_params_available) {	/* WIP: set to TRUE at outset of mining and skip sending the params to device initially (do it here instead?) */
			if (!this->NewParameters(&gMiningParameters)) {//copy the current (global) parameters, generate new midstate, set this solver's `midstate` and `target`. [WIP] ! <---
				// ... <------- let's condense these IFs.
				Pause_ms(PARAMS_WAIT_MS);	//<---
			}
			//otherwise
			//solver_status = SolverStatus::Solving;	//<---
		}

		// [TODO]: pause if `g_params_changing`
		if (!have_params || this->new_params_available) {
			solver_status = SolverStatus::UpdatingParams;
			have_params = this->NewParameters(&gMiningParameters);		//hashrate calc. vars reset here? <--
			if (have_params) {
				solver_status = SolverStatus::Resuming;	// was ::Solving. "Resuming" will force the Hashrate Calculation to refresh.
				//if (SendToTheDevice(... ? )) {...}	//update on the device. [WIP]: replacing with passing midstate, target as arguments to kernel <--
			}
			else {
				if (DEBUGMODE)printf("Solver# %u is waiting for parameters \n", solver_no);
				//...
				Pause_ms(250);	//wait 0.25s
				continue;
			}
		}
		// ^ reset hashrate calc? <-- [todo] / [wip].

		if (solver_status == SolverStatus::Resuming) { //the wait reason (network/parameters change) is resolved:
			LOG_IF_F(INFO, DEBUGMODE, "solver# %u resuming.", solver_no);		// Remove! <-----
			solver_status = SolverStatus::Solving;	//this->resuming = false;	// [WIP].
			this->ResetHashrateCalc();	// [TESTME]
			//if (solver_status == SolverStatus::WaitingForNetwork)	solver_status = SolverStatus::Solving;
		}
		
		unsigned short solns_found{ 0 };							//<---
		if (!cuda_device->find_solutions( &solns_found )) {			//<---
			this->solver_status = SolverStatus::DeviceError;
			//this->device_status = DeviceStatus::Fault;
			LOG_IF_F(INFO, /*ANY?*/NORMALVERBOSITY, "Error launching kernels from solver# %u (%s) !", this->solver_no, this->gpuName.c_str());	//<--- remove debug messages!
			err = true;
			//...
		// ------>	IF ANY SOLUTIONS STILL REMAIN IN THE DEV_SOLUTIONS ARRAY THEY MUST BE CLEARED, H_ COUNTERPARTS ALSO !	<------
			break;
		}// [TESTME] <--


		//ftime(&tEnd);	//not really "end" of the solve. updated each run that no sol'ns are found <-- [wip] / [testme].
		LOG_IF_F(INFO, DEBUGMODE, "solver# %u ::Solve() found %u sol'ns this launch ", solver_no, solns_found);//<--- [REMOVE]

	} while (gApplicationStatus != APPLICATION_STATUS_CLOSING && gSolving[solver_no]==true);		// if _FAULT or _MEMFAULT...  <- [todo]
//	} while (gApplicationStatus != APPLICATION_STATUS_CLOSING && gCudaSolving);		// if _FAULT or _MEMFAULT...  <- [todo]


	// remove
	LOG_IF_F(INFO, DEBUGMODE, "do-while loop over - gSolving[%u]: %s, gApplicationStatus: %u", solver_no, std::to_string(gSolving[solver_no]).c_str(), gApplicationStatus);
	// remove
	//Solving[this->cuda_device->dev_no] = false;
	return err ? false : true;		// true=OK, false=Err
}



// combine into EnqueueSolutions() if possible. [TODO]
bool genericSolver::ClearSolutions(void)
{	// [TESTME] !
	bool success{ false };
	LOG_IF_F(INFO, DEBUGMODE, "Initializing solutions array (solver# %u). [TODO]: Make device type fully generic.", solver_no);
	memset(cuda_device->h_solutions, 0xFF, SOLUTIONS_SIZE);		// clear host-side solutions array for this device
	cuda_device->h_solutions_count = 0;
	cudaMemset(cuda_device->d_solutions, 0xFF, SOLUTIONS_SIZE);			// legal?
	cudaMemset(cuda_device->d_solutions_count, 0, SOLUTIONS_COUNT_SIZE);	// <--- [WIP] / [FIXME]
	return success;
}



//	Free relevant memory used on the device and host side, resets the device and clears relevant vars in the solver.
//	[TODO / FIXME]		Moving this into cudaDevice class, and just call the relevant func. for the device type <-- [WIP]
bool genericSolver::Shutdown (void)
{ // consolidate w/ destructor?    // Called when mining stops, in each device solver's thread.
	domesg_verb("Shutting down CUDA device# " + std::to_string(cuda_device->dev_no) + ".", true, V_MORE);
	LOG_IF_F(INFO, /*HIGH*/NORMALVERBOSITY, "Shutting down CUDA device #%d...", cuda_device->dev_no);

// === call the device type-specific shutdown func. ===
	if(device_type==DeviceType::Type_CUDA)
		const bool shutdown_result = cuda_device->Clean_Up();		// true= OK		[WIP] [TESTME] !
//	else if(device_type==DeviceType:: ...)	// [TODO].
	else {
		LOG_F(WARNING, "Unexpected device type %u in genericSolver::Shutdown()!" BUGIFHAPPENED, static_cast<ushort>(device_type));
	}

// === specific to generic solver ===
	solver_status = SolverStatus::NotSolving;	// update device status
	pause = PauseType::NotPaused;
	hash_count = 0;	// [MOVEME] ?
	gSolving[cuda_device->dev_no] = false;	// for UI.	[todo]: make device type-agnostic.
	// hashrate, solution counts etc. [todo]
	// ...

	LOG_IF_F(INFO, /*HIGH*/NORMALVERBOSITY, "Cleaning up behind CUDA device# %d", cuda_device->dev_no);
	//Clean_Up() ? //<--------- [WIP] / [FIXME] ! <-----------------------------
	// ...
	
	LOG_IF_F(INFO, /*HIGH*/NORMALVERBOSITY, "Host thread for CUDA Device# %d ending", cuda_device->dev_no);
	// COMBINING.... ^
	// anything else to reset? do in destructor?
	return true;	// WIP. return false if err! <--
}


void genericSolver::ResetHashrateCalc (void)
{
	LOG_IF_F(INFO, DEBUGMODE, "Resetting hashrate calc. for Solver# %u", solver_no);
	ftime(&tStart);

	hash_count = 0;		//hashes computed this solve
	hash_rate = 0.0;
	cnt[cuda_device->dev_no] = 0;
	// clear h_solutions/dev_solutions and counts here? <--- [WIP]
}


bool genericSolver::GetSolverStatus( double* hashrate, uint64_t* hashcount )
{ //this: genericSolver, not cudaDevice.
// mutex-lock here _should_ be unnecessary (just reading)
// ...
//	const ushort deviceNo = this->solver_no;
//	solvingGpuStatus[ deviceNo ]->hashrate = ..
	*hashcount = this->hash_count;
	*hashrate = this->hash_rate;
// ...
	return true;	// OK. `void` return type?
}


//ushort Solvers_allocated{ 0 };	//<--- [MOVEME]. to include genericSolver objects for any device type (generic) which are instantiated

bool free_solvers(void)
{
	if (!Solvers.size()) {
		LOG_IF_F(INFO, DEBUGMODE, "No solvers to free");
		return true;	//OK
	}
// ^^ delete this ^^

	const size_t oldsize = Solvers.size();
	size_t freed = 0;
	LOG_IF_F(INFO, DEBUGMODE && !Solvers.empty(), "Freeing %zu solvers...", Solvers.size());
	while (!Solvers.empty()) {
		try {
			delete Solvers.front();	//<-- free any allocated solver
			Solvers.pop_back();		//<-- remove pointer to it from vector.
			++freed;
		}
		catch (...) { // [todo]: what kind of exception
			LOG_F(ERROR, "Caught exception in free_solvers()!");
			return false;
		}
	}

	if (freed > 0)																		//remove <---
		LOG_IF_F(INFO, DEBUGMODE, "Done. Solvers' size: %zu objects.", Solvers.size());	//remove <---
	return true;	//OK
}

// Call from: CosmicWind.h, CosmicWind constructor?	[WIP].				<---
// [WIP]:	Just any detected CUDA devices (for now!), indexed from 0. <---
// [WIP]: Also, check any ->Update and ->Initialize() calls... <---
// [WIP]: this function can be condensed down substantially.
bool allocate_gpusolvers_cuda(const ushort howmany)
{ // input: device type and how many devices? do any sanity checking. <-- [WIP]
	if (howmany > MAX_SOLVERS) return false;
	ushort allocated{ 0 };

	//Solvers.clear();	//somewhere?
	if (Solvers.size()) { /* should be empty! */ /* any point? */
		LOG_IF_F(WARNING, DEBUGMODE, "`Solvers` not empty! " BUGIFHAPPENED);		// shouldn't happen <-- [TESTME]
		return false; }

	bool error{ false };
	for (ushort i = 0; i < howmany && Solvers.size() < MAX_SOLVERS; ++i) {	// [CHECKME].
		LOG_IF_F(INFO, DEBUGMODE, "allocating CUDA genericSolver: device# %u with size %zu bytes", i, sizeof(genericSolver));
		try { // [WIP]: genericSolver constructor args: device type, `i`: CUDA device index, solver #. <---
			const ushort solver_num = static_cast<ushort>(Solvers.size());	//vector's size should increase -after- genericSolver() constructor finishes. <--
			genericSolver* temp_gpusolver_ptr = new genericSolver(DeviceType::Type_CUDA, i, solver_num); //[TESTME] //<-- throw exception from constructor if failed? <--
			Solvers.push_back(temp_gpusolver_ptr);				// should facilitate use of Solvers[0], Solvers[1]... etc. to point to the genericSolver class instance.	[WIP]!
		} // [TESTME].
		catch (ExceptionType e) {	// [WIP] !	[TODO]: log/indicate the kind of exception.
			if (e==ExceptionType::CtorFailed)
				LOG_F(ERROR, "Exception caught while allocating genericSolver# %u: constructor failed. Out of memory?", i);
			else
				LOG_F(ERROR, "Unknown Exception caught while allocating genericSolver# %u !", i);

			error = true;
		}

		if ( !Solvers[i] )
			error = true;	//make sure it was allocated

		LOG_IF_F(INFO, DEBUGMODE, "allocate_gpusolvers_cuda(): size is %zu <--", Solvers.size());	//<------- REMOVE
		if (error) break;	//if any solver wasn't instantiated
	}//for

	if (!error) {
		LOG_IF_F(INFO, DEBUGMODE, "Instantiated %zu genericSolvers.", Solvers.size());
		return true;	// OK: allocated solver(s).
	} else {
		LOG_IF_F(INFO, DEBUGMODE, "couldn't allocate a solver, cleaning up");
		if(allocated > 0)
			free_solvers();	//<---

		return false;	// Err
	}
}


bool genericSolver::InitGenericDevice(void)
{
	if (device_type == DeviceType::Type_CUDA)
	{
	// [WIP]: combine Init_Device() and Allocate()
		if (!cuda_device->Init_Device())	/* logs any errors descriptively */
			return false;	//err
		if (!cuda_device->Allocate()) {
			LOG_F(ERROR, "Device unavailable?");
			this->device_status = DeviceStatus::Unavailable;				//<-- sort of confusing
			this->solver_status = SolverStatus::DeviceError;				//
			//this->cuda_device->status = DeviceStatus::Unavailable;
			return false;	//err
		}
		
		//this->SetIntensity();		// [TODO] <---

	}
	//else if (device_type == DeviceType::...)	[TODO]: other device types.
	else {
		LOG_F(ERROR, "Unsupported device type.");
		return false;	//err
	} // [todo]: support other device types here.

	return true;	//OK
}


//void Cuda_UpdateDeviceIntensity(const unsigned short deviceID) {
void genericSolver::SetIntensity(void)
{	//gCuda_Threads[deviceID] = (1u << gCudaDeviceIntensities[deviceID]);			//gCudaDeviceIntensities[deviceID]);
	this->threads = 1u << this->intensity;
	//Cuda_ResetHashrateCalc(deviceID);		//if changed while running? [FIXME].
}



const uint8_t version_bytes[4] = { 0xC4, 0x15, 0xB8, 0x00 };	// cosmic v4.1.5, preview8, reserved byte.

//bool ComputeMidstate(const uint64_t* initial_message)
bool ComputeMidstate(const uint64_t* message, uint64_t* out)
{
	LOG_IF_F(INFO, HIGHVERBOSITY, "Building midstate...");			//for this solver
	if (!message) return false;

// precomputing the first keccak256 round for speed (generating outstate)
	uint64_t C[4]{}, D[5]{};

	C[0] = message[0] ^ message[5] ^ message[10] ^ 0x100000000ull;
	C[1] = message[1] ^ message[6] ^ 0x8000000000000000ull;
	C[2] = message[2] ^ message[7];
	C[3] = message[4] ^ message[9];

	D[0] = ROTL64(C[1], 1) ^ C[3];
	D[1] = ROTL64(C[2], 1) ^ C[0];
	D[2] = ROTL64(message[3], 1) ^ C[1];
	D[3] = ROTL64(C[3], 1) ^ C[2];
	D[4] = ROTL64(C[0], 1) ^ message[3];

	out[0] = message[0] ^ D[0];
	out[1] = ROTL64(message[6] ^ D[1], 44);
	out[2] = ROTL64(D[2], 43);
	out[3] = ROTL64(D[3], 21);
	out[4] = ROTL64(D[4], 14);
	out[5] = ROTL64(message[3] ^ D[3], 28);
	out[6] = ROTL64(message[9] ^ D[4], 20);
	out[7] = ROTL64(message[10] ^ D[0] ^ 0x100000000ull, 3);
	out[8] = ROTL64(0x8000000000000000ull ^ D[1], 45);
	out[9] = ROTL64(D[2], 61);
	out[10] = ROTL64(message[1] ^ D[1], 1);
	out[11] = ROTL64(message[7] ^ D[2], 6);
	out[12] = ROTL64(D[3], 25);
	out[13] = ROTL64(D[4], 8);
	out[14] = ROTL64(D[0], 18);
	out[15] = ROTL64(message[4] ^ D[4], 27);
	out[16] = ROTL64(message[5] ^ D[0], 36);
	out[17] = ROTL64(D[1], 10);
	out[18] = ROTL64(D[2], 15);
	out[19] = ROTL64(D[3], 56);
	out[20] = ROTL64(message[2] ^ D[2], 62);
	out[21] = ROTL64(D[3], 55);
	out[22] = ROTL64(D[4], 39);
	out[23] = ROTL64(D[0], 41);
	out[24] = ROTL64(D[1], 2);

	print_bytes((uint8_t*)out, 200, "midstate");	//DBG
	return true;	//OK
}

#include "util.hpp"
#define MINTADDR_STR_LEN_WITH_0X 42
constexpr bool VERSION_IN_SOLN = true;	//write the version/build# in solution nonces
#include <libsodium/sodium.h>			//<--- Ensure LibSodium is Init'ed already <--- [TODO]
bool genericSolver::NewParameters(const miningParameters* new_params)
{ // Update the Solver ("this") with the latest parameters.		//<--- [WIP]: watch your step.
	LOG_F(INFO, "Preparing new Parameters for Solver # %u...", this->solver_no);
	if (new_params->params_changing) {	//<-- Confusing! Solver has a `params_changing`!
		LOG_IF_F(INFO, HIGHVERBOSITY, "Solver# %u not updating because the specified parameters are changing.", this->solver_no);
		return false;	// Don't update while the parameters are changing (network thread)- the calling function
						//  (Solver thread) should briefly pause, `continue;` in its loop, and update on a subsequent run.
	}
	this->params_changing = true;		//<-- Confusing! Solver has a `params_changing`!
	this->solver_status = SolverStatus::UpdatingParams;			//<-------
	//std::lock_guard<std::mutex> lockg(mtx_solverparams);		//don't rely on any changing params! check this mutex before access.

// === write Challenge (32 bytes) from the contract/pool into prefix/message ===		//sanity-check challenge string- already done? [TODO] <---
	const Uint256 challenge256 = Uint256(new_params->challenge_str.substr(2).c_str());	//parse hexstr to Uint256, omit `0x` hex specifier
	if (challenge256 != Uint256::ZERO)
		challenge256.getBigEndianBytes(this->hash_prefix);		//copy 32 bytes from uint256 to start of message (byte array)
	else {
		LOG_F(ERROR, "err: empty 256-bit challenge in NewMessage()!");
		return false;	//error
	}
//	memcpy(&this.hash_prefix[0], bytes_challenge, 32);		//then to the hash-prefix (bytes 0-31).

// === write Minting Address (20 bytes, Pool's or Miner's) into prefix/message ===
// in POOL mode, use the pool's minting address. in SOLO mode, use the Miner's.
	const std::string str_minting_address = gSoloMiningMode ? gStr_SoloEthAddress : new_params->mintingaddress_str;	//<-- orig.
	LOG_F(INFO, "got mint address:	%s", str_minting_address.c_str());	// <- dbg only
	if (!checkString(str_minting_address, MINTADDR_STR_LEN_WITH_0X, true, true))	return false;	//verify valid address is set: not null length, 20 hex bytes w/ `0x`
	if (!HexToBytes_b(str_minting_address.substr(2), &this->hash_prefix[32]))		return false;	//convert hexstr to byte-array, out to `bytes_address`	<--- CHECKME: & OF THE START OF THE ARRAY <---
//	memcpy(this.hash_prefix, bytes_challenge, 32);
// [TESTME]: 32-byte (256-bit) challenge to bytes 0-31 of hash_prefix. minting address to bytes 31-83.

	memcpy(this->initial_message, this->hash_prefix, 52);		// copy hash-prefix to first 52 bytes of initial message	[MOVEME]?
	if (VERSION_IN_SOLN) randombytes_buf(&this->initial_message[56], 28);	//write random bytes after the 52-byte prefix and 4-byte version (or)
	 else randombytes_buf(&this->initial_message[52], 32);	// entire solution nonce is random (after the 52-byte prefix).

//	memcpy(this->solution_bytes, &this->initial_message[52], 32);		//was solution[32] <---
	print_bytes(this->hash_prefix, 32, "challenge");				//dbg only
	print_bytes(&this->hash_prefix[32], 20, "minting address");			//dbg only
	print_bytes(this->hash_prefix, 52, "hash prefix[]");				//dbg only
	print_bytes(this->initial_message, 84, "initial mining message");
//	print_bytes(this->solution_bytes, 32, "solution[]");				//dbg only

//memcpy(this->midstate, mid, 200);					//copy 200-byte midstate
	ComputeMidstate((uint64_t*)&this->initial_message, this->midstate);	//args:input, output
	print_bytes((uint8_t*)this->midstate, 200, "midstate");	//debug only
//
	this->difficulty = new_params->difficultyNum;
	this->target = new_params->uint64_target;
	return true;	//solver has params now <--
}

//
//
// Converting this to a member function of GenericSolver class.
// Note to self, remove the initmesg_bytes (and prefix_bytes?) from the miningParameters typedef-struct
//	and instead write the output (incl. random bytes) to member byte-arrays in the genericSolver <----- sound good?
//
// Proxy-Mint test: put partial miner's address at start of solution nonce, the rest after the 8-byte counter value (`cnt`)?
// in this version, (NON PROXY MINT), it uses 4 bytes of version info, 28 random bytes.
bool genericSolver::SendToDevice(void)
{ // WIP: Consider passing the midstate and target to the kernel for this solver, instead of updating device memory. <----
	if (this->device_type == DeviceType::Type_CUDA)
	{

	}
//	else if (...) { ... }	// other device types
	else {
		// "Device types other than CUDA are not yet supported by genericSolver class"	//<----- [WIP].
		return false;	//Error
	}

	return false;	//Error
}




//bool Cuda_Call(const cudaError_t apiCallResult, const std::string theTask, const unsigned short deviceID);	// hashburner.cuh

// [MOVEME]
//typedef struct gpuStatus {
//	 double hashrate;
//	 double solvetime;
//	 ...
//} gGPUStatus[CUDA_MAX_DEVICES];

//gGPUStatus solvingGpuStatus[CUDA_MAX_DEVICES]{};
