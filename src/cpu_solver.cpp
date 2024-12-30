#pragma once

// CPU_SSHA3.cpp: C++ Solidity-SHA3 Hashing Functions
//
// This is, for now, a simple recreation of the CUDA Solver (CUDA C kernel and host code),
// but controlling/calling into a native C++ keccak256 solver. Moving into the COSMICCPUENG lib

#include <stdio.h>
#include <iostream>
#include <cinttypes>
#include <string.h>
#include <array>
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
//#include <intrin.h>

#include <libsodium/sodium.h>  // move me <--
#include "defs.hpp"
#include "util.hpp"

#include "cpu_solver.h"
#include "network.hpp"

//#include "keccak256_engine.h"  // header file for matching native solver lib
//#include "CPUSolver_Native.h"
//#include "util.h"

//void print_bytes(const uint8_t inArray[], const uint8_t len, const std::string desc);  // net_pool.cpp

// previously in separate project. CPU mining library (COSMICCPUEENG.lib/.dll)
extern uint64_t cpu_target;  
//extern uint64_t cpu_mid[25];
//extern uint64_t cpuThreadBestResults[DEF_MAX_CPU_SOLVERS];

extern ushort gVerbosity;
extern uint64_t g64BitTarget;  // CosmicWind.cpp

using namespace System;
using namespace System::Windows;
using namespace System::Threading;
using namespace System::Windows::Forms;
using namespace System::Security::Cryptography;
using namespace System::Diagnostics;

//
// fixme/note to self: Don't access the mistate or target for a thread if it is being updated (add lock)
//

// Rotation Macros
#define ROTL64(x, y) (((x) << (y)) ^ ((x) >> (64 - (y))))
#define ROTR64(x, y) (((x) >> (y)) ^ ((x) << (64 - (y))))

// for CPU solver<->CosmicWind:
double cpuThreadHashRates[DEF_MAX_CPU_SOLVERS] { 0 };
uint64_t cpuThreadHashCounts[DEF_MAX_CPU_SOLVERS] { 0 };
uint64_t cpuThreadSolutionsFound[DEF_MAX_CPU_SOLVERS] { 0 };
uint64_t cpuThreadInvalidSolutions[DEF_MAX_CPU_SOLVERS] { 0 };
double gCpuThreadSolveTimes[DEF_MAX_CPU_SOLVERS] { 0 };
//uint64_t cpuThreadBestResults[DEF_MAX_CPU_SOLVERS] = { 0 };  // defined in cpueng
// (todo) consolidate these? ^

uint8_t solution_cpu[32] {0};  // 256-bit solution nonce (todo: separate for each CPU solver thread?)
unsigned int gCpuSolverThreads {0};
bool cpu_hasupdated {false};  // testing only <--


// CLEARCPUMININGVARS(): ...
void ClearCpuMiningVars(void)
{
	gCpuSolverThreads = 0; // redundant?
	
	for (unsigned int th = 0; th < DEF_MAX_CPU_SOLVERS; ++th)
	{
		gCpuThreadSolveTimes[th] = 0;
//		cpuThreadBestResults[th] = UINT64_MAX;
		cpuThreadHashCounts[th] = 0;
		cpuThreadHashRates[th] = 0;
		// TODO: anything else?
	}
}

extern uint64_t cpu_mid[25];// testing 
//void Cpu_NativeSolveLoop(void);  // move me  (see cpusolver_native.cpp)

// CPUSOLVERLOOP: Individual CPU solver loop on own thread. Move this function to a file compiled as native and use a different timer? <-- [TODO]
//#pragma unmanaged
void cpuSolverLoop(const unsigned short cpuThreadNum, miningParameters *params)  /* [WIP] */
{	
	Stopwatch^ sw1 = Stopwatch::StartNew();  // timer for performance measurement, etc.
	//uint64_t nonce = 0;  /* (CNT_OFFSET * 8) + (CNT_OFFSET * cpuThreadNum); */   // space out appropriately and separate search space from other CPU threads. <-- fixme!
	uint64_t nonce = CNT_OFFSET * cpuThreadNum;
	double seconds_this_solve{ 0 };

//	cpuThreadBestResults[cpuThreadNum] = UINT64_MAX;  // init val
	gCpuThreadSolveTimes[cpuThreadNum] = 0;  // 0 seconds

	printf("CPU solver # %d starting. The CPU Mining Target is: %" PRIx64 " \n", cpuThreadNum, cpu_target);
	//if (gVerbosity == 3)  print_bytes((uint8_t*)cpu_mid, 200, "cpu midstate");  // cast to array of uint8_t (bytes)

	uint64_t* keccakState = new uint64_t[25];  // actual mining state
	uint64_t* C = new uint64_t[5];					   // scratch array
	uint64_t* D = new uint64_t[5];				   // "
	uint64_t* scr = new uint64_t;					   // scratch uint64 (used?)
	memset(keccakState, 0, 200);  // init 200-byte (1600 bits), all 0s
	memset(C, 0, 40);					  // likewise for 40-byte buf C[] (320 bits)
	memset(D, 0, 40);					  // and D[]
	*scr = 0;								  // used? <-

	// if threads# is set to 0, mining cancelled by user. if application is closing, CPU threads end themselves.
	while (gApplicationStatus != APPLICATION_STATUS_CLOSING && gCpuSolverThreads > 0)
	{ // actual hashing loop. (todo: move this to native)
		if (cpu_hash(nonce, cpuThreadNum, keccakState, C, D, scr) == true) { //true: potential solution.	[TESTME] */
			LOG_IF_F( INFO, DEBUGMODE, ">> Potential Solution found by CPU solver #%u! (uint64 nonce: %" PRIx64 ") \n\n", cpuThreadNum, nonce);
			LOG_IF_F(INFO, NORMALVERBOSITY, ">> Solution Found by CPU Thread %u ! << \n", cpuThreadNum);
			//enqueue_solution_old(&nonce, DeviceType::Type_CPU, cpuThreadNum, params);  // 8-byte solution nonce for message <--- FIXME
			//continue;   <----- was here. seems counterproductive because of the following...
		}
		
		++nonce;  // ... was before cpu_hash() call above.	combine `nonce` and `cpuThreadHashCounts[]` into same var (array for threads)? [todo].
		++cpuThreadHashCounts[cpuThreadNum];  // one nonce tested / hash completed, iterate counter.

		// - first draft of time/performance measurement of CPU threads.  seconds represented as double (they're fractional, decimal precision available...) -
		seconds_this_solve = (double)sw1->Elapsed.TotalSeconds;  // extra cast, chasing a bug <-
		if (seconds_this_solve > 0) {  /* don't div by 0 */
			gCpuThreadSolveTimes[cpuThreadNum] = seconds_this_solve;  // for UI display. TODO: proper time structure (as necessary).
			cpuThreadHashRates[cpuThreadNum] = (double)(cpuThreadHashCounts[cpuThreadNum] / seconds_this_solve);  }
	} // while() over

	delete[] keccakState;  // free memory we allocated when CPU solver thread began:
	delete[] C;	
	delete[] D;	
	delete scr;
	return;
}

//
// CPU SOLVER THREAD CLASS
public ref class ThreadC
{
	unsigned short pubThreadNum;

public:
	ThreadC(const unsigned int threadNum) // function parameters would go here
	{ // Constructor
		pubThreadNum = threadNum;  // store thread # (serially assigned)
	}

	void cpuSolverThreadEntryPoint()
	{
		String^ threadName = Thread::CurrentThread->Name;
		Thread::CurrentThread->BeginThreadAffinity();

		if (gVerbosity >= V_NORM)  printf("CpuSolver: starting CPU thr %d for mining \n", pubThreadNum);
		//GenerateCpuWork();
		cpuSolverLoop( pubThreadNum, &gMiningParameters );  // mining will begin  [WIP]: pass reference to mining parameters via function arg? <-

		if (gVerbosity >= V_NORM) { printf("# CpuSolver: thread ending. \n"); }
	}

	~ThreadC(void /*const unsigned short threadNum*/)
	{ /* Destructor */ }
};


//FIXME
uint8_t dummy_cpusolution_bytes[32];
uint8_t dummy_cpuprefix_bytes[52];

// GENERATECPUWORK: Equivalent of send_to_device() for GPUs, but refreshes CPU threads' work instead
int GenerateCpuWork( miningParameters *params )
{
	// WIP: adjustable for tokens with different max difficulty
	uint8_t local_init_message[84]{ 0 };
	uint8_t random_bytes[24]{ 0 };

	LOG_IF_F(INFO, HIGHVERBOSITY, "Generating work for CPU threads");
	randombytes_buf(random_bytes, 24);							// get 24 random bytes (TEST: first 8 bytes of nonce are zeroes
	//memcpy(local_init_message, params->prefix_bytes, 52);		// bytes of the pool's minting address and the current challenge
	memcpy(local_init_message, dummy_cpuprefix_bytes, 52);		// bytes of the pool's minting address and the current challenge

	memcpy(&local_init_message[60], random_bytes, 24);			// 24 random bytes [60...83] from libsodium	[testme].
	// if challenge has updated since initial mining params:
	if (!cpu_hasupdated) {
	//	local_init_message[0] = 0x02;	// testing stuff
		cpu_hasupdated = true;
	}
	
	//memcpy(params->cpusolution_bytes, &local_init_message[52], 32);		// 256 bits/32 bytes
	memcpy(dummy_cpusolution_bytes, &local_init_message[52], 32);		// 256 bits/32 bytes
	StoreCpuThreadMidState((uint64_t*)local_init_message);

	if (DEBUGMODE) { /* debug stuff: */
		print_bytes(random_bytes, 24, "random bytes");
		print_bytes(local_init_message, 84, "cpu initial message");
		print_bytes(solution_cpu, 32, "solution[]" );	//solution "template"
		//printf("\n\n - Forcing easy cpu_target for testing! - \n");
	} else  cpu_target = g64BitTarget;  // was: Get64BitTarget(). see CosmicWind.cpp <--

	return 0;  // no error
}

// SPAWNCPUSOLVERTHREAD: Spawns a new, independent CPU thread to perform work that has
// already been Generated. Starts solving immediately, if successful returns true, othw. false.
bool SpawnCpuSolverThread(const unsigned int threadNo)
{
	ThreadC^ o1 = gcnew ThreadC(threadNo);  // instantiate
	Thread^ t1 = gcnew Thread(gcnew ThreadStart(o1, &ThreadC::cpuSolverThreadEntryPoint));
	//Console::WriteLine("thread " + cpuSolverThreads[i]->Name + " started \n");
	t1->Start();						// launch thread
	t1->BeginThreadAffinity();  // we don't want it moving (todo) <-

	// TODO: see if this works. Should be `true` if the thread started successfully
	return t1->IsAlive;
}