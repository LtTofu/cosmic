#pragma once
// cpusolver.h : CPU mining functionality
// 2020 LtTofu

#include "cpu_solver.h"
#include "defs.hpp"
//#include "../Core/Core.h"		// optionally
#include "network.hpp"

#define DEF_MAX_CPU_SOLVERS 64

// Macros:
#define ROTL64(x, y) (((x) << (y)) ^ ((x) >> (64 - (y))))
#define ROTR64(x, y) (((x) >> (y)) ^ ((x) << (64 - (y))))
// ...

// Forward Declarations:
void cpuSolverLoop(const unsigned short cpuThreadNum, miningParameters *params);
bool SpawnCpuSolverThread(const unsigned int threadNo);
void ClearCpuMiningVars(void);
void StoreCpuThreadMidState(const uint64_t* message);
//int GenerateCpuWork(MiningParams* params);
//uint64_t Get64BitTarget(void);			  // CosmicWind.cpp

//uint64_t bswap_64(const uint64_t input);  // see CPU engine lib.


// [todo]: define a custom type, reduce # of function parameters to cpu_hash()?
bool cpu_hash(const uint64_t nonce, const unsigned short theThread, uint64_t* state, uint64_t* C, uint64_t* D, uint64_t* scratch64);

// Externals:
extern double cpuThreadHashRates[DEF_MAX_CPU_SOLVERS];
extern uint64_t cpuThreadHashCounts[DEF_MAX_CPU_SOLVERS];
extern uint64_t cpuThreadSolutionsFound[DEF_MAX_CPU_SOLVERS];
extern uint64_t cpuThreadInvalidSolutions[DEF_MAX_CPU_SOLVERS];
extern double gCpuThreadSolveTimes[DEF_MAX_CPU_SOLVERS];
// for CPU solver<->CosmicWind.  [todo]: consolidate these. ^

extern unsigned short gVerbosity;
extern unsigned short gApplicationStatus;								// see CosmicWind.h

extern unsigned int gCpuSolverThreads;
extern uint64_t cpu_target;
extern uint64_t cpu_mid[25];

extern uint64_t gU64_DifficultyNo;										// or include "network.h"
//extern uint8_t hash_prefix[52];

extern uint8_t solution_cpu[32];  // 256-bit solution nonce (todo: separate for each CPU solver thread?)
//extern uint64_t cpuThreadBestResults[DEF_MAX_CPU_SOLVERS];			// init to UINT64_MAX (see cpusolver.cpp)
//extern bool cpu_hasupdated;  // testing only <--
