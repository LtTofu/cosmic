#pragma once
//#include <cinttypes>  // <inttypes.h>  ...for unsigned integer types (e.g. uint8 to 64_t)
//#include <cstdint> // <- TESTME
//#include <bitcoin-cryptography-library/Uint256.hpp>
#include <string>
#include <queue>

#include "defs.hpp" // <--   //-or-  constexpr auto CUDA_MAX_DEVICES = 19;
#include "coredefs.hpp"

#include "network.hpp"  // or:  extern miningParameters gMiningParameters;  // network.cpp/.h
// [WIP]  finish consolidating defs
//


std::string uint8t_array_toHexString ( const uint8_t* data, const int len );

int GenerateCpuWork(miningParameters* params);									// cpusolver.cpp

#include "util.hpp"
//void print_bytes  ( const uint8_t inArray[], 
//					const uint8_t len, 
//					const std::string desc );

unsigned short UpdateMiningParameters_new( miningParameters* params,
											const bool compute_targ,
											const unsigned short exponent );
//											const bool challengeChanged, 
//											const bool targetChanged, 
//											const bool mintingAddrChanged );	//Cosmic.cpp

//size_t JSONRPC_Response_Write_Callback(const char* contents, const size_t size, const size_t nmemb, const void* userp );
void SolutionSentSuccessfully ( const unsigned short dev_num, 
								const DeviceType dev_type, 
								const bool devshare );

void Pool_HandleVerifiedShare ( QueuedSolution *p_soln,		/* const? */
								miningParameters *params );
bool CheckDevShare ( void );

unsigned short Comm_CheckForSolutions ( miningParameters *params );
// check for any funcs moved to network.cpp ^  [TODO/FIXME]


// === externals ===
extern uint64_t stat_net_consecutive_errors;
extern ushort gVerbosity;
//extern uint64_t cpuThreadSolutionsFound[DEF_MAX_CPU_SOLVERS];

extern bool gSoloMiningMode;
extern uint64_t gNum_InvalidSolutionCount[CUDA_MAX_DEVICES];
extern uint64_t gNum_SolutionCount[CUDA_MAX_DEVICES];
extern uint64_t gNum_DevSharesSent;
extern unsigned int devshare_counter;
//extern uint64_t cpuThreadInvalidSolutions[DEF_MAX_CPU_SOLVERS];

extern std::string gStr_PoolHTTPAddress;
extern const std::string gStr_DonateEthAddress;
extern std::string gStr_MinerEthAddress;
