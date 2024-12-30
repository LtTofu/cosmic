#pragma once
// coredefs.cpp : just essential variables, mutexes etc.

//#include "coredefs.h"
#include "defs.hpp"
//#include "coredefs.hpp"

#include <queue>
#include <mutex>
#include "network.hpp"	//<--- remove?
// === queues and synchronization ===
std::queue<std::string> q_events;
std::queue<QueuedSolution> q_solutions;
std::queue<unsigned int> q_totxview;
// ^ WIP: improving queues implementation

// === only if compiling file as native ===
// note: the <mutex> header is blocked/unsupported when compiling with /clr.
#if (_MANAGED == 0) && (_M_CEE == 0)
std::mutex mtx_solutions;
std::mutex mtx_events;
std::mutex mtx_txview;
//std::mutex mtx_globalparams;	//mtx_coreparams
std::mutex mtx_totxview;
std::mutex mtx_txncount;  // <- new. used at submit time (solo mode only).
#endif
// ===

// application
ushort gApplicationStatus{ 0 };
ushort gVerbosity{ V_NORM };
bool balloonShown = false;		// only shows balloon the first time COSMiC is minimized to tray.
								// ( that session. save to Configuration? [TODO]. )

// mining on cuda devices (GPUs)
bool gCudaSolving{false};									// <--- make device-generic. [WIP]. <---
//bool gCudaDeviceEnabled[CUDA_MAX_DEVICES] {};				// ^


bool gSolvers_allocated[MAX_SOLVERS];	// [MOVEME]. <----

bool gDeviceEnabled[MAX_SOLVERS] {};	// ^

bool gCudaDevicesDetected[CUDA_MAX_DEVICES] {};				// <--- number of solvers instead,
															//		function-local cuda device count [WIP] <--
unsigned int gCudaDevicesStarted{ 0 };

double cuda_solvetime[CUDA_MAX_DEVICES] {};
std::string	gpuName[CUDA_MAX_DEVICES] {};

// Solo Mode:
bool gSoloMiningMode = false;
std::string gStr_TokenName{ "" };
struct txViewItem gTxViewItems[DEF_TXVIEW_MAX_ITEMS];
unsigned int gTxViewItems_Count{ 0 }; //<-- async access? (fixme)
//int gTxView_WaitForTx{ NOT_WAITING_FOR_TX };
int gSolo_ChainID = CHAINID_ETHEREUM;

// network
unsigned int		gNetInterval{ DEFAULT_NETINTERVAL_POOL };
double				gAutoDonationPercent{ 1.5 };
unsigned int 		devshare_counter{ 0 };
int					gDiffUpdateFrequency{ 50 };
const int			gMintAddrUpdateFreq{ 75000 };
std::string			gStr_PoolHTTPAddress{ "http://mike.rs:8080" };
std::string			gStr_ContractAddress{ "0xB6eD7644C69416d67B522e20bC294A9a9B405B31" };
//const std::string	gStr_DonateEthAddress{ "0xa8b8ea4c083890833f24817b4657888431486444" };	//constexpr?

// network statistics
unsigned long long stat_net_badjson_responses{ 0 }, stat_net_network_errors{ 0 };
bool stat_net_last_success{ true }, gNotifiedOfSolSubmitFail{ false };
std::string str_net_lastcurlresult{ "" };


double stat_net_avgethop_totaltime{ -1 };											//solo mode
double stat_net_lastpoolop_totaltime{ -1 }, stat_net_avgpoolop_totaltime{ -1 };		//pool mode

// core mining parameters & internals
double				gNum_Hashrate[CUDA_MAX_DEVICES] {};
uint64_t			gNum_DevSharesSent { 0 };


uint64_t g64BitTarget{ 0 };
int gLibSodiumStatus { -1 };  // 0=OK, inited. 1=already inited. -1=error initing
bool gStopCondition{false};
uint64_t stat_net_consecutive_errors { 0 };
unsigned int gLogicalProcessors { 0 };
//  Uint256 gU256_MiningTarget;  // for direct comparison only ( see gMiningTarget_bytes[32] )


//#include "generic_solver.hpp"
//#include "hashburner.cuh"

//std::vector<genericSolver*> Solvers;		// max desired size MAX_SOLVERS total 64-bit pointers to genericSolver instances	[WIP]
//std::vector<cudaDevice*> CudaDevices;	// max desired size CUDA_MAX_DEVICES total 64-bit pointers to cudaDevice instances		[WIP]
