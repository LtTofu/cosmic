// network.hpp : Common network defs for Pool and Solo mode
//				 ...and also the new network queue defs
#pragma once

#include <string>
//#include <thread>
//#include <queue>
//#include <mutex>
//#include <cinttypes>		// was: #include <stdint.h>

#include "coredefs.hpp"   // new
//#ifndef NETWORK_H
// ...
//#define NETWORK_H
//#endif


extern miningParameters gMiningParameters;	// network.cpp

extern bool g_params_changing;				// network.cpp
//...

extern uint64_t gU64_DifficultyNo;		// global 64-bit difficulty (For solvers). phasing out global params.

// function prototypes:
// ----
void ClearSolutionsQueue	(void);

void PopSolution			(void);

bool GetTxViewItem			(txViewItem *output_txn, 
							 unsigned int *txview_solution_no);

bool GetSolutionFromQueue	(QueuedSolution *out_soln);

int recurring_network_tasks (const bool compute_targ, 
							 const unsigned short exponent, 
							 const miningParameters old_params,	/* passed by value: current params for comparison. 	*/
							 miningParameters *params);		/* passed by reference: new params populate here.	*/

std::string	LibCurlRequest	(const char *data /*, CURLcode *result*/);

unsigned int GetSolnQueueContents ( void );
// defined in network.cpp (file which is compiled as native) for a couple reasons,
// one of which is that the <mutex> header is blocked when compiling with /clr.
GetParmRslt Pool_GetMintingAddress	(const bool	initial, 
									 miningParameters *params);

GetParmRslt Pool_GetChallenge		(const bool	initial, 
									 miningParameters *params);

GetParmRslt Pool_GetDifficulty		(const bool	initial, 
									 miningParameters *params);

GetParmRslt Pool_GetTarget			(const bool	initial, 
									 miningParameters *params);

//
enum class SubmitShareResult { SHARE_ACCEPTED = 0, SUBMIT_ERROR = 1, SHARE_REJECTED = 2 };  // [MOVEME]

SubmitShareResult Pool_SubmitShare	(QueuedSolution *p_soln, 
									 miningParameters *params);		// [new] 20200714. more OOP, less function params.
	
bool VerifySolution					(QueuedSolution *p_soln, 
									 miningParameters *params);

bool DoVerify						(const QueuedSolution *soln, 
									miningParameters *params);

uint64_t SetMiningTarget			(const Uint256 u256_newtarget, 
									 miningParameters *params);
//
//void Pool_ProcessSolution ( QueuedSolution* p_tempsoln );
//void Solo_ProcessSolution ( QueuedSolution* p_tempsoln );

void ResetCoreParams (miningParameters *params);

// [idea]  when possible just include this file (network.hpp) and not the individual headers for both modes :)
bool Pool_GetMiningParameters (const bool isInitial, const bool computeTarget, miningParameters *params);	//net_pool.cpp
bool Solo_GetMiningParameters (const bool isInitial, const bool compute_target, miningParameters *params);	//net_solo.cpp

