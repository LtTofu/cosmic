#pragma once

#ifndef COREDEFS_HPP
#pragma message( "including " __FILE__ ", timestamp: " __TIMESTAMP__ )
#define COREDEFS_HPP

#include <queue>
#include <string>
#include <bitcoin-cryptography-library/cpp/Uint256.hpp>

#define TEST_BEHAVIOR	//remove/comment this out

#include "types.hpp"

constexpr auto MAX_SOLVERS			= 128;			//256 <--
//constexpr auto CUDA_MAX_DEVICES	= 19;
#define CUDA_MAX_DEVICES		19
//#define DEF_MAX_CPU_SOLVERS	64



// if this file included by one compiled as native:
#if (_MANAGED == 0) && (_M_CEE == 0)
#include <mutex>

extern std::mutex mtx_solutions;   // for accessing the solutions queue
extern std::mutex mtx_events;	   // for accessing the events log in main window
extern std::mutex mtx_txview;	   // for accessing the transactions view items
//extern std::mutex mtx_globalparams;  // for solution[32], solution_cpu[32], hash_prefix[52], init_message[84]? etc.
extern std::mutex mtx_totxview;	   // for transport of transaction view item#s native<->CWind form
#endif
//the <mutex> header is blocked/unsupported in MSVC when compiling files with /clr option.

//typedef struct Solution_
//{ // === pool mode / common: ===
//	uint8_t		solution_bytes[32]	= { 0 };
//	std::string	solution_string		= "";				// 256-bit solution stored as a hexadecimal string. (only for display now?) [todo]
//
//	std::string	challenge_string	= "";	/* "0x0" */	// challenge this sol'n was solved for	(stored here to easily compare with current challenge)
//	std::string	digest_string		= "";	/* "0x0" */	// keccak256 digest after CPU verification
//
////=== any mode: ===
//	uint64_t	poolDifficulty		= 0;				// diff#.  note: uint256 maxtarget / difficulty# = uint256 diff. targ.
// //uint64_t	common_difficulty;						//			[todo]: for both modes
// //uint64_t	uint64_target;							//					<----
//// [TODO]: common difficulty# member for both modes.
//
//	int			deviceOrigin		= 0;				// solving device #
//	DeviceType	deviceType			= DeviceType::Type_NULL;
//	bool		verified			= false;			// TRUE if has passed CPU validation. Prevents re-verifying sol'ns in network outage.
//	bool		devshare			= false;			// if this is a devshare. left set if network error occurs (don't need to re-check)
//
////=== solo mode: ===
//	std::string	signature_r		= "";				// (solo mode) the signature's "r" portion, hex string
//	std::string	signature_s		= "";				// ditto, the "s" portion (hexstr)
//	std::string	signature_v		= "";				// v: a single byte	(hexstr)
//	long		solution_no		= -1;				// [FIXME]  unsigned int? <-  txview item#, indexed from 0. solo mode only!
//													//	(if this sol'n has been added to/is being submitted by Solutions View. item# indexed from 0)
//} QueuedSolution;

extern double			gNum_Hashrate[CUDA_MAX_DEVICES];
extern std::string		gpuName[CUDA_MAX_DEVICES];				// CUDA device names
extern uint64_t			gNum_DevSharesSent;

extern unsigned short gVerbosity;							// how many messages to show user, _MAX=debug behavior
extern unsigned short gApplicationStatus;
extern bool gCudaSolving;

extern bool gCudaDevicesDetected[CUDA_MAX_DEVICES];			// true if device detected, false otherwise
extern unsigned int gCudaDevicesStarted;					// how many devices were started at mining start.
extern int gSolo_ChainID;
extern std::string gStr_TokenName;

extern unsigned int gTxViewItems_Count;						//reset native count, beware async access. (phase out)

extern bool balloonShown;									// "still running in the BG" balloon

// network
extern bool stat_net_last_success;						// if last libcURL operation successful (true)			
extern std::string str_net_lastcurlresult;					// "OK" or the error string from libcurl	
extern bool gNotifiedOfSolSubmitFail;						// if user was notified of a failed share submit (net error)

// statistics
extern double stat_net_avgethop_totaltime;					// WIP (see above)
extern unsigned long long stat_net_badjson_responses;		// # of malformed JSON responses (or cloudflare pages)
extern unsigned long long stat_net_network_errors;			// # of network errors (non-OK HTTP codes)
extern double stat_net_avgethop_totaltime;									// solo mode
extern double stat_net_lastpoolop_totaltime, stat_net_avgpoolop_totaltime;	// pool mode

extern unsigned int		gNetInterval;						// init Network Access Interval (ms), 'normally' set by config/options
extern double			gAutoDonationPercent;				// default of 1.5%, read in from config file
extern unsigned int 	devshare_counter;					// counts up to 200 and resets
extern int				gDiffUpdateFrequency;				// read in from config file
extern const int		gMintAddrUpdateFreq;				// minting address retrieved from pool initially, and occasionally thereafter
extern std::string		gStr_ContractAddress;				// will be prepended with 0x. default contract: 0xBTC
extern bool				gSoloMiningMode;					// false: pool mode,  true: solo mode

extern uint64_t			stat_net_consecutive_errors;		// counter of bad responses from the pool (if any).

extern const std::string gStr_DonateEthAddress;				//
extern std::string		gStr_PoolHTTPAddress;				// TODO: pool select dialog box!

extern int				gLibSodiumStatus;					// 0=no error, 1=already inited, -1=error, not ready

extern bool				gStopCondition;						// stop triggered by an event other than Start/Stop Mining button. Handled by timer1.
extern bool				gComputeTargetLocally;				// the most significant 64 bits of a bignum target (sol'n digest must be <= to)

//extern struct txViewItem gTxViewItems[DEF_TXVIEW_MAX_ITEMS];

//#include "network.hpp"
extern std::queue<std::string>		q_events;
extern std::queue<QueuedSolution>	q_solutions;
extern std::queue<unsigned int>		q_totxview;

//#include "generic_solver.hpp"
//#include "hashburner.cuh"

// extern std::vector<genericSolver*> Solvers;
// extern std::vector<cudaDevice*> CudaDevices;


#pragma message( "not including " __FILE__ ", COREDEFS_HPP already defined" )
#endif	// COREDEFS_HPP