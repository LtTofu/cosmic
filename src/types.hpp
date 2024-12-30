#pragma once
#ifndef COSMIC_TYPES
#pragma message("Including " __FILE__ ", last modified " __TIMESTAMP__ ".")
#define COSMIC_TYPES
//#define ushort unsigned short

//enum CudaGpuStatus { gpuNull, gpuIdle, gpuInited, gpuMining, gpuFault };	//
//enum CudaGpuPause { notPaused, hwmonPause, networkPause, otherPause };		// wip
enum class ExceptionType { Null = 0, CtorFailed = 1, NoDevice = 2};

// some of these are probably extraneous [WIP].
enum class SolverStatus {
	Null = 0, Ready = 1, Starting = 2, NotSolving = 3, Solving = 4, WaitingForParams = 5, UpdatingParams = 6, 
	WaitingForNetwork = 7, Resuming = 8, Error = 9, DeviceError = 10, DeviceNotInited = 11 };

enum class PauseType { NotPaused = 0, WaitingForNetwork = 1, WaitingForDevice = 2, 
	GPUTempAlarm = 3, FanAlarm = 4, OtherReason = 5 };	//for solver's `pause`.

enum class DeviceType { Type_NULL = 0, Type_CUDA = 1, Type_CPU = 2, Type_VK = 3 };

enum class GetParmRslt { OK_PARAM_UNCHANGED = 0, GETPARAM_ERROR = 1, OK_PARAM_CHANGED = 2 };

enum class ParamsStatus { ParamsNotReady = 0, ParamsReady = 1, ParamsChanging = 2 };

enum class RawTxType { Null = 0, Mint = 1, Transfer = 2 };


// [TODO]: re-order members to avoid padding. `typedef` should no longer be needed. <--
// [TODO]: class instead of struct?
typedef struct Params_
{
//	(type?)	transaction_count {};
	uint64_t uint64_target{ 0 };		//for solving devices. derived from the uint256 target.
	uint64_t difficultyNum{ 0 };		//maxtarget is divided by this to get mining target
	uint8_t target_bytes[32]{};			//main mining target, computed or parsed from `target_string`
										// (uint256 represented as byte array)

										//base10 target string retrieved from a tokenpool -or-
	std::string	target_str = { "" };	//base16 (0x+hex) target string from contract (for solo mode).

	std::string	mintingaddress_str = { "" };	//pool's minting address -or- the miner's address (in solo mode).
	std::string	mineraddress_str = { "" };		//miner's address (for tokenpool's submitShare method).
	std::string	challenge_str = { "" };			//challenge as hex string

	bool	challenge_changed{ false };		//used in parameters retrieval functions
	bool	mintaddr_changed{ false };		//
	bool	difficulty_changed{ false };	//
	bool	target_changed{ false };		//
//
	bool	params_changing{ false };	//set TRUE when the first param change is detected by the Network Thread.
										//Solvers should pause mining until the parameters are ready.
} miningParameters;


// === TXview (Solutions tab) ===	[MOVEME]
struct txViewItem
{
	std::string txHash;				// 0xTxhash or an error from the node (processed)
	std::string errString;			// error string from the node last submit try, if any.

	unsigned short status;					// like TXITEM_STATUS_EMPTYSLOT ...
	unsigned short submitAttempts;			// attempts so far, from 0.
	uint64_t networkNonce;			// transaction count nonce starting from 0 with a new account

	unsigned short last_node_response;		// see NODERESPONSE_...

	std::string str_solution;

	std::string str_challenge;
	std::string str_digest;

	std::string str_signature_r;
	std::string str_signature_s;
	std::string str_signature_v;

	int solution_no;
	bool slot_occupied;  //

	unsigned short deviceOrigin;
	DeviceType deviceType;
};  // move to "txview.h"?

class genericSolver;
// added: default values to initialize objects of this type. compile as C++11. (placeholder name_ to resolve err C5208.)
typedef struct Solution_
{ // === pool mode / common: ===
	uint8_t		solution_bytes[32] = { 0 };
	std::string	solution_string = "";				// 256-bit solution stored as a hexadecimal string. (only for display now?) [todo]

	std::string	challenge_string = "";	// "0x0"	// challenge this sol'n was solved for	(stored here to easily compare with current challenge)
	std::string	digest_string = "";		// "0x0"	// keccak256 digest after CPU verification

//	=== any mode: ===
	uint64_t	poolDifficulty = 0;				// diff#.  note: uint256 diff. targ. = uint256 maxtarget / difficulty#
//	uint64_t	common_difficulty;				// [todo]: for both modes
//	uint64_t	uint64_target;					// [TODO]: common difficulty# member for both modes.

	genericSolver* solver;
	int			deviceOrigin = 0;				// solving device #
	DeviceType	deviceType = DeviceType::Type_NULL;

	bool		verified = false;			// TRUE if has passed CPU validation. Prevents re-verifying sol'ns in network outage.
	bool		devshare = false;			// if this is a devshare. left set if network error occurs (don't need to re-check)

 //	solo mode:
	std::string	signature_r = "";				// (solo mode) the signature's "r" portion, hex string
	std::string	signature_s = "";				// ditto, the "s" portion (hexstr)
	std::string	signature_v = "";				// v: a single byte	(hexstr)
	long		solution_no = -1;				// [FIXME]  unsigned int? <-  txview item#, indexed from 0. solo mode only!
												// (if this sol'n has been added to/is being submitted by Solutions View. item# indexed from 0)
} QueuedSolution;


//struct shared_mining_params {
//	miningParameters params;
//	ParamsStatus status;
//	std::mutex mtx;
//};


#else
#pragma message("Not re-including " __FILE__ ".")
#endif	//COSMIC_TYPES
