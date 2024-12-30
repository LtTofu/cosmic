//#pragma once
// defs.h : Defines for COSMiC V4

#ifndef DEFS_HPP
#pragma message( "including " __FILE__ ", timestamp: " __TIMESTAMP__ )
#define DEFS_HPP

#include <string>
#include <stdint.h>  // standardize headers in use.
//#include <cstdint>
//#include <inttypes>
//#include <cinttypes>
#define ushort unsigned short
//#include "coredefs.hpp"	//<---

// compile-time debugging settings:
#define HWMON
#define TEST_BEHAVIOR						// enables some testing-specific behavior
#define DEBUG_NETWORK					0	// display and/or log network messages like parsed network responses, json requests etc.
#define DEBUG_NET_BGWORKER				0	// 
#define DEBUGGING_PERFORMANCE				// profile performance of timer, etc.
#define DEBUGGING_COSMICWIND_GUI			// 
#define DEBUG_PRINTBYTES				1	// print neat tables of bytes
//#define DEBUG_NET_PRINTJSONMESGS		0	// print/log network requests, replies. spammy.

#define STR_CosmicVersion "COSMiC v4.1.5 Dev TEST"

#define CUDA_MAX_DEVICES 19
//#constexpr auto CUDA_MAX_DEVICES = 19;			// [old]

// === keystore defs ===				// [TODO]: user-configurable encryption settings.
#define CIPHERTEXT_LEN (crypto_secretbox_xchacha20poly1305_MACBYTES + MESSAGE_LEN)

#define MESSAGE_LENGTH_STRING	64			// 32 bytes, or 64 hex digits.  was: 66 (included `0x`.)
#define MESSAGE_LEN				32			// bytes  was: 66 (included `0x`.)
#define SALT_LENGTH				32			// in bytes
#define DERIVED_KEY_LENGTH		32			// "

#define PUBLIC_KEY_LENGTH	64				// 
#define PBKDF2_ITERATIONS	900000			// for key derivation from password (todo: user-configurable?)
// ===

// - cpu mining mostly working. some implementation WIP. -
//constexpr auto DEF_MAX_CPU_SOLVERS = 64;	// consolidate w/ CpuSolver.h.
//constexpr auto CUDA_MAX_DEVICES = 19;		// for now.
//constexpr auto MAX_SOLVERS = 128;			// potentially solvers with different types [WIP].
											// multiple CPU threads should belong to a single solver.

//includes double-quote (") but not /\'
#define DEF_JSONCHARS R"delim(0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-:?!{}[] ")delim"
constexpr auto				DEF_MAX_NETRESPONSE_LEN			= 600;
constexpr unsigned short	DEF_POOL_RESPONSE_MAX_LENGTH	= 150;  // was `auto`

#define POOLMODE (gSoloMiningMode==false)
#define SOLOMODE (gSoloMiningMode==true)

extern unsigned short gVerbosity;
// verbosity for stdout and events viewer messages
constexpr auto V_LESS = 0;						// mostly quiet, just crucial info displayed
constexpr auto V_NORM = 1;						// typical verbosity, novice-friendly messages only
constexpr auto V_MORE = 2;						// details or extra info
constexpr auto V_DEBUG = 3;						// highest verbosity ("debug mode")
												// (should also get enabled automatically if any error reading config.)

#define DEFAULT_VERBOSITY	V_NORM
#define	HIGHVERBOSITY		(gVerbosity>V_NORM)
#define	NORMALVERBOSITY		(gVerbosity>=V_NORM)
#define DEBUGMODE			(gVerbosity==V_DEBUG)

// status of the application as a whole
#define APPLICATION_STATUS_STARTING		 0
#define APPLICATION_STATUS_READY		 1
#define APPLICATION_STATUS_MINING		 2
#define APPLICATION_STATUS_CLOSING		 3

// individual device status
enum class DeviceStatus { Null = 0, Initializing = 1, Starting = 2, Ready = 3, Mining = 4, 
							Stopping = 5, Fault = 6, MemoryFault = 7, Unavailable = 8 };

// computed target vs. pool-provided target
#define MAXTARG_EXPONENT_0XBITCOIN_ERC918	234  // 2^234 for 0xBitcoin, ERC918 tokens with the same MaxTarget

// pause conditions
//enum class DevicePaused { None = 0, FanRPMFail = 1, GPUOverheat = 2 };	//<-- [new]


// === for solo mining ===

#define CHAINID_ETHEREUM			 	1
#define CHAINID_ROPSTEN					3
//#define CHAINID_ETHEREUM_CLASSIC		??		// [todo]
// ...

// === transactions view ===
//

#define txViewColorSuccess Drawing::Color::PaleGreen	// <--- [MOVEME]
#define txViewColorFailed  Drawing::Color::Salmon		// <---

#define NOT_WAITING_FOR_TX	-1
//#define DEFAULT_NETINTERVAL_POOL	 400

// === ui stuff ===
#define DEF_EVENTS_LISTBOX_CAPACITY		8192
#define BUGIFHAPPENED " Please report this bug."	//append to strings
#define DEFAULT_HUD_UPDATESPEED			2
#define DEFAULT_EXECUTIONSTATE_SETTING  2

// === default values ===
#define DEFAULT_CUDA_INTENSITY			24
#define DEFAULT_PAUSE_ON_NETERRORS		20		// after this many consecutive network errors

// off-set the search space of multiple GPUs if using a shared initial message
#define CNT_OFFSET	35000000000000

// selectable keccak256 engines
#define CUDA_ENGINE_HASHBURNER			0		// highest performance
#define CUDA_ENGINE_COMPATIBILITY		1		// older devices, unusual compute versions

// threads-per-block (for work size) for different cuda device generations.
constexpr auto TPB50 = 1024u;				// compute_50 and up
constexpr auto TPB35 = 384u;				// compute_35 and down (old, not used).
constexpr uint64_t COUNT_OFFSET = 35000000000000;

// Node Response Types (not all-inclusive).		// (Solo mode)
#define NODERESPONSE_OK_OR_NOTHING				0		// no error or no node responses rec'd yet
#define NODERESPONSE_NONCETOOLOW				1		// txn with same nonce already exists
#define NODERESPONSE_REPLACEMENT_UNDERPRICED	2		// pending txn with same nonce already exists
#define NODERESPONSE_INSUFFICIENT_FUNDS			3		// insuf. ether balance to send solution txn
#define NODERESPONSE_INFURA_ACCOUNT_DISABLED	4		// Infura specific
#define NODERESPONSE_OTHER						5		// room to grow

// Network intervals
constexpr auto DEFAULT_NETINTERVAL_POOL = 400;		// only used if nothing set by the Configuration (which shouldn't happen.)
constexpr auto DEFAULT_NETINTERVAL_SOLO = 500;		// different access interval if in solo mode. mind the # of requests to endpoint.
constexpr auto DEFAULT_DIFFUPDATE_FREQUENCY = 50;	// <-

constexpr auto DEF_MAX_PARSED_LENGTH = 1800;  // reject excessively long responses (like web pages)
#define DEF_HEXCHARS "0123456789ABCDEFabcdef"
#define DEF_NUMBERS "0123456789"

#define HASH_LENGTH 32

constexpr auto HEXSTR_LEN_UINT256 = 64;
constexpr auto HEXSTR_LEN_UINT256_WITH_0X = 66;

// Timing stuff
enum Timings { null = 0, getTxReceipt = 1, getDifficulty = 2, getGasPrice = 3, getTxnCount = 4, getBalances = 5, getPoolMintAddr = 6, profTimer1 = 7 };
constexpr auto TIMINGS_COUNT = 8; // +1, array size

//enum class kernLaunchResult { launch_INIT = 0, launch_OK = 1, launch_SOLNFOUND = 2, launch_ERROR = -1, launch_ERROR_NONFATAL = -2 };  // [MOVEME]?

constexpr double DEFAULT_AUTODONATE_PERCENT = 1.5;


#define SET_MAX_TXSEND_ATTEMPTS			15
#define TXVIEW_SOLUTION_NULL -1
//
// refactor this as enum in types.hpp. [TODO]
//

// for adding solutions from solbuf into the Solutions View (Txview)
#define EMPTY_SLOT			-1			// if gSolutionsBuffer[?].solution_no == one of these
#define	ALREADY_IN_TXVIEW	-2			// -2: already added to the txview (to avoid adding a sol'n >once.)

// mode #s. for combobox_modeselect.	[old]
#define MODE_POOL			0
#define MODE_SOLO			1
#define MODE_HYBRID			2			// todo

//#define DEF_TXVIEW_MAX_ITEMS			300		// <---- TODO: enforce when adding new items!
constexpr auto DEF_TXVIEW_MAX_ITEMS = 300;		// <----
//
#define TXITEM_STATUS_EMPTYSLOT			-1		// item is not populated by a sol'n
#define TXITEM_STATUS_SOLVED			0		// solved and waiting for submit
#define TXITEM_STATUS_SUBMITTING		1		// actively submitting
#define TXITEM_STATUS_SUBMITWAIT		2		// waiting for retry
#define TXITEM_STATUS_SUBMITTED			3		// submitted, waiting for status update
#define TXITEM_STATUS_CONFIRMED			4		// waiting for retry
#define TXITEM_STATUS_FAILED			5		// submitted to network but failed
#define TXITEM_STATUS_TERMINAL			6		// exhausted retry attempts
#define TXITEM_STATUS_STALE				7		// unsubmitted solution for old challenge
//



bool checkDoEveryCalls ( const unsigned short whichEvent );

void AddEventToQueue ( const std::string theText );


#else
#pragma message( "not re-including DEFS." )		// __FILE__
#endif	//DEFS_HPP
