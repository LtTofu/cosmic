// COSMIC V4 ERC918 Token Miner by LtTofu
// 64-bit Windows software for pool mining of ERC-918 tokens such as 0xBitcoin, 0xLitecoin, KIWItoken, 0xCATE ...
// 

// COSMiC - "V4" (C++/.NET Windows GUI version), inspired by 0xbitcoin-miner by Infernal_Toast,
//					Zegordo, Mikers, Azlehria, 0x1d00ffff...

#define COSMIC
#define TEST_BEHAVIOR
// move to header.

//#include <string>	//redundant?
#include <loguru/loguru.hpp>
#include <libsodium/sodium.h>		//cryptography library
//#include "generic_solver.hpp"		// genericSolver class
#include "cuda_device.hpp"

#include <msclr/marshal.h>
//#include <msclr/marshal_cppstd.h>
//#include <msclr/marshal_windows.h>>
#include "defs.hpp"
//extern std::string gStr_ContractAddress;
extern int gSolo_ChainID;	//coredefs
//<-- or include "coredefs.hpp"

//#include "util.hpp"

using namespace System;

using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
//using namespace System::Data;
using namespace System::Drawing;
using namespace System::Text;
using namespace System::Threading;
using namespace System::Numerics;
using namespace System::Globalization;
//using namespace System::Windows::Forms;
using namespace System::Runtime::InteropServices;
using namespace System::Configuration;				//<---


//#define SAVE_PKEY_TOCONFIG	1
//#define SAVE_PKEY_TOKEYFILE 0
//#define SALT_LENGTH 32
//constexpr auto PUBKEY_LENGTH = 32;				// length, in bytes, of a key used to decrypt the keystore (from a user-supplied password)  <-- [MOVEME] ?
//constexpr auto CIPHERTEXT_BYTESLENGTH = 48;		// 80;		// [MOVEME] ?
//constexpr auto CIPHERTEXT_STRINGLENGTH = 160;	// for reference


//#include "Keystore.h"	//	[MOVEME] ?

//#include "Forms/OptionsForm.h"			// General Options & Pool Mode config dialog
//#include "Forms/AboutForm.h"			// About COSMiC Dialog
#include "Forms/GpuSummary.h"			// individual GPU statistics, etc
#include "Forms/ConfigIntensityForm.h"	// Set intensity of CUDA devices
#include "Forms/ConfigHwMon.h"			// Configure HW monitor/safety features
#include "Forms/MakeKeystore.h"			// 
#include "Forms/ConfigSoloMining.h"		// Configure Solo Mining dialog
//#include "Forms/EnterPassword.h"		// Simple password input dialog (for Solo Mode)
#include "Forms/TxReceiptForm.h"		// Simple transaction receipt viewer wind

#define WIN32_LEAN_AND_MEAN
#include <CosmicWind.h>		 // includes Windows.h, generic_solver.hpp


//#include "defs.hpp"			// includes "Core" defs

#include "network.hpp"
#include "net_pool.h"
#include "net_solo.h"	// -or-
//extern Uint256 gU256_TxnCount;
//#include "hashburner.cuh"
#include "cpu_solver.h"		// new
#include "hwmon.h"

#include <inttypes.h>	// only for format specifiers.

//namespace? {


void ClearEventsQueue(void);  // cosmicwind_native.cpp
//bool Cuda_Call(const cudaError_t apiCallResult, const std::string theTask, const unsigned short deviceID);	// hashburner.cu

#include <bitcoin-cryptography-library/cpp/Uint256.hpp>
//
uint64_t SetMiningTarget (const Uint256 u256_newtarget, miningParameters *params);		// network.cpp
int DoMiningSetup(const bool compute_target, const unsigned short maxtarget_exponent, miningParameters* params);		// network.cpp

//extern bool gDeviceManuallyUnpaused[CUDA_MAX_DEVICES] = { false };


// INIT_SODIUM: Perform needed initialization for LibSodium, store initialization status
bool Init_Sodium(void)
{ // Impt Note:  COSMiC requires the (provided) libsodium.dll to facilitate Solo Mode's keystore encryption
  //			 and the application probably won't launch without it present.
	LOG_F(INFO, "Initializing LibSodium...");
	gLibSodiumStatus = sodium_init();

	if (!gLibSodiumStatus) {
		printf("Success! \n");
		LOG_F(INFO, "Initialized Sodium successfully.");
		return true; }  // OK
	else if (gLibSodiumStatus == 1) {
		printf("already inited! \n");
		LOG_F(INFO, "Sodium already initialized.");
		return true; }  // OK
	else {
		printf("error %d! \n", gLibSodiumStatus); // presumably -1 per libsodium docs
		LOG_F(ERROR, "Error while initializing Sodium.");
		return false; }  // Err

	// TODO: solo mode should be disabled if LibSodium not inited- keystore will be unavailable
	//		 and some other functions used like randombytes_* won't work <---
	return true;  // OK, inited
};

/*_inline*/ Drawing::Color GetWarningColor(const unsigned short deviceIndex, int scratchTemp)
{ // GETWARNINGCOLOR(): returns .NET system color corresponding to a temperature reading in degrees C
	scratchTemp = gWatchQat_Devices[deviceIndex].gputemp;  // store temperature in param (pass 0.)
	if (scratchTemp < 20 && scratchTemp > -999)  // see below:
		return Drawing::Color::LightBlue;    // chilly! probably subambient
	else if (scratchTemp < 75 || scratchTemp <= -999)
		return Drawing::Color::White;	     // good temperature (or null) = white
	else if (scratchTemp >= 74 && scratchTemp < 78)
		return Drawing::Color::LightYellow;  // decent temperature = yellowish 
	else if (scratchTemp >= 78 && scratchTemp < 82)
		return Drawing::Color::LightSalmon;	 // acceptable temperature
	else if (scratchTemp >= 82 && scratchTemp < 85)
		return Drawing::Color::LightPink;	 // getting hot = pinkish
	else if (scratchTemp >= 85 && scratchTemp < 87)
		return Drawing::Color::Orange;		 // arguably too hot = brighter
	else if (scratchTemp >= 87)
		return Drawing::Color::Red;			 // overheating = red
	else {
		printf("GetWarningColor(): Device %d, got out-of-range temperature value %d.\n", deviceIndex, scratchTemp);
		return Drawing::Color::LightGray;  }
}


void CWind_DebugInfo( miningParameters *params )
{
	if (gVerbosity < V_DEBUG)  return;
	uint8_t scratchBytes[32] = { 0 };

	printf("Available Devices\n--\n");
	const unsigned short num_of_solvers = static_cast<unsigned short>( Solvers.size() );
	for (unsigned short dev = 0; dev < num_of_solvers; ++dev) { // [WIP]: make device #'s type-generic.
		if (Solvers[dev] != nullptr) { // if (device_solver_allocated[dev]) ... <---
			printf("Device # %d: ", dev);	// was: CUDA Device- Making device type-generic		[WIP]
			Solvers[dev]->enabled ? printf("intensity %d	( %" PRIu32 " threads )\n", Solvers[dev]->intensity, Solvers[dev]->threads) : printf("disabled \n");	// [FIXME]. <--
		} else {
			LOG_F(ERROR, "Bad device# %u: Solver does not exist!", dev);	// Bug if happened
			return;
		}
	}
	printf("\nTxView Items counter: %d \n", gTxViewItems_Count);
	printf("Solo Mode setting: %d \n", static_cast<int>(gSoloMiningMode));	//bool

	printf("CPU Mining stuff: \n");
	printf("- CPU Solver Threads(active?): %d \n", gCpuSolverThreads);
	printf("- Difficulty Target (CPUs): %" PRIx64 " (64-bit) \n", cpu_target);
//	printf("- Difficulty Target (GPUs): %" PRIx64 " (64-bit) \n", g64BitTarget);
	print_bytes( params->target_bytes, 32, "256-bit mining target" );

	printf("Solo Mode params: \n");
	printf("- Gas Price: %" PRIu64 " gwei (decimal) \n- Gas Price (hexstr): %s gwei \n", gU64_GasPrice, gStr_GasPrice_hex.c_str());
	printf("- Gas Limit: %" PRIu64 " (decimal) \n- Gas Limit (hexstr): %s \n", gU64_GasLimit, gStr_GasLimit_hex.c_str());
	printf("- Chain ID: %d \n", gSolo_ChainID);
	printf("- SoloMode Network Access Interval: %d ms \n", gSoloNetInterval);
	printf("- API Endpoint (node): %s \n", gStr_SoloNodeAddress.c_str());
	printf("- Mineable Token Contract Address: %s \n\n", gStr_ContractAddress.c_str());
	printf("* Transactions Count (network nonce):  0x%s \n", gTxCount_hex.c_str());	// <- 
	printf("* Solutions Buffer contains: %zu items \n\n", q_solutions.size() );

	printf("gStr_MiningTarget: %s \n", params->target_str.c_str());					// <-
	printf("Difficulty #: %" PRIu64 " \n", params->difficultyNum);					// <- [WIP].

	printf("\nTiming Settings: \n");
	for (uint8_t i = 0; i < TIMINGS_COUNT; ++i)
		printf( " [%u]= %u ", i, doEveryCalls_Settings[i] );
	printf("Current Values: \n");
	for (uint8_t i = 0; i < TIMINGS_COUNT; ++i)
		printf( " [%u]= %u ", i, doEveryCalls_Values[i] );
}

// ComputeMiningTarget(): Computes the Mining Target w/ the token contract's Max Target (specified w/ arg), as Uint256.
//	In the case of 0xBTC-alikes, maxTarget is 2^234, Mining Target is that / by the Difficulty #. Returning Uint256::ZERO is an error.
// [TODO / WIP]: make the maxTarget exponent a function param to ComputeMiningTarget().
BigInteger ComputeMiningTarget(const uint64_t difficultyNum, unsigned short exponent) /* make the 2nd argument an int? [todo] */
{
	msclr::interop::marshal_context mctx;
	VLOG_F(V_MORE, "Computing mining target locally  (from MaxTarget: 2^%d, difficulty #: %" PRIu64 " )", exponent, difficultyNum);
	if (!difficultyNum) { /* don't div. by zero */
		LOG_F(ERROR, "ComputeMiningTarget(): invalid difficulty %" PRIu64 ", no target computed", difficultyNum);
		return BigInteger::Zero; }	//0=err
	if (exponent < 1 || exponent >= UINT8_MAX) { /* somewhat silly since the function arg. is a uint8. consider changing to int (see below) */
		LOG_F(WARNING, "ComputeMiningTarget(): bad maxTarget exponent %d. using default of 234 (0xBTC-style)", exponent);
		exponent = 234; }	//default to 0xBTC maxtarget <--

// [todo] BigInteger::Pow() expects exponent as int. consider making the 2nd arg of this function int type and removing the cast below.
//	auto big_target = BigInteger::Pow( BigInteger(2), static_cast<int>(exponent)) / BigInteger(difficultyNum);	//condensed ver.
	auto max_target = BigInteger::Pow( BigInteger(2), static_cast<int>(exponent) );		// 2^(exponent) / difficultyNum
	auto big_target = max_target / BigInteger( difficultyNum );
	LOG_IF_F(INFO, HIGHVERBOSITY, "    max target:	%s ", mctx.marshal_as<std::string>(max_target.ToString("x64")).c_str());
	LOG_IF_F(INFO, HIGHVERBOSITY, "uint256 target:	%s ", mctx.marshal_as<std::string>(big_target.ToString("x64")).c_str());
//	Console::WriteLine("uint256 target:  " + big_target.ToString("x64"));

	return big_target;  //bigInt_ComputedTarget;
}

// MiningTargetFromPoolTarget(): Takes a string representing the Mining Target from pool/node and populates gU256_MiningTarget (256-bit unsigned bignum)
//								 Also gets a 64-bit uint target for the GPUs to solve for. Called by FreshTarget().
BigInteger MiningTargetFromPoolTarget(std::string& targetString)
{	// TODO: phase out gStr_MiningTarget and pass directly from net thread on retrieval (if changed) via getter func? public member of backgroundWorker?
	// check for bad input.  EDIT: don't expect a specific length. Pool target is a _decimal_ (base 10 representation in JSON) not base-16:
	//	max targetString length of 90 (a little bigger than enough to accommodate a base10 string representation of 2^234.)
	//if 0 is returned (not a valid target), calling func should treat it as an error

  //  Clean up this function, some first-draft junk in here <--- [WIP/FIXME]!
	bool is_hexstring = false;
//	domesg_verb("Parsing 256-bit target from decimal targetStr: " + targetString, true, V_DEBUG);  // TODO: check for 0x in targetString?

	const bool bExpectHexString = gSoloMiningMode;					// POOL:  expect base10 string.			SOLO:  expect base16 (hex).
	const unsigned short expectLength = gSoloMiningMode ? 66 : 0;	// POOL:  no specific length expected.  SOLO:  expect 0x + 256-bit #, 66 chars long

	if (checkString(targetString, expectLength, bExpectHexString, bExpectHexString)) {
//	if (targetString.substr(0, 2) == "0x") {	/* check validity of the rest of the string. <--- [FIXME] */
		targetString = targetString.substr(2);	// trim off "0x" if present
		is_hexstring = true; }					// parse as base-16
	else {
		domesg_verb("bad target string.", true, V_DEBUG);
		return BigInteger::Zero; }  // 0=err

	// debug stuff:
	if (bExpectHexString) {	/* base16, solo mode: */
		if (!checkString(targetString, 64, false, true)) {		/* don't expect `0x`. Redundant, see above */
			domesg_verb("bad target string.", true, V_DEBUG);
			return BigInteger::Zero; }
	} else {				/* base10, pool mode: */
		if (targetString.length() < 1 || targetString.length() > 78) { /* [CHECKME] uint256 maxval expressed as base10 dec. string: 78 digits long. */
			domesg_verb("Target received from pool has unexpected length " + std::to_string(targetString.length()) + " (expected 66)", true, V_DEBUG); // <--
			return BigInteger::Zero; }
		// ^ [WIP]: targetString length check.  any logical reason an erc918-ish contract would dramatically increase maxTarget? ^
		if (!checkString(targetString, 0, false, false)) {
			domesg_verb("bad input while parsing base10 target.", true, V_MORE);
			return BigInteger::Zero; }


		if (!is_hexstring && targetString.find_first_not_of(DEF_NUMBERS) != std::string::npos) {
			domesg_verb("invalid characters while parsing (base10) target received from pool: " + targetString, true, V_DEBUG);
			return BigInteger::Zero; }
	}

	System::Numerics::BigInteger newBig = System::Numerics::BigInteger::Zero;
	System::IFormatProvider^ iFormat = gcnew System::Globalization::CultureInfo("en-US");  // for parsing
	String^ scratchMStr = gcnew String(targetString.c_str());  // to new managed string for use w/ BigInteger class

	if (is_hexstring)  //if (!gSoloMiningMode)
	{ // (solo mode:)  target from contract should be a 256-bit hex string ("0x" already trimmed off above)
		if (BigInteger::TryParse(scratchMStr/*->Substring(2)*/, System::Globalization::NumberStyles::HexNumber, iFormat, newBig) != true) {
			domesg_verb("Couldn't parse Mining Target retrieved from the node. ", true, V_MORE);  // out to newBig ^
			return BigInteger::Zero; }
	} else
	{ // (pool mode:) - target from pool should be an integer (base 10)
		if (BigInteger::TryParse(scratchMStr, System::Globalization::NumberStyles::Integer, iFormat, newBig) != true) {
			domesg_verb("Unable to parse the Mining Target retrieved from the pool. ", true, V_MORE);  // out to newBig ^
			return BigInteger::Zero; }
	}
	return newBig;
}

void Test_CompareTargets (const uint64_t diffNum, const unsigned short exp, miningParameters *params)  /* [MOVEME] */
{
	BigInteger test_computedTarget = ComputeMiningTarget(diffNum, exp);
	BigInteger test_parsedTarget = MiningTargetFromPoolTarget( params->target_str );  // <--- function param instead!
	Console::WriteLine("Computed Target: {0} ", test_computedTarget);
	Console::WriteLine("Parsed Target  : {0} ", test_parsedTarget);
	if (test_computedTarget == test_parsedTarget) { Console::WriteLine("The targets match. "); }
	  else { LOG_F(WARNING, "Test_CompareTargets(): tested targets do not match!"); }
}


// FRESHTARGET(): Runs after a new target is received and at mining start. Called by
//   DoMiningSetup() and UpdateGpuMiningParameters(). If `compute_locally` is true, the uint256 target will be computed
//	  from the configured maxtarget (default: 2^234) divided by `difficulty_num`. If false, mining target will be
//	  retrieved, parsed and `difficulty_num` will be ignored
//	- In a break from tradition- func returns the uint64 target, or 0 if error (0 isn't a solveable target).
uint64_t FreshTarget (miningParameters *params, const bool compute_locally, const unsigned short exponent)
{ //make mutex a member of the params struct? [todo]
	if (!params->difficultyNum && compute_locally) { // diff# needed to compute target. but not to parse the pool's target.
		domesg_verb("FreshTarget(): bad difficulty # of 0 \n", true, V_DEBUG);
		return 0; } //err

	msclr::interop::marshal_context marshalctx;
	BigInteger big_target{ BigInteger::Zero };

	// compute mining target from contract MaxTarget and difficulty #, or parse the pool-provided target- a 256-bit unsigned integer represented as base10 (not hex). Parsing to byte array
	compute_locally ? big_target = ComputeMiningTarget(params->difficultyNum, exponent) : big_target = MiningTargetFromPoolTarget( params->target_str );  //[TESTME]: verifying operation in both modes.
	if (big_target == BigInteger::Zero)
	{ // should never happen. but if a bug were to make the target zero:
		LOG_F(WARNING, "Target is zero- no solutions will be found! Aborting mining");		// can't solve for target of 0
		domesg_verb("Target is zero- unable to proceed. Please check settings and network connection.", true, V_LESS);	//<--- inform user [WIP].
		gStopCondition = true;																			// <-- [TESTME]: stop to save power, just in case.
		return 0;																						// don't send this to the device- stop mining.
		// ...
	}
	//LOG_IF_F(INFO, DEBUGMODE, "New Target (bigint):  %s ", marshalctx.marshal_as<std::string>(big_target.ToString("x64")).c_str());		// "
	if (DEBUGMODE) { Console::WriteLine("New Target (BigInteger):  " + big_target.ToString("x64")); }

// check if big_target is zero or otherwise invalid. rather than in the 2 functions called above. <-- [ TODO / WIP ]
// [todo/fixme]: slightly convoluted conversion adventure
	Uint256 u256_target{ Uint256::ZERO };
	u256_target = Uint256(marshalctx.marshal_as<std::string>(big_target.ToString("x64")).c_str());	// marshal 32-byte hex representation of `big_target`, as c-string, to appropriate uint256 constructor.																										-or-: send them the initial parameters.		<--
	if (u256_target == Uint256::ZERO) { return 0; }		//

	const uint64_t u64_target = SetMiningTarget( u256_target, params );	 // sets target in params (has lock-guard)
	if (!u64_target) { return 0; }					// "

// OK: everything should be set now.
	LOG_IF_F( INFO, DEBUGMODE, "FreshTarget(): got 64-bit target of  %" PRIx64 "  ( dec: %" PRIu64 " ).", u64_target, u64_target );
//	domesg_verb("FreshTarget(): got 64-bit target of:  " + std::to_string(u64_target), true, V_DEBUG);  // <-- [wip]
	return u64_target;	// OK: also return the uint64 target.
}



// [WIP]: version to update the gMiningParameters, but does not push new params to device(s). Instead `new_params_available` is set
//			in the existing Solvers and they can pull the latest params on their own (in genericSolver::Update()).
// [TODO]: make this function native?
// // [WIP]: FreshTarget() is a managed function and uses .NET's Numerics::BigInteger class. So if this function is made
// //		native, there will be additional transitions between native and managed code. However, the <mutex> header is blocked
//			when compiling with /clr. So, using `g_params_changing` to keep Solver threads informed not to generate/enqueue any
//			solutions that we know will be stale.
//
//unsigned short UpdateMiningParameters_new(miningParameters* params, const bool compute_targ, const unsigned short exponent,
//	const bool challengeChanged, const bool targetChanged, const bool mintingAddrChanged)
unsigned short UpdateMiningParameters_new(miningParameters* params, const bool compute_targ, const unsigned short maxtarg_exponent)
{	// UPDATEGPUMININGPARAMS: Rebuilds hash_prefix, init_message with the new challenge and the pool's minting address.
	//							Sets `new_params_available` in each Solver object, so they can retrieve parameters on their own.

	// [WIP]: Currently using common symbols (midstate, target) for all devices.
	// The common symbols must be updated ONCE from the network thread (ideal) or main thread
// FreshTarget() uses the newly-retrieved target string (gStr_MiningTarget), or difficulty# (in the case of locally computed target.)
// Populates diff/target-related parameters
	LOG_IF_F(INFO, DEBUGMODE, "UpdateMiningParameters_new():  challengeChanged: %d, targetChanged: %d, diffChanged:%d, mintingAddrChanged: %d, compute_targ: %d",
		(int)params->challenge_changed, (int)params->target_changed, (int)params->difficulty_changed, (int)params->mintaddr_changed, (int)compute_targ);
	if (gSoloMiningMode)										/* Clear. Stop waiting on any pending Tx, if challenge changes.				*/
		gTxView_WaitForTx = NOT_WAITING_FOR_TX;					/* Not explicitly reset to -1 when a Pending Tx is confirmed, because the	*/
																/* challenge will change regardless. However, if the Tx Fails, it is reset.	*/
	const uint64_t newtarg_64 = FreshTarget(params, compute_targ, maxtarg_exponent);
	LOG_IF_F(INFO, DEBUGMODE, "UpdateMiningParameters(): new 64-bit target: %" PRIx64 " (hex.) | %" PRIu64 " (dec.) \n", newtarg_64, newtarg_64);
	if (!newtarg_64) { /* target of 0 is not valid */
		domesg_verb("Error occurred while processing new Mining Target: target is zero.", true, V_MORE);
		return 1;
	}

	const unsigned short num_solvers = static_cast<unsigned short>(Solvers.size());
	// === new parameters available for Solvers ===
#ifdef PUSH_SOLVER_PARAMS
		// Iterate through the enabled CUDA GPUs and update them
		bool pushResult = push_params( params );		//<---- [WIP].
		LOG_IF_F(INFO, HIGHVERBOSITY, "Updated %u devices successfully", num_solvers);
#else
	//const unsigned short num_solvers = static_cast<unsigned short>(Solvers.size());
	if (num_solvers > 0 && num_solvers < MAX_SOLVERS) {
		for (unsigned short i = 0; i < num_solvers; ++i)
		{ //the Solvers should update themselves. THE PARAMS MUST BE READY TO GO IN `gMiningParameters`! Use Mutex `mtx_coreparams` ? <---
			if (!Solvers[i]->enabled) continue;
			LOG_IF_F(INFO, DEBUGMODE, "Setting Solver # %u new_params_available", i);
			Solvers[i]->new_params_available = true;	//<----
		}
	} else {
		LOG_F(ERROR, "Bad Solvers count:	%u (%zu) !", num_solvers, Solvers.size());
		return 3;	//err
	}
#endif

	// CUDA Devices using common symbols. [WIP]: Solvers w/ independent parameters
	//if (!update_device_symbols(params)) {
	//	LOG_F(ERROR, "Error updating common device symbols!");
	//	return false;
	//}
	return false;	//err
}
//}
/* Note to self: Network BGworker thread.												*/


//
// [WIP]: Just common params for all devices! Initialize device-specific stuff from their genericSolver instance. <----
//
int DoMiningSetup(const bool compute_target, const unsigned short maxtarget_exponent, miningParameters *params)
{ // was: initial_parameters (passed by value). Temporarily passing reference to global `gMiningParameters`.
//	uint64_t newtarg64 {0};  // [moveme]?

	ClearSolutionsQueue();  // for solutions to be verified/submitted
	ClearEventsQueue();	     // for events list box inputs from native

//	depending on mode, get updated mining parameters from the configured pool/node, verify success:
//	passing isInitial `true` will retrieve minting address, too- in case another pool was selected or the pool itself is reconfigured
// [fixed]:  isInitial was false.
	const bool result = gSoloMiningMode ? Solo_GetMiningParameters(true, compute_target, params) : 
		Pool_GetMiningParameters(true, compute_target, params);		//get params for Pool or Solo mode
	if (!result) { // <- use enum? [todo / fixme].
		domesg_verb("Error getting mining parameters. Please check your pool/node address:port, network connection and try again.", true, V_NORM);
		return 1;	//err
	}

	// [WIP] / [FIXME]: use `params` rather than gMiningParameters- pass reference to the parameters defined at a broader scope rather than using a global,
	//			  this should make sure the params are clear anytime mining re-starts, cut down on globals, hopefully make code easier to follow. <---
	//			  Gets/stores mining target as 32-byte array, gets uint64 target for GPU/CPU solvers, etc.
	const uint64_t newtarg64 = FreshTarget( params/*&gMiningParameters*/, compute_target, maxtarget_exponent );
	if (!newtarg64) { /* err: 0 is not a valid target. */
		domesg_verb("Error occurred while processing new mining target.", true, V_MORE);
		return 2;	//err
	} else
		LOG_IF_F(INFO, DEBUGMODE, "DoMiningSetup(): 64-bit target:  %" PRIx64 " ", newtarg64);	//dbg <-

// === [note]:	At mining start, this func. is responsible for sending mining parameters to the solvers' associated device. Either individually ("Approach A"), 
//				or by copying common parameters to Symbols used by all CUDA devices ("Approach B").
// [impt.]: Doing so from the MAIN thread. During mining, Solvers and their associated Device's parameters will be updated from the NETWORK thread.

// [idea]:	Update the devices THEMSELVES from THE INDIVIDUAL SOLVER's thread instead (pause kernel launching while doing so)

// === iterate through and send midstate and 64-bit uint target to CUDA device(s).	"approach A". ===
// [note]:  this approach to be used when each device has its own initial message, midstate and target.
//			(they can share the same challenge/mint address/target, but not the whole initial message)
//	for (unsigned short cudadev_index = 0; cudadev_index < CUDA_MAX_DEVICES; ++cudadev_index) {
////	if (!gCudaDeviceEnabled[cudadev_index])
//			continue; // skip disabled/absent devices
////	if (cuda_allocatevars(cudadev_index) > 0)
//			return 1;
////	if (send_to_device(cudadev_index, newtarg64, (uint64_t*)params->initmesg_bytes)) {
//			if (p_genericSolvers[cudadev_index]->Update(params)) { 
//			LOG_IF_F(ERROR, NORMALVERBOSITY, "Error sending initial work to CUDA device #%d!", cudadev_index);
//			return 1;
//		} //if.
//	}

//	=== copy to a single set of symbols in constant memory for all CUDA GPUs to use ===
// [note}: this approach assumes each solver/device's `cnt` starts offset from 0 to separate their search spaces
	
	//update_device_symbols( params );		// "approach B". <--------

	const ushort num_solvers = static_cast<ushort>(Solvers.size());
	if (!num_solvers || num_solvers >= MAX_SOLVERS) {
		LOG_IF_F(ERROR, DEBUGMODE, "Bad number of solvers (%u) in DoMiningSetup()!" BUGIFHAPPENED, num_solvers);
		return 5;	//err: no or too many solvers exist
	}

	LOG_IF_F(INFO, DEBUGMODE, "New mining parameters ready for %u Solvers", num_solvers);
	for (ushort i = 0; i < num_solvers; ++i) {
		if (Solvers[i] != nullptr)
			Solvers[i]->new_params_available = true;	//solvers should now update themselves from the global mining params
		 else {
			LOG_F(ERROR, "Null solver (# %u) in DoMiningSetup()!" BUGIFHAPPENED, i);	//redundant
			return 6;	//error
		}
	}

	return 0;	//OK
}


// ===== ===== ====
std::string DeviceType_Str[4] = { "NULL", "CUDA", "CPU", "OTHER" };
//
// GPU solvers on independent CPU threads
public ref class SolverThread
{
public:
	DeviceType device_type;
	ushort device_no;
	ushort solver_no;
	genericSolver* p_solver;
//	miningParameters* initial_params;	// [todo]/[idea]: independent mining params stored in each solver.

// [FIXME]: make fully generic :) There is still some cuda specific-stuff. Don't mix up solver# and cuda device# !
// BUT if the GPUs are all CUDA, their numbers (indexed from 0) should coincide. Solvers for non-CUDA devices will be
// instantiated and numbered _after_ any CUDA devices in the system. CPU mining threads to be grouped by a _single_ solver [todo]. <---
public:
	// note to self: `solver` is a pointer to a native class object, passed to a managed class' constructor <--- [WIP].
	SolverThread (/*const*/ genericSolver* solver /*, const miningParameters initParams */)
	{ //constructor:
		this->device_type = solver->device_type;
		this->device_no = solver->api_device_no;
		this->solver_no = solver->solver_no;
		this->p_solver = solver;

		// skip sanity checks for clarity, and check type/# in the instantiating function?

		// check valid CUDA device#, or id# of other solver type. [WIP] / [TODO]
		if (device_type == DeviceType::Type_CUDA)
		{
			if (solver_no > CUDA_MAX_DEVICES /* || ... */) { // anything else? Consider doing all checks before instantiating. <--- [WIP].
				LOG_F(ERROR, "In SolverThread(): bad device number!");
				this->device_type = DeviceType::Type_NULL;	// solver should not start.
				// [TODO]: do not start the device. Destroy SolverThread. Throw exception here? <---- [TODO] / [FIXME].
				//			(Do _not_ expect the Destructor to be called, since no SolverThread object was instantiated.)
			}
		}
		else /*if (device_type == DeviceType::Type_VK) */ {
			LOG_F(INFO, "[TODO]: Support other GPU types.");
			//throw(...);	// [TODO]. catch in instantiating func.!
		}

		// [WIP] / [TODO].  Initial parameters?
	}

	~SolverThread ()
	{ //destructor:
		LOG_IF_F(INFO, DEBUGMODE, "~SolverThread() ran.");
		// [idea]: do per-device cleanup here instead?
	}

public:
// [WIP]:  hook this part up... run CUDA_Solve in the GpuSolver instance. <----
	void SolverThread_entry()
	{ // starting cuda host code for device `dev`, on a separate CPU thread
		if (this->device_type < DeviceType::Type_CUDA || this->device_type > DeviceType::Type_VK) {
			LOG_F(ERROR, "Solver# %u: bad device type %d!" BUGIFHAPPENED, solver_no, static_cast<int>(device_type));
			return;
		}

		//name thread after its solver#, the device type, and relevant API's device# (or CPU thread#).
		//StringBuilder sb_threadname; [todo]
		const std::string thread_name = DeviceType_Str[static_cast<ushort>(this->device_type)] +
			std::to_string(this->device_no) + " (Solver #" + std::to_string(this->solver_no) + ")";
		loguru::set_thread_name(thread_name.c_str());

		LOG_IF_F(INFO, HIGHVERBOSITY, "Host thread starting");
		if (device_type == DeviceType::Type_CUDA && device_no >= CUDA_MAX_DEVICES) { // [TODO]: support other device types. <--- move this?
			LOG_F(ERROR, "Bad CUDA device # %u!" BUGIFHAPPENED, this->device_no);
			return; //do not start
		} // ^ and any other checks like this... consolidate. <--- [WIP] / [TODO].

		g_params_changing = false;  // [MOVEME] ?
		gNetPause = false;			// [MOVEME] ?

		// SOLVER OR CUDA-DEVICE? [FIXME]
		//genericSolver must already be initialized.	Was: CUDA_Solve(...);
		if(!p_solver->InitGenericDevice()) {	// [TODO]: generic Init function <---
			LOG_F(ERROR, "Could not initialize device for Solver# %u (%s)!", this->solver_no, this->p_solver->gpuName.c_str());
			// any cleanup? <--- [TODO] / [WIP] !
			return;
		}

		// FIXME: There is some functional overlap between Start() and InitGenericDevice(),
		//			also consider changing where Allocate is called. "InitCudaDevice()" better name? <----
		Solvers[solver_no]->Start(/*init_params*/) ? LOG_F(INFO, "Host thread for device# %u complete.	(OK)", device_no) : 
			LOG_F(ERROR, "StartMining() for device# %u complete.	(Error)", device_no);		// device type? [TODO]
	} //.SolverThread_entry()

}; //.class SolverThread


// === ===
//int SpawnSolverThread (const DeviceType device_type, const int device_num);

// see declaration in generic_solver.hpp.	Note: native class type.	[TODO]: platform-independent threading.
int genericSolver::SpawnThread (void)
{
	const std::string str_threadname = "Solver" + std::to_string(this->solver_no);	// [todo]: name solver threads w/ the type?
	LOG_IF_F(INFO, HIGHVERBOSITY, "Starting genericSolver (thread %s)...", str_threadname.c_str());	// <--- [FIXME]: should be device type-agnostic!

//	SolverThread^ o1 = gcnew SolverThread( solver_no, device_type, device_num /*, PARAMS*/ );
	SolverThread^ o1 = gcnew SolverThread(this /*, PARAMS*/);
	Thread^ t1 = gcnew Thread(gcnew ThreadStart(o1, &SolverThread::SolverThread_entry));

	this->solver_status = SolverStatus::Starting;	//<-- necessary?
	t1->Name = gcnew String(str_threadname.c_str());
	t1->Start();

	return 0;  // [TODO]: verify thread launched. return 1 otherwise.
}
// === ===

// don't launch kernels, and yield CPU time if paused for a network reason (unavailable, waiting on a solution submission)
// (send 1 sol'n per challenge in Solo Mode.)
void Pause_ms (const unsigned long ms)
{ //Windows-specific.	[todo]: platform-independent, native pause function.
	LOG_IF_F(INFO, DEBUGMODE, "Pausing for %lu ms", ms);	// DWORD: unsigned long
	Sleep(ms);
}

void Console_Splash () {
	printf("Welcome to COSMiC V4!	(DEV TEST)\n");
	printf("Build Date: " __DATE__ " " __TIME__ "\n");
}

void SpawnConsoleOutput(void)
{ // spawns a simple console (output) window to view stdout messages. complements the log.
	domesg_verb("Spawning console window. ", true, V_NORM);  // UI event
	LOG_F(INFO, "Spawning console window.");				// [todo]: domesg callback when logging.
	try {
		AllocConsole();
		HANDLE stdHandle;
		int hConsole;
		FILE* fp;
		stdHandle = GetStdHandle(STD_OUTPUT_HANDLE);
		hConsole = _open_osfhandle((uint64_t)stdHandle, _O_TEXT);	// uint64_t should be more than big enough
		fp = _fdopen(hConsole, "w");
		freopen_s(&fp, "CONOUT$", "w", stdout);
	} catch (...) {
		LOG_F(ERROR, "Exception caught trying to get stdout console. ");  // never happens. but improve this  [todo]
		return;
	}
	LOG_F(INFO, "Spawned console window.");
	Console_Splash();	// say hi!
}

// [moveme] to logging.hpp/cpp ?
void log_cb_close(void* user_data) {}  // dummy loguru "close" callback
void log_cb_flush(void* user_data) {}  // dummy loguru "flush" callback
void log_callback_print(void* user_data, const loguru::Message& mesg)
{ // conditionally print messages to stdout when logging
	if (gVerbosity >= V_NORM) { printf("%s%s \n", mesg.prefix, mesg.message); }
	// [ref]:  reinterpret_cast<CallbackTester*>(user_data)->num_print += 1;
}

struct CallbackTester {
	bool dummy{ false };
};

void Start_Logging()
{ //
	CallbackTester logcb_tester;  // <- (remove)
	loguru::add_file	( "log.txt", loguru::FileMode::Append, gVerbosity );  // [TODO] setting the verbosity from command line
	loguru::add_callback( "log_callback", log_callback_print, &logcb_tester, loguru::Verbosity_INFO, log_cb_close, log_cb_flush );
	loguru::set_thread_name( "Main Thread" );							  // will display in log alongside relevant messages
}

// MAIN (entry point for the COSMiC miner.)
[STAThreadAttribute]
int Main (void)
{
#ifdef TEST_BEHAVIOR
	SpawnConsoleOutput();
#endif
	Start_Logging ();

//  [TODO]  Move any LibCurl init here? or after Form init?
	if (!Init_Sodium()) { // prepare sodium cryptography lib
		MessageBox::Show("LibSodium could not be initialized. Please re-extract COSMiC.", "COSMiC", MessageBoxButtons::OK, MessageBoxIcon::Stop);
		return 0; }

	LOG_F(INFO, "Spawning CosmicWind form. ");
	Application::EnableVisualStyles();
	Application::SetCompatibleTextRenderingDefault(false);
	
	Cosmic::CosmicWind form;
	Application::Run( %form );

	// application run finished:
	LOG_F(INFO, "Thanks for using COSMiC!");
	return 0;  // no error
}

//} //namespace
