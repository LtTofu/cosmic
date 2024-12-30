#pragma once
// net_pool.cpp : COSMiC V4 Pool Interface
//

#define WIN32_LEAN_AND_MEAN

#include <Windows.h>
//#include "curl/curl.h"				// for network functionality  note: only used in one place in this file
#include <libsodium/sodium.h>			// cryptography, random numbers
#include <cinttypes>	 				// <inttypes.h>  ...for unsigned integer types (e.g. uint8 to 64_t)
#include <iomanip>
#include <queue>
//#include <mutex>
#include <loguru/loguru.hpp>
#include <json11/json11.hpp>
//
#include "defs.hpp"
#include "coredefs.hpp"		// new <--- [TESTME] check for any conflicts!
#include "network.hpp"			// note: compile "solutions" object without /clr.
#include "cpu_solver.h" 
#include "util.hpp"
#include "net_pool.h"			// <- [TODO] check for conflict! <-
//
using namespace json11;
//

//#define DEF_COUNT_REJECTED_AS_INVALID

//#include "net_solo.h"  (or)...
//void Solo_CheckStalesInTxView(void);					// net_solo.cpp. TODO: move pool/solo mode common functionality into network.cpp/.h.

// Moved from net_pool.cpp to CosmicWind.cpp (which is compiled with /clr). Find it a more appropriate home after migrating the native code to a separate library.

//
extern std::string str_net_lastcurlresult;				// "OK" or the error string from libcurl
extern bool gNotifiedOfSolSubmitFail;					// if user was notified of a failed share submit (net error)

// statistics
extern bool stat_net_last_success;						// true if previous network operation succeeded.
extern unsigned long long stat_net_badjson_responses;	// parsing
extern unsigned long long stat_net_network_errors;

extern struct txViewItem gTxViewItems[DEF_TXVIEW_MAX_ITEMS]; // see CosmicWind.h
// [WIP]: consolidating externals/declarations w/ "Core" project.
//

//const std::string	gStr_DonateEthAddress{ "0xa8b8ea4c083890833f24817b4657888431486444" };	// constexpr
std::string gStr_MinerEthAddress{""};	//[moveme]
//

inline bool UpdateSolutionCounts(const unsigned short device_num, const DeviceType device_type, const bool is_devshare)
{
	if (device_type == DeviceType::Type_CUDA)
		gNum_SolutionCount[device_num] += 1;  // increment GPU's valid solution count
	else if (device_type == DeviceType::Type_CPU)
		cpuThreadSolutionsFound[device_num] += 1;  // increment CPU thread's solution count
	else { //shouldn't happen.
		domesg_verb("Unexpected Solving Device type: " + std::to_string(static_cast<unsigned int>(device_type)), true, V_NORM);
		return false;
	}

	if (!is_devshare) {
		domesg_verb("Share submitted.", true, V_MORE);
		LOG_IF_F( INFO, gVerbosity>=V_MORE, "Share submitted." );
	} else {
		domesg_verb("Dev share submitted.", true, V_MORE);
		gNum_DevSharesSent += 1;  // per-device?
		LOG_IF_F( INFO, gVerbosity>=V_MORE, "Dev share submitted (count: %" PRIu64 ").  Devshare cycle: %d / 200", gNum_DevSharesSent, devshare_counter );
	}
	return true;	//OK
}

void SolutionSentSuccessfully(const unsigned short dev_num, const DeviceType dev_type, const bool devshare)
{
	UpdateSolutionCounts(dev_num, dev_type, devshare);  // <- [TESTME]

	stat_net_last_success = true;	// redundant?
	gNotifiedOfSolSubmitFail = false;	// reset since share was sent successfully

	// example: a "cycle" of 200 shares, enables simple math for auto-donation adjustment in 0.5% increments
	std::string s_devshare_logmesg = "devshare_counter was " + std::to_string(devshare_counter);  // <-
	devshare_counter >= 199 ? devshare_counter = 0 : ++devshare_counter;					 
	s_devshare_logmesg += ", is now " + std::to_string(devshare_counter);					// <- remove dbg stuff
	LOG_IF_F(INFO, DEBUGMODE, s_devshare_logmesg.c_str());									// <-
}

// [TODO] make sure this function is thread-safe. should be now!
// #define DEF_COUNT_REJECTED_AS_INVALID
//
void Pool_HandleVerifiedShare ( /*const?*/ QueuedSolution *p_soln, miningParameters *params )
{  // Note: This function is only called if Pool Mining. Solo solutions are picked out & handled by the UI code. (see CosmicWind.h)
	if (gSoloMiningMode) {
		domesg_verb("DEBUG: Pool_HandleVerifiedShare() should only run in Pool Mode. ", true, V_DEBUG);
		return;
	}  // no return val, WIP/FIXME

	LOG_IF_F(INFO, DEBUGMODE, "Handling already-verified share. (pool mode)");		//dbg
	const SubmitShareResult sendResult = Pool_SubmitShare( p_soln, params );			//0=OK, >0=error types. [todo]: use enum

	if (sendResult == SubmitShareResult::SHARE_ACCEPTED) {
		PopSolution();  // 
		SolutionSentSuccessfully(p_soln->deviceOrigin, p_soln->deviceType, p_soln->devshare);
	} else if (sendResult == SubmitShareResult::SUBMIT_ERROR) {
	  	std::string errString = "Network error while submitting ";
		p_soln->devshare ? errString += "devshare" : errString += "share";
	  	if(DEBUGMODE)
	  		errString += p_soln->solution_string + " for challenge " + p_soln->challenge_string.substr(0, 8) += ". ";

		if (!gNotifiedOfSolSubmitFail) {
			domesg_verb(errString, true, V_NORM);  // as event (also prints to stdout).  V_MORE? <-
			gNotifiedOfSolSubmitFail = true;
		}
	} // the solution is left in the queue for re-submit attempt(s)
	else if (sendResult == SubmitShareResult::SHARE_REJECTED) {
#ifdef DEF_COUNT_REJECTED_AS_INVALID
		gNum_InvalidSolutionCount[(p_soln->deviceOrigin)] += 1;  // <- [WIP]: Count these separately from sol'ns that fail CPU verification (not sent).
#endif
	  	// [todo]: if solved by a cpu, iterate the # of invalid solutions for that thread <--
	  	// ...
	  	domesg_verb("Share " + p_soln->solution_string.substr(0,32) + "... rejected by the pool. (stale?) ", true, V_NORM);
	  	PopSolution();  // pop frontmost sol'n from q_solutions
	} else
		printf("Unexpected return code %d from Pool_SubmitShare(). \n", sendResult);

	VLOG_F(V_MORE, "- Share Type Counter: %d ", devshare_counter);
	//return sendResult;
}

// CheckDevshareAndSend: Takes a pre-verified solution (in queue slot `slotNum`), determines if
// Regular or DevShare and calls Pool_SubmitShare w/ the solution slot#. <- old behavior
bool CheckDevShare(void)
{ // find if this is a regular or devshare	(ex. Setting of 1.5% produces 3 dev-shares out of every 200)
	bool is_devshare = false;
	const unsigned int scratch = 7 + unsigned int(gAutoDonationPercent * 2.0);  // can be set in increments of half a %
	if ((devshare_counter > 6) && (devshare_counter < scratch))
		is_devshare = true;
//	q_solutions.front().devshare = is_devshare;
	return is_devshare;
}


// Retrieves the mining parameters from pool. If isInitial==true, run everything.
// If not, only the stuff that changes frequently (see: not pool mining address).
// [note to self]:  see Pool_Get...() functions to retrieve mining params in `network.cpp`.
bool Pool_GetMiningParameters(const bool isInitial, const bool computeTarget, miningParameters *params)
{ // in Pool Mining Mode, the Mining Target is computed from the MaxTarget for now.
  // (WIP): This is to support ERC-918s that use a maxTarget different from 0xBitcoin's 2^234.
  //		Add a simple option to select "override maxTarget and Compute" or "use Pool Provided Target"
	bool err_occurred{ false };
	GetParmRslt rslt{ GetParmRslt::OK_PARAM_UNCHANGED };  // 0=OK, parameter unchanged.  1=error from function.  2=OK, param changed.
	// Move "Got new challenge/minting address/difficulty messages here?	[todo]

	// Get the latest challenge (no matter what):
	rslt = Pool_GetChallenge(isInitial, params);
	if (rslt == GetParmRslt::OK_PARAM_UNCHANGED) {}	//<-- nothing
	else if (rslt == GetParmRslt::GETPARAM_ERROR)
		err_occurred = true;	//if err, still try to retrieve the other params
	else if (rslt == GetParmRslt::OK_PARAM_CHANGED) {
		params->challenge_changed = true;
		params->params_changing = true;
	} else {
		LOG_F(ERROR, "Bad result from Pool_GetChallenge():	%d!" BUGIFHAPPENED, static_cast<int>(rslt));
		return false;
	} //abort

	// At mining start, get pool's Minting Address to mine with. Also occasionally check if it has changed (unlikely, pool reconfigured)
	if (isInitial || (!gSoloMiningMode && checkDoEveryCalls(Timings::getPoolMintAddr)))
	{	// [pool mode]: check occasionally. [solo mode]: only once at mining start.
		rslt = Pool_GetMintingAddress(isInitial, params);
		if (rslt == GetParmRslt::OK_PARAM_UNCHANGED) {}		//<-- nothing
		else if (rslt == GetParmRslt::GETPARAM_ERROR)
			err_occurred = true;	// "
		else if (rslt == GetParmRslt::OK_PARAM_CHANGED) {
			params->mintaddr_changed = true;
			params->params_changing = true;
		} else {
			LOG_F(ERROR, "Bad result from Pool_GetMintingAddress():	%d!" BUGIFHAPPENED, static_cast<int>(rslt));
			return false;	//abort
		}
	}

	// [WIP] depending on the setting "compute target locally", only the difficulty# -or- the uint256 difficulty target should be retrieved
	// request difficulty#/target from the pool on initial run (mining start) or every n runs
	if (isInitial || checkDoEveryCalls(Timings::getDifficulty))
	{	//compute target from difficulty # and specified (or default) maxTarget divisor.
		if (computeTarget) {
			rslt = Pool_GetDifficulty(isInitial, params);  // updates gU64_DifficultyNo
			if (rslt == GetParmRslt::OK_PARAM_UNCHANGED) {}		//<-- nothing
			else if (rslt == GetParmRslt::OK_PARAM_CHANGED) {
				params->difficulty_changed;
				params->params_changing = true;
			}
			else if (rslt == GetParmRslt::GETPARAM_ERROR)
				err_occurred = true;
			else {
				LOG_F(ERROR, "Bad result from Pool_GetDifficulty():	%d!" BUGIFHAPPENED, static_cast<int>(rslt));
				return false;	//abort
			}
		} else {
		// request difficulty# and difficulty target string from the pool on initial run (mining start) or when checkDoEveryCalls() satisfied
			rslt = Pool_GetTarget(isInitial, params);  // updates gStr_MiningTarget
			if (rslt == GetParmRslt::OK_PARAM_UNCHANGED) {}		//<-- nothing
			else if (rslt == GetParmRslt::OK_PARAM_CHANGED) {
				params->target_changed = true;	//difftarget_changed = true;
				params->params_changing = true;
			}
			else if (rslt == GetParmRslt::GETPARAM_ERROR)
				err_occurred = true;
			else {
				LOG_F(ERROR, "Bad result from Pool_GetTarget():	%d!" BUGIFHAPPENED, static_cast<int>(rslt));
				return false;	//abort
			}
		}
	} //condense this [TODO].

	params->params_changing = false;	//<---

	if (err_occurred) {
		domesg_verb("Warning: error(s) while retrieving latest mining params ", true, V_MORE);		//mining event message (remove?) <---
		if (params->params_changing && DEBUGMODE) printf("Param(s) changing, Solvers should pause"); //remove <---
		//++stat_net_consecutive_errors;
	}
	if (isInitial && gVerbosity >= V_MORE) { /* debug info: */ 
		LOG_F(INFO, "Got mining parameters from pool at %s: \n" "Challenge: %s \nPool-provided Mining Target: %s \n" "Difficulty: %" PRIu64 " \nMinting Address: %s \n", 
		gStr_PoolHTTPAddress.c_str(), params->challenge_str.c_str(), params->target_str.c_str(), params->difficultyNum, params->mintingaddress_str.c_str());
	}

	return (!err_occurred);		//true=OK
} // [TESTME]. <---

