// network.cpp / .hpp : Common network functions for both Pool and Solo modes. See defs in solutions.hpp
//						also a new Solutions Queue (with mutex access control for asynchronous access)
// 2020 LtTofu

//#include "network.hpp"
#include <mutex>
#include <forward_list>			// <- Debug use only. Remove
//#include <iostream>

#include <loguru/loguru.hpp>
#include <libsodium/sodium.h>
#include <curl/curl.h>
#include <bitcoin-cryptography-library/cpp/Keccak256.hpp>
#include <json11/json11.hpp>

#include "defs.hpp"
#include "coredefs.hpp"											//<---
extern struct txViewItem gTxViewItems[DEF_TXVIEW_MAX_ITEMS];	//<---
#include "net_solo.h"
//
#include "net_pool.h"	// [CHECKME]: this will be needed
#include "cpu_solver.h"
#include "util.hpp"
//
#include "network.hpp"
#include "generic_solver.hpp"
//#include "cuda_device.hpp"

// === debug stuff ===
#define DEBUG_TEST_FOR_DUPLICATE_SOLNS	1			// expensive, normally disabled
//#define USE_PROXYMINT_TEST	0					// [WIP]: for proxy mint helper contract. goal: cheaper mint/donation in solo mode
//#define DEBUG_NETWORK		0						// prints network responses, some performance profiling stuff. rarely desirable.
													// params retrieval, compare etc.

const std::string gStr_DonateEthAddress{"0xa8b8ea4c083890833f24817b4657888431486444"};	//using constexpr breaks something [fixme].
uint64_t gU64_DifficultyNo{0};						// [old] global 64-bit difficulty (for CPU solvers). phasing out global params

bool g_params_changing{ false };					// [old] <---
miningParameters gMiningParameters {};				// 
// [todo / WIP]: flush the solutions that the change would invalidate, if any. pause launching kernels while core params update.
//				  if `true`, `mtx_coreparams` probably being locked/unlocked by network thread.


// idea
std::forward_list<QueuedSolution> fl_pastsolutions;	// for testing only. temporarily store a copy of already-sent solutions <--
													// to check for duplicates. troubleshooting pool interaction w/ mvis.ca
// For testing use only!
bool check_vs_prior_solutions (const std::string test_nonce)
{ // This is for troubleshooting.
	if (fl_pastsolutions.empty())  { return true; }  // if no solutions recorded yet
	
	uint32_t howmany{}, dups{};		// test_nonce compared w/ `howmany` prior nonces, looking for duplicates (sanity check) <--
	for (auto it = fl_pastsolutions.begin(); it != fl_pastsolutions.end(); ++it)  /* while()? */
	{
		if (test_nonce != it->solution_string) {
			if (DEBUGMODE) { printf("CheckVsPreviousSolutions():  != %s.  (no match) \n", it->solution_string.c_str()); } // <--- spammy
			++howmany;
			continue;		// check next sol'n. [todo]: make sure fl_pastsolutions is cleared when the challenge changes. <--
		}

		const std::string s_logmesg{ "Tested sol'n is a duplicate to previously-mined nonce " + it->solution_string + ", solved for challenge " +
			 it->challenge_string + ", difficulty #: " + std::to_string(it->poolDifficulty) + "!" };	//<-- TODO: common difficulty var member (not mode-specific).
		//if (DEBUGMODE)	{ std::cout << "CheckVsPreviousSolutions():	" << test_nonce << " matches " << it->solution_string << "!![fixme]" << std::endl; } //<-- dbg: shouldn't happen!
		LOG_F(WARNING, s_logmesg.c_str());
		++dups; //redundant
		++howmany;	// "
		return false;  // stop early, match found <-- disabled to count how many dups, if any
	} //.for

	if (DEBUGMODE) {
		LOG_F(INFO, "Checked %s against %" PRIu32 " previous solutions w/ %" PRIu32 " duplicates:		", test_nonce.c_str(), howmany, dups);
		if (!dups) {
			printf("[OK!]\n");
			return true;
		} else {
			printf("[ERR]\n");
			return false;
		}
	}
	return true;  // OK- no matches to prev sol'ns
}

bool check_duplicate_share ( const uint8_t solution_bytes[32] )
{
	if (!check_vs_prior_solutions("0x" + uint8t_array_toHexString(solution_bytes, 32))) {
		LOG_IF_F(INFO, NORMALVERBOSITY, "Duplicate solution found: not adding to queue.");
		domesg_verb("Duplicate solution found: not adding to queue. ", true, NORMALVERBOSITY);
		return false;		// do not submit a duplicate. Investigate the cause
							// (IF this happens. The statistical likelihood should be very low).
							//update: so far so good. suspect dup. shares caused by packet loss
	}
	else { return true; }
}
// ^ for debugging- remove.


void PopSolution(void)
{
	//LOG_IF_F(INFO, gVerbosity==V_DEBUG, "DBG: PopSolution(): Acquiring mtx_solutions");
	std::lock_guard<std::mutex> lockg(mtx_solutions);  // should release the mutex when `lockg` goes out of scope
	if (q_solutions.empty()) {
		//LOG_F(INFO, "PopSolution(): the queue is empty ");
		return; }

	//LOG_IF_F(INFO, V_DEBUG, "Popping sol'n %s	from queue", q_solutions.front().solution_string.substr(0,32).c_str());
	QueuedSolution tempsoln = q_solutions.front();		// testing use only <--
	q_solutions.pop();
	fl_pastsolutions.push_front(tempsoln);				// <--
}


void ClearSolutionsQueue(void)
{
	size_t howmany{ 0 };	//remove
	std::lock_guard<std::mutex> lock(mtx_solutions);
	while (!q_solutions.empty()) {
		LOG_IF_F(INFO, HIGHVERBOSITY, "Removing sol'n: %s from queue.", q_solutions.front().solution_string.c_str());
		q_solutions.pop();
		++howmany;	//remove
	}
	LOG_IF_F(INFO, DEBUGMODE, "Removed %s solutions from queue", std::to_string(howmany).c_str());
}

bool GetSolutionFromQueue(QueuedSolution* out_soln)
{
//	if (!gSoloMiningMode) {
//		domesg_verb("GetSolutionFromQueue(): call in Solo mode only! ", true, V_DEBUG);  /* return false; */ } // <--- FIX??? checking if appropriate <---
//	if (gVerbosity == V_DEBUG)  printf("GetSolutionFromQueue(): acquiring mutex... \n");  // debug
	std::lock_guard<std::mutex> lockg(mtx_solutions); // get exclusive access to the queue, or block til then <-

	if (!q_solutions.size()/* || q_solutions.empty()*/)
		return false;
//
// test:
	if (q_solutions.front().solution_no != -1) {
		//LOG_IF_F(WARNING, gVerbosity>V_NORM, "Solution # %" PRIu32 " already processed! This is a bug", q_solutions.front().solution_no);	//<-- what's the right format specifier?
		printf("Warning: Solution # %" PRIu32 " already processed! " BUGIFHAPPENED, q_solutions.front().solution_no); //<- should never happen
		LOG_IF_F(WARNING, gVerbosity >= V_NORM, "Warning: Solution # %" PRIu32 " already processed! This is a bug ", q_solutions.front().solution_no); //<- should never happen
		//
		q_solutions.pop();
		return false;
	}
// .test
//
	if (q_solutions.front().solution_string.length() != HEXSTR_LEN_UINT256_WITH_0X) {
		LOG_F(ERROR, "GetSolutionFromQueue(): Bad solution nonce length %zu characters, expected 66", q_solutions.front().solution_string.length() );
		return false; }  // solutions queue empty, or error
// looks OK:
	*out_soln = q_solutions.front();  // deref. pointer & copy queue's frontmost solution.  memcpy req'd instead? <-- [TESTME]
//	domesg_verb("GetSolutionFromQueue(): Got solution nonce " + q_solutions.front().solution_string, false, V_DEBUG);

	if (gSoloMiningMode)
		q_solutions.pop();
	// in Pool Mode, the sol'n is not popped yet in case this func has to run again for re-try (such as network error)
	// in Solo Mode, the txView (Solutions Viewer) handles re-try and the sol'n is popped from queue _now_.
	// in either Mode, if a solution fails CPU verification, it will be removed from the queue, and not sent.
	return true;  // a solution was written to out_soln
}


#define DEBUG_VERIFYSOLN

// [wip / idea]: uint256 target as member of QueuedSolution (currently just mining difficulty#).
bool VerifySolution (QueuedSolution *p_soln, miningParameters *params)
{
#ifdef DEBUG_VERIFYSOLN
	QueuedSolution debug_queuedsolution = *p_soln;		// <-
	miningParameters debug_miningparams = *params;			// <-
#endif

	LOG_IF_F(INFO, DEBUGMODE, "Verifying Solution %s from device id %d, device type %d...", p_soln->solution_string.c_str(), p_soln->deviceOrigin, p_soln->deviceType);
	print_bytes(p_soln->solution_bytes, 32, "solution");

	if (!checkString(p_soln->solution_string, 66, true, true)) {// [WIP / TODO]: use the bytes instead.
		LOG_IF_F(WARNING, DEBUGMODE, "malformed solution string in queued sol'n!");
		return false;
	}// ^ just in case. remove debug stuff [todo]

	uint8_t message[84]{};
	memcpy(message, &p_soln->solver->hash_prefix, 52);
	memcpy(&message[52], p_soln->solution_bytes, 32);
	print_bytes(message, 84, "message");

	uint8_t hashOut[Keccak256::HASH_LEN]{ 0 };					// 32-byte array for 256-bit digest
	const bool hash_success = Keccak256::getHash(message, 84, hashOut);	// hash with another keccak256 implementation to verify sol'n from GPU or CPU solver
	if (!hash_success) {
		LOG_F(ERROR, "in VerifySolution(): reference keccak256 hash failed!");
		return false;
	}

	uint8_t target_bytes[32]{ 0 };
	memcpy(target_bytes, (uint8_t*)p_soln->solver->target, 32);
	const Uint256 U256_target = Uint256(target_bytes);
	const Uint256 U256_digest = Uint256(hashOut);			// construct Uint256 digest from bytes instead :)
	if (U256_digest == Uint256::ZERO) {
		LOG_F( ERROR, "in VerifySolution(): bad digest!" );
		return false;	//failed
	}
	
//#ifdef TEST_BEHAVIOR
//	uint8_t dbg_bytes[32] {};
//	U256_digest.getBigEndianBytes(dbg_bytes);
//	print_bytes(dbg_bytes, 32, "keccak256 digest");
//	U256_target.getBigEndianBytes(dbg_bytes);
//	print_bytes(dbg_bytes, 32, "mining target");
//#endif	//^ remove!

	// compare keccak256 hash to 256-bit unsigned mining target.  store digest as std::string in queued sol'n object
	p_soln->digest_string = "0x" + HexBytesToStdString(reinterpret_cast<const unsigned char*>(hashOut), Keccak256::HASH_LEN);  // <-- [WIP/TODO]: pass bytes directly.

	if (U256_digest <= U256_target && U256_digest != Uint256::ZERO) { /* true: solution passes difficulty target check, false otherwise. */
		domesg_verb("Solution validated by CPU  (Digest: " + p_soln->digest_string + "). ", true, V_NORM); //<-- FIXME
		return true; }	// passed
	else {
		domesg_verb("Solution failed CPU verification  (Digest: " + p_soln->digest_string + "). ", true, V_LESS);  // <-- "
		return false; }	// failed
}

//
bool DoVerify( QueuedSolution *p_soln, miningParameters *params )
{
	if (p_soln->verified) {
		LOG_IF_F(INFO, DEBUGMODE, "Solution #%d already verified.", p_soln->solution_no);	//<-- remove
		return true; //<-- new
	}

	unsigned short devID{ UINT8_MAX };  // inited to appropriate null. 0 is a typical CUDA device #.
	if (p_soln->deviceOrigin >= 0 && p_soln->deviceOrigin < CUDA_MAX_DEVICES)
		devID = p_soln->deviceOrigin;
	else {
		domesg_verb("Solution w/ bad device # " + std::to_string(p_soln->deviceOrigin) + " in buffer. Removing.", true, V_DEBUG);
		PopSolution();	 // [TESTME]
		return false;	//not verified
	}

	return VerifySolution(p_soln, params);  // true: passed verification,  false: otherwise
}


bool GetTxViewItem( txViewItem *output_txn, unsigned int *txview_solution_no )
{
//	unsigned int txview_slotno{ 0 };	// 0= null item
	std::lock_guard<std::mutex> lockg(mtx_totxview);
	if (q_totxview.empty())
		return false;			// no solution-txn #s in queue

//	unsigned int txview_slotno = 0;
	unsigned int txview_slotno = q_totxview.front();	// FIFO queue
	if (txview_slotno < DEF_TXVIEW_MAX_ITEMS) { /* valid slot #s range: 0 - DEF_TXVIEW_MAX_ITEMS */
		domesg_verb("GetTxViewItem(): got Solo-mode Solution #" + std::to_string(txview_slotno), false, V_DEBUG);  // <-
		*output_txn = gTxViewItems[txview_slotno];  // write to txViewItem var in calling function
		
		*txview_solution_no = txview_slotno;
		q_totxview.pop();

LOG_IF_F( INFO, DEBUGMODE, "q_totxview contents: %zu items", q_totxview.size() );
		return true;  // txview item written to output
	}
	
	LOG_F(ERROR, "GetTxViewItem(): Bad txview item#: %d. Discarding frontmost of q_solutions", txview_slotno);
	q_totxview.pop();  // remove from queue
	return false;
}

// moved from net_pool.cpp
unsigned int GetFreeTxViewSlot(void)
{
	//std::lock_guard<std::mutex> lockg(mtx_txview); <- see calling func
	for (unsigned int slot = 0; slot < DEF_TXVIEW_MAX_ITEMS; ++slot)
	{
		if (!gTxViewItems[slot].slot_occupied)
			return slot;  // return empty slot's #
	}
	return 0;  // not a valid slot#
}

// [todo / wip]: finish improved TXView.
bool PopulateTxItem( const QueuedSolution *p_soln )
{
	// if this SOLO solution has been verified, add it to the array of Transactions-to-be.
	// TODO: speed this up by adding a QueuedSolution member to txViewItem (gTxViewItems[]) to
	// copy it more directly from, (or point to), the frontmost element in q_solutions? <--
	//
LOG_IF_F(INFO, DEBUGMODE, "PopulateTxItem(): acquiring mtx_txview");
	std::unique_lock<std::mutex> ulock(mtx_txview);  // [WIP] here & anywhere else gTxViewItems[] is accessed
	
	const unsigned int slotNo = GetFreeTxViewSlot();
	if (slotNo >= DEF_TXVIEW_MAX_ITEMS)
		return false;  // err <-- [WIP]

LOG_IF_F(INFO, DEBUGMODE, "Populating TxView item#: %d", slotNo);	// -1: error <----------- Remove This
LOG_IF_F(INFO, DEBUGMODE, "And pushing verified solution# %d to TxView:	%s", slotNo, p_soln->solution_string.c_str());
	gTxViewItems[slotNo].status = TXITEM_STATUS_SOLVED;
	gTxViewItems[slotNo].txHash.clear();		// = "";
	gTxViewItems[slotNo].errString.clear();		// = "";
	gTxViewItems[slotNo].str_solution = p_soln->solution_string;		// streamline this	[todo]
	gTxViewItems[slotNo].str_challenge = p_soln->challenge_string;
	gTxViewItems[slotNo].str_digest = p_soln->digest_string;
	gTxViewItems[slotNo].deviceOrigin = p_soln->deviceOrigin;
	gTxViewItems[slotNo].deviceType = p_soln->deviceType;
	gTxViewItems[slotNo].networkNonce = UINT64_MAX;			// null. will get txcount at submit time
	gTxViewItems[slotNo].solution_no = slotNo;				// <-
//
	gTxViewItems[slotNo].last_node_response = NODERESPONSE_OK_OR_NOTHING;
	gTxViewItems[slotNo].submitAttempts = 0;
	gTxViewItems[slotNo].str_signature_r.clear();	// = "";	// = soln.signature_r; ??
	gTxViewItems[slotNo].str_signature_s.clear();	// = "";	// = soln.signature_s; ??
	gTxViewItems[slotNo].str_signature_v.clear();	// = "";	// = soln.signature_v; ??
	gTxViewItems[slotNo].slot_occupied = true;
	ulock.unlock();  // done ^
// ...

	std::lock_guard<std::mutex> lockg(mtx_totxview);
	q_totxview.push(slotNo);
	return true;
}

// [todo] condense function
void Pool_ProcessSolution( QueuedSolution *p_tempsoln, miningParameters *params )
{
	if (p_tempsoln->verified)
	{ // pool share found & already verified (network err sending it last run?)
		domesg_verb("Comm_CheckForSolutions(): checking if devshare & sending to pool. \n", false, V_DEBUG);  // <- debug status, remove
		p_tempsoln->devshare = CheckDevShare();  // devshare? y/n
		/*const unsigned short result =*/ Pool_HandleVerifiedShare( p_tempsoln, params );  // don't re-verify. Pass the local solution by value. [WIP]
		return;  // to next solution in queue, if any <------- !!!! return val to calling func 
	}

	// not yet verified, do that now:
	p_tempsoln->verified = DoVerify( p_tempsoln, params );  // digest into temp_soln. [todo]: pass temp_soln by reference for speed.
	if (p_tempsoln->verified)
	{ /* did share pass verification? */
		if (p_tempsoln->deviceType == DeviceType::Type_CUDA)		{ gNum_SolutionCount[p_tempsoln->deviceOrigin] += 1; }					// <- [TESTME].
		 else if (p_tempsoln->deviceType == DeviceType::Type_CPU)	{ cpuThreadSolutionsFound[p_tempsoln->deviceOrigin] += 1; }				// <-
		 else { LOG_F( WARNING, "invalid deviceType %d in Pool_ProcessSolution()! Please report this bug", p_tempsoln->deviceType ); }		// <-

		p_tempsoln->devshare = CheckDevShare();
		/*const unsigned short result = */ Pool_HandleVerifiedShare( p_tempsoln, params );  // send. if network err: it will remain in the queue <-- PopSolution() ?
		// ^ pop if the above send succeeded? done in the func. if net error, it _should_ remain in the queue til sent, mining stop,
		//		 or challenge/other mining param change necessitating that q_solutions be emptied.
	} else {
		domesg_verb("Comm_CheckForSolutions(): sol'n " + p_tempsoln->solution_string + " failed CPU verification, removing from queue ", true, V_DEBUG);
		//gNum_InvalidSolutionCount[p_tempsoln->deviceOrigin] += 1;
		PopSolution();
	}
}

// [todo]  condense function
void Solo_ProcessSolution (QueuedSolution *p_tempsoln, miningParameters *params)	/* [WIP] */
{
	//	[TESTME]: don't add TXview items for >1 solution per challenge. [TODO]: if 'topmost' (first) Sol'n
	// for a challenge fails for some reason, need to fall back to the next? Clean up when chal. changes.
	if (gTxView_WaitForTx != NOT_WAITING_FOR_TX) {
		LOG_IF_F(INFO, DEBUGMODE, "Solo_ProcessSolution(): not handling, waiting for solution #%d (this challenge)", gTxView_WaitForTx);
		return;
	} else
		LOG_IF_F(INFO, DEBUGMODE, "Solo_ProcessSolution(): handling sol'n %s: ", p_tempsoln->solution_string.c_str());
	//	[TESTME]


	if (p_tempsoln->verified) { //speed this up by adding a QueuedSolution member to txViewItem (gTxViewItems[]) to get it from q_solutions.front()? <-- [todo / idea]
		PopulateTxItem(p_tempsoln);  // <-- [WIP]
		return;
	} //process next solution, if any.

//=== if solution needs to be verified first: ===
	LOG_IF_F(INFO, DEBUGMODE, "Solution %s needs verification. Doing that now. (solo mode) ", p_tempsoln->solution_string.c_str());  // <- prepend solution hexstr with `0x`! [todo/fixme]
	p_tempsoln->verified = DoVerify(p_tempsoln, params);	//sol'n digest written in temp_soln
	if (p_tempsoln->verified) /* did share pass verification? */
		PopulateTxItem(p_tempsoln);		// [WIP]: unused return value (bool)
	else {
		LOG_IF_F(WARNING, DEBUGMODE, "Discarding nonce: %s (failed local verification).", p_tempsoln->solution_string.c_str());
		PopSolution();
	} // [TESTME]
	// if net error, it _should_ remain in queue til sent, mining stop, or challenge/other mining param change necessitates that q_solutions be emptied.
	return;
}

//extern txViewItem gTxViewItems[DEF_TXVIEW_MAX_ITEMS];

// moved from managed code.  [TODO]: simplify/break up this function
unsigned short Comm_CheckForSolutions( miningParameters *params )
{
// ---
	QueuedSolution temp_soln{};		// inited w/ null values (see definition of `QueuedSolution`)
	bool clear_queue{ false };
// get sol'n(s) from front of FIFO queue `q_solutions` via native getter func, as <mutex> header is blocked when compiling with /clr
// pops solution before returning in solo mode only. in pool mode they must be popped after sending them (successfully) to the pool.
	
	while (GetSolutionFromQueue(&temp_soln) && !clear_queue)	/* loop til no sol'ns remaining. stop if any are stale */
	{ // (1 max submission per call when in Solo Mode? setting?)
		LOG_IF_F(INFO, DEBUGMODE, "Getting solution from queue, for challenge %s: ", temp_soln.challenge_string.c_str());

		// [TESTME]: solo mode stuff
		if (gTxView_WaitForTx != NOT_WAITING_FOR_TX) {
			LOG_IF_F(INFO, HIGHVERBOSITY, "Tx already pending for this challenge. Discarding nonce %s.", temp_soln.solution_string.c_str());
			continue;
		}	//[FIXME}: just don't call Comm_CheckForSolutions() if waiting for any Tx, (etc.?) <---

		// [TESTME]: if a queued solution is stale, stop processing the queue and clear it <---
		if (temp_soln.challenge_string != params->challenge_str) {	//check for stale.	[WIP] [MOVEME] ?
			LOG_IF_F(INFO, DEBUGMODE, "Sol'n nonce %s is stale. Solved for previous challenge %s. ", 
				temp_soln.challenge_string.c_str(), params->challenge_str.c_str());	// [todo]: ui message. log callback? <--
			clear_queue = true;		// THE SOL'NS QUEUE SHOULD BE CLEARED! <---
			//...
			break;	//was continue;
		}

 		if (!checkString(temp_soln.solution_string, HEXSTR_LEN_UINT256_WITH_0X, true, true)) { //fast string checks [WIP]
			LOG_F(WARNING, "Queued solution has bad format. Removing it." BUGIFHAPPENED);	// <-- debug only [TESTME]
			continue;	//try next sol'n in queue (if any)
		}
		
		gSoloMiningMode ? Solo_ProcessSolution(&temp_soln, params) : Pool_ProcessSolution(&temp_soln, params);	//Pool or Solo mode
		// either mode:  anything else? <-- [todo]
	}

	if (clear_queue)
		ClearSolutionsQueue();	// [TESTME] <---

	return 0;	// OK
}



#include "coredefs.hpp"	// [MOVEME] <----
//#include <mutex>

// [NOTE] / [TODO]:	function defined here because the <mutex> header is 
//		blocked when compiling with /clr (net_solo.cpp). Reworking this.
// [TODO]:	use better container/OOP approach for the TxView's items.
// [WIP]:	Protect TxViewItems[] from simultaneous asynchronous access by the main/network threads. [FIXME] <--
void Solo_CheckStalesInTxView(const miningParameters* params)
{ // (2020.01.22): if a major mining parameters has changed, check any SOLO transactions in Tx list
  //	that were solved for an old challenge, and mark them as stale so they're not submitted.
	unsigned int stales{ 0 };
	std::unique_lock<std::mutex> ulock_txview(mtx_txview);	// [TESTME] <--
	for (unsigned int t = 0; t < DEF_TXVIEW_MAX_ITEMS; ++t)
	{
		if (gTxViewItems[t].status == TXITEM_STATUS_EMPTYSLOT)
			continue;  // skip to next txView item

		if (gTxViewItems[t].str_challenge != params->challenge_str)
		{ //if txview solution item `t` was solved for a (now) outdated challenge, and is awaiting submission:
			if (gTxViewItems[t].status == TXITEM_STATUS_SOLVED || gTxViewItems[t].status == TXITEM_STATUS_SUBMITWAIT ||
				gTxViewItems[t].status == TXITEM_STATUS_SUBMITTING /* note to self: _SUBMITTING status used? */)
			{
				gTxViewItems[t].status = TXITEM_STATUS_STALE;   // <----- Only if solution/tx is in a state where setting it Stale makes sense! (waiting to send, etc.)
				LOG_IF_F(INFO, HIGHVERBOSITY, "Solution # %d was solved for outdated challenge %s. Marking as stale.", t, gTxViewItems[t].str_challenge.c_str());	//debug stuff
				++stales;
			}
		}
	}
	ulock_txview.unlock();	// [TESTME] <--

	LOG_IF_F(INFO, HIGHVERBOSITY && stales, "Marked %u sol'ns as stale. New challenge is: %s ", stales, params->challenge_str.c_str());	//debug stuff
}



// (WIP:)  Don't get Target from pool unless Advanced Option for it is enabled. It's computed locally <--
// (WIP:)  Make separate network activities' intervals customizeable in General Options or Advanced Options
// [WIP]: refactoring this. consider adding 'bool isInitial' param. <--
int recurring_network_tasks(const bool compute_targ, const unsigned short exponent, const miningParameters old_params, miningParameters *params)	/* [WIP] */
{ //RECURRING_NETWORK_TASKS(): Gets mining parameters from the pool at rate configured by the user

//lock/unlock around where params are accessed? other threads will likely read multiple params, not just one.
//	std::unique_lock<std::mutex> ulock(mtx_coreparams);		// but, don't leave mutex locked for excessively long time (waiting for the network, etc.)

	//Get mining parameters (Solo or Pool mode). If nonzero return code, return 1 (err). if successful, params written to `new_params`. <-- [wip]
	LOG_IF_F(INFO, DEBUGMODE && DEBUG_NETWORK, "recurring_network_tasks():  getting mining parameters");
	const bool rslt = gSoloMiningMode ? Solo_GetMiningParameters(false, compute_targ, params) : Pool_GetMiningParameters(false, compute_targ, params);
	if (!rslt) return 1;	//error

// [DEBUG]: Verify _changed bools are set TRUE when changing, & on _initial_ params retrieval (pool mode checked, check the solo-mode param retrieval functions.)
// [WIP]: redundant solo mode check? Solo_GetMiningParameters() doesn't retrieve Minting Address, it's derived from the loaded Eth. account
// - has the challenge changed? checked in Pool or Solo-mode GetChallenge func.
// - has the mining target changed? or the difficulty # ? (if locally computing target).
// removed redundant check of uint256 target, target _string_ is checked already when it's retrieved. [WIP]
// [TESTME]: ensure proper behavior on initial parameters retrieval (mining starting).
//	ulock.unlock();  // [idea] <-- How often does any other thread really need to access coreparams? [WIP]

// === Got mining params from pool/node. Now check new params against current: ===
	LOG_IF_F(INFO, DEBUGMODE && DEBUG_NETWORK, "recurring_network_tasks():	checking for changed params");
	if (params->challenge_changed || params->target_changed || params->difficulty_changed || params->mintaddr_changed)
	{ //update the GPUs	[WIP] <---

		ClearSolutionsQueue();	//moved
	// [todo]: only discard sol'ns that become stale after the parameter change(s). Difficulty could change frequently (such as on a VARDIFF pool).
	//			some solutions may be valid for the new difficulty target. Challenge/Minting Address change invalidates any solutions in the queue.
	//	ClearStaleSolutions (params);

		LOG_IF_F(INFO, DEBUGMODE && DEBUG_NETWORK, "recurring_network_tasks():	updating devices/params");
		const unsigned short update_result = UpdateMiningParameters_new(params, compute_targ, exponent); //stop mining if hardware/software failure occurred.	[wip] <--
		if(update_result != 0) {
			LOG_F(WARNING, "Err %d while updating mining parameters!", update_result);
			domesg_verb("Error(s) encountered while updating mining parameters. Code " + std::to_string(update_result), true, V_NORM);	//ui message
			return update_result;
		}

		//std::unique_lock<std::mutex> ulock(mtx_globalparams);	//<--- Move?
	//	g_params_changing = true;	//pause launching kernels while parameters are changing. [MOVEME] ?
		g_params_changing = false;	//new message/target should now be populated (if needed), Solvers should read params. [WIP]
		//ulock.unlock();

		const ushort solver_count = static_cast<ushort>( Solvers.size() );
		if (solver_count > 0 && solver_count < MAX_SOLVERS) { //[FIXME] ? <---
			for (ushort s = 0; s < solver_count; ++s) {
				LOG_IF_F(INFO, DEBUGMODE, "Network Thread has new Params for Solver# %u", s);
				if (Solvers[s]->enabled)
					Solvers[s]->new_params_available = true;	//let solvers know: they'll retrieve their own params.
			}
		}
		// ^ [MOVEME] ?

		if (gSoloMiningMode)
			Solo_CheckStalesInTxView( params );	//check for stales in the TXview (solo mode only)
		if (gCpuSolverThreads > 0)
			GenerateCpuWork( params );			//update any CPU solvers (if running)			// <-- [TESTME]
	//	if (clearSolnQueue == true)	ClearSolutionsQueue();				//if above conditions met
	// [IDEA]: if any sol'ns in queue, check if they still meet new target. if so, update them w/ the new difficulty# to claim.
	}

	return 0;
}
// ^ ^ [WIP] / [FIXME]: solvers must also be updated on new difficulty! ^ ^


uint64_t SetMiningTarget (const Uint256 u256_newtarget, miningParameters *params)
{
	// [WIP / FIXME]: difficulty# set? <------
LOG_IF_F(INFO, DEBUGMODE, "Setting new mining target, acquiring mtx_coreparams.");
	//std::lock_guard<std::mutex> lockg(mtx_coreparams);

//	gU256_MiningTarget = u256_newtarget;	// stored as a Uint256 for easy direct comparison in recurring_network_tasks(). now redundant
	u256_newtarget.getBigEndianBytes( params->target_bytes );	//out to 256-bit global mining target (as byte array) <-
	print_bytes( params->target_bytes, 32, "new target" );	//debug

// get most-significant 64 bits of U256 target:
	uint64_t u64_target{ 0 };	// 0ull?
	memcpy(&u64_target, &u256_newtarget.value[6], 8);	/* success!  could also copy the relevant 8 bytes from gMiningTarget_bytes[].					*/
	if (u64_target) {									/* note: a Uint256's 32-bit .value[] elements are stored little-endian, so getting [6] and [7]. */
		cpu_target = u64_target;						// "
		params->uint64_target = u64_target;
		LOG_IF_F(INFO, HIGHVERBOSITY, "Set 64-bit target:	%" PRIx64 ".", u64_target);
	} else { //target of 0 can't be solved for:
		LOG_IF_F(ERROR, NORMALVERBOSITY, "Bad 64-bit target of 0!" BUGIFHAPPENED);	//<-- [fixme] if this happens!
		if (DEBUGMODE) { gStopCondition = true; }
		return 0;	// 0=err
	}

	return u64_target;
}


unsigned int GetSolnQueueContents(void)
{ // [TESTME]
	if (q_solutions.empty())
		return 0;
	const unsigned int output = static_cast<unsigned int>(q_solutions.size());
	LOG_IF_F(INFO, DEBUGMODE, "Queue contents:	%zu sol'ns", q_solutions.size());
	return output;
	//return (unsigned int)(q_solutions.size());
}

void ResetCoreParams(miningParameters* params)
{
	LOG_IF_F(INFO, DEBUGMODE, "Resetting core params. ");
//	std::lock_guard<std::mutex> lockg(mtx_coreparams);  // acquire mutex  (for changing gMiningParameters, or any globals that still remain).
	
	// [Note to self]  When reading these strings, check if they're empty.
	params->mintingaddress_str.clear();			// 
	params->mineraddress_str.clear();
	params->challenge_str.clear();				// was 0x00
	params->target_str.clear();					// gStr_MiningTarget = "0x00";

	params->uint64_target = 0;
	params->difficultyNum = 0;
	memset(params->target_bytes, 0, 32);

	params->challenge_changed = false;
	params->difficulty_changed = false;
	params->mintaddr_changed = false;
	params->target_changed = false;
	params->params_changing = false;
// consider resetting these after each mining session:
// doEveryCalls_Values[] = {0};
// gStr_TokenName.clear();
}



//
bool CheckRequestIDString(json11::Json j_reply, const std::string str_id)
{
	if (!j_reply.object_items().count("id")) { 
		LOG_F(WARNING, "id key not found in response.");
		return false; }
	if (!j_reply["id"].is_string()) {
		LOG_F(WARNING, "id key in network response is not a string.");
		return false; }
	//
	if (j_reply["id"].string_value() == str_id)  { return true; }  // match
	 else {
		LOG_F(WARNING, "id mismatch in network response: %s \n", j_reply["id"].string_value().c_str()); // <- remove?
		return false; }
}


// Callback function for the libcurl request performed in PoolRequest(), gets response.
size_t JSONRPC_Response_Write_Callback(const char* contents, const size_t size, const size_t nmemb, const void* userp)
{
	if (DEBUGMODE && DEBUG_NETWORK) { printf("Response: %s \n", contents); }
	((std::string*)userp)->append((char*)contents, size * nmemb);
	return size * nmemb;
}

//
// [idea] pass the same handle in to each call as function param, rather than running curl_easy_init() each time?
std::string LibCurlRequest(const char* data /*, CURLcode *result*/)
{ // POOLREQUEST: Receives data from pool and submits shares.
  // [todo]: batch up JSON requests where possible for less overhead impact
  // [todo]: a function (on its own thread?) which handles these requests, modularize this func.
	CURL* curlHandle = curl_easy_init();
	if (!curlHandle) {
		LOG_F(ERROR, "LibCURL handle error. Unable to access the network.");
		domesg_verb("LibCurl error. Unable to access the network.", true, V_LESS);  // always
		return "Error"; }
//
#ifdef DEBUGGING_NETWORK_
	if (gVerbosity == V_DEBUG)  { printf("REQUEST: %s \n", data); } // <- Remove (previewing the json request)
#endif
//
	std::string str_response{""};		// stores the raw response from the pool (expecting JSON format)
	struct curl_slist* headers = NULL;
	headers = curl_slist_append(headers, "content-type: application/json");

	//REF: curl_easy_setopt(curl, CURLOPT_USERPWD, ""); // username:password
	curl_easy_setopt(curlHandle, CURLOPT_HTTPHEADER, headers);
	curl_easy_setopt(curlHandle, CURLOPT_URL, gStr_PoolHTTPAddress.c_str());  // todo/idea: if user's pool address is bad/down, use default Mike's or TMP?
	curl_easy_setopt(curlHandle, CURLOPT_POSTFIELDSIZE, (long)strlen(data));  // <-
	curl_easy_setopt(curlHandle, CURLOPT_POSTFIELDS, data);
	curl_easy_setopt(curlHandle, CURLOPT_USE_SSL, CURLUSESSL_TRY);
	curl_easy_setopt(curlHandle, CURLOPT_WRITEDATA, &str_response);  // <-

	// set timeouts (todo: user-configurable?)
	curl_easy_setopt(curlHandle, CURLOPT_DNS_CACHE_TIMEOUT, 3);		// secs, dns only
	curl_easy_setopt(curlHandle, CURLOPT_TIMEOUT_MS, 8000);			// ms, whole request
	curl_easy_setopt(curlHandle, CURLOPT_CONNECTTIMEOUT, 5);		// secs, connection portion (TODO: user-configurable timeouts)
	curl_easy_setopt(curlHandle, CURLOPT_ACCEPTTIMEOUT_MS, 5000);	// ms, waiting for connection acceptance

	// callback function for getting response, result to `str_response`
	curl_easy_setopt(curlHandle, CURLOPT_WRITEFUNCTION, JSONRPC_Response_Write_Callback);

	const CURLcode curlResult = curl_easy_perform(curlHandle);	// perform transfer operation, get result
	curl_slist_free_all(headers);								// free slist
	curl_easy_cleanup(curlHandle);

	// === check the result of the libcurl operation ===
	if (curlResult != CURLE_OK)
	{
		// non-OK return code from LibCURL operation
		str_net_lastcurlresult = std::string(curl_easy_strerror(curlResult)); // save error string from libcurl
		domesg_verb("Network error: " + str_net_lastcurlresult, true, V_MORE);  //

		++stat_net_network_errors;	// count a network error (total)
		++stat_net_consecutive_errors;		// count a sequential network error
		stat_net_last_success = false;
		stat_net_lastpoolop_totaltime = -1;  // these readings are not reliable if result is != CURLE_OK.
		return "Error: " + str_net_lastcurlresult;
	}

	// TODO/FIXME: enforce a maximum length and truncate or reject over-long responses like Cloudflare pages <--
	if (str_response.length() > DEF_MAX_NETRESPONSE_LEN) { return "Error: response too long "; }

	if (str_response.find("cosmic") == std::string::npos)
	{ // quick check for part of 'id' key expected (so we can skip parsing early if invalid response)
		printf("LibCurlRequest(): Bad json response \n");
		++stat_net_consecutive_errors;
//		++stat_net_network_errors;
		return "Error: bad json response ";
	}

	double timeTaken_secs{ 0 };
	// [TODO / FIXME]: if getinfo call failed:
	if (curl_easy_getinfo(curlHandle, CURLINFO_TOTAL_TIME, &timeTaken_secs) != CURLE_OK) { 
		str_net_lastcurlresult = std::string( curl_easy_strerror(curlResult) );
//		++stat_net_network_errors;   //	...
		return "Error: " + str_net_lastcurlresult;
	}
	stat_net_lastpoolop_totaltime = timeTaken_secs * 1000;  // seconds to ms
//
#ifdef DEBUG_NETWORK
	printf("network operation took %f ms \n", stat_net_lastpoolop_totaltime);	// <- remove [WIP]
#endif
//
	if (stat_net_avgpoolop_totaltime > 0) {
		stat_net_avgpoolop_totaltime += stat_net_lastpoolop_totaltime;			// 
		stat_net_avgpoolop_totaltime /= 2;										// very basic average again previous (for now. [todo])
	} else { stat_net_avgpoolop_totaltime = stat_net_lastpoolop_totaltime; }	// first network operation this session

	str_net_lastcurlresult = "OK";		//  [fixme] ?
	stat_net_last_success = true;		//  [fixme] ?
	stat_net_consecutive_errors = 0;	// valid? reset consecutive net error counter <-- ensure net pause behavior consistent.

	return str_response;	//												  many consecutive bad json responses or many libcurl errors <-
}


GetParmRslt Pool_GetChallenge( const bool initial, miningParameters *params )
{
	//const std::string str_requestid = GetRequestIDString();
	json11::Json j_request_getchallenge = json11::Json::object ({
		{ "jsonrpc", "2.0" },
		{ "method", "getChallengeNumber" },
		{ "id", "cosmic" /*str_requestid*/ }
	});
//	const std::string str_request_getchallenge = j_request_getchallenge.dump();
//	printf("\nDEBUG: Pool_GetChallenge(): j_request JSON dump: %s \n", str_request_getchallenge.c_str());

	// send request:
	const std::string str_response = LibCurlRequest( j_request_getchallenge.dump().c_str() );  // <- was: str_request_getchallenge.c_str()
	if (!checkErr_a(str_response)) { return GetParmRslt::GETPARAM_ERROR; }  // check for error from libcurlrequest.

	// condense this w/ a helper func [TODO]:
	std::string str_parse_err{ "" };
	const json11::Json j_response = json11::Json::parse(str_response, str_parse_err, json11::JsonParse::STANDARD);
	if (!str_parse_err.empty()) { //if error string's length is >0
		domesg_verb("Error parsing challenge from pool: " + str_parse_err, true, V_DEBUG);  // <- [FIXME] ?
		return GetParmRslt::GETPARAM_ERROR; }	// 1: err
	if (!j_response.object_items().count("result")) {
		domesg_verb("Pool_GetChallenge(): key `result` not found. ", false, V_NORM);
		return GetParmRslt::GETPARAM_ERROR; }	// "
	if (!j_response["result"].is_string()) {
		domesg_verb("Pool_GetChallenge(): key `result` is not a string. ", false, V_NORM);
		return GetParmRslt::GETPARAM_ERROR; }	// "

//	const std::string str_parsed = j_response["result"].string_value();
	//checkString(): sanity-check, err-check. expects 66 characters: "0x" and 32 bytes
	if (!checkString( j_response["result"].string_value(), 66, true, true )) {  /*expect 32 bytes preceded by 0x*/
		domesg_verb("Error retrieving challenge from pool.", true, V_DEBUG);
		return GetParmRslt::GETPARAM_ERROR; }	// 1=err
	//			
	// NEEDS CHECK OF ID KEY STRING <--- [TODO/FIXME]
	//
	//std::lock_guard<std::mutex> lockg(mtx_coreparams);
	if (j_response["result"].string_value() != params->challenge_str || initial)				/* cannot read it while new one is being written				*/
	{ //set the global challenge (if new or initial retrieval):
		params->challenge_str = j_response["result"].string_value();
		domesg_verb("New Challenge:  " + j_response["result"].string_value(), true, V_NORM);	//<-- use log callback [TODO].
		LOG_IF_F(INFO, HIGHVERBOSITY, "Got Challenge: %s", j_response["result"].string_value().c_str());

		return GetParmRslt::OK_PARAM_CHANGED;			// OK: new challenge set
	} else { return GetParmRslt::OK_PARAM_UNCHANGED; }	// OK: challenge unchanged
}

// GETDIFFICULTYTARGET(): Gets the difficulty # -and- decimal target from the pool/node. Returns: 0=OK, param unchanged. 1=if error. 2=OK, difficulty/target changed.
// Moved here from net_pool.cpp to this file (which is compiled with /clr). Find it a more appropriate home after migrating the native code to a separate library.
// previous function body got the difficulty # and the target. separate?
GetParmRslt Pool_GetDifficulty( const bool initial, miningParameters *params )
{ 
	// (prototype) json request for tokenpool method `getMinimumShareDifficulty`:
	// [note]: the miner's eth address sent as a parameter, to get miner-specific difficulty, VARDIFF, etc.
	json11::Json j_request_getdifficulty = json11::Json::object ({ 
		{ "jsonrpc", "2.0" }, 
		{ "method", "getMinimumShareDifficulty" }, 
		{ "params", json11::Json::array({ params->mineraddress_str }) },	/* [note]:  not a key pair- just param 0, the miner's eth address */
		{ "id", "cosmic" }												/* or: GetRequestIDString()- [WIP].  */
	});

// === send request: ===
	const std::string str_response = LibCurlRequest( j_request_getdifficulty.dump().c_str() );  // [todo]: this string possibly unneccessary.	//str_request_getdifficulty.c_str()
	if (!checkErr_a(str_response))  { return GetParmRslt::GETPARAM_ERROR; }  // -1: err

// parse response:
	std::string str_parse_error{ "" };  //json11::Json::parse() error, if any
	const json11::Json j_response = json11::Json::parse(str_response, str_parse_error, json11::JsonParse::STANDARD);
	if (!str_parse_error.empty()) { //or:  if (errStr.length())
		domesg_verb("Error parsing difficulty from pool: " + str_parse_error, true, V_DEBUG);  // [FIXME] ?
		return GetParmRslt::GETPARAM_ERROR; }  // 1: err
//moved here:
	if (!CheckRequestIDString(j_response, "cosmic")) {
		LOG_F(WARNING, " `id` mismatch or missing in reply"); // <--
		return GetParmRslt::GETPARAM_ERROR;  // NEEDS CHECK OF ID KEY STRING <--- [ WIP / FIXME]
	}
//moved here.
	if (!j_response.object_items().count("result")) {  // [WIP]: legal to call .is_number() if key absent from string?
		LOG_F(WARNING, "Pool_GetDifficulty(): key `result` not found.");
		return GetParmRslt::GETPARAM_ERROR; }  // "
	if (!j_response["result"].is_number()) {
		LOG_F(WARNING, "Pool_GetDifficulty(): key `result` is not a number.");
		return GetParmRslt::GETPARAM_ERROR; }  // "

	const uint64_t u64_parsed = j_response["result"].int_value();
	if (u64_parsed == 0 || u64_parsed == UINT64_MAX) {
		domesg_verb("Error retrieving difficulty from pool. "/* + responseStr*/, true, V_DEBUG);
		return GetParmRslt::GETPARAM_ERROR; }  // "
	
// [WIP] ensure multiple threads cannot simultaneously access mining params.
	//std::lock_guard<std::mutex> lockg(mtx_coreparams);
	if (initial || u64_parsed != params->difficultyNum)
	{ // set the global pool difficulty# (on initial diff retrieval, or if new.)
		params->difficultyNum = u64_parsed;  // was `gU64_DifficultyNo`
		VLOG_F (V_NORM, "New Difficulty #:	%s ", std::to_string(u64_parsed).c_str() );  // alternatively: "... %" PRIu64 ". ", u64_parsed);
		domesg_verb("New Difficulty:	" + std::to_string(u64_parsed), true, V_NORM);
		return GetParmRslt::OK_PARAM_CHANGED;			// 2: OK, new difficulty set
	} else { return GetParmRslt::OK_PARAM_UNCHANGED; }	// 0: OK, no change
}


// WIP/TODO: getMinimumShareTarget
//	 - Pool_GetDifficulty(): need to send the user's mining address as param?  (to facilitate custom difficulty set by the pool, incl. VARDIFF).
// FreshTarget() needs to read from the NEWLY RETRIEVED target/difficulty as needed to set the Computed or Pool-Reported Target.

// POOL_GETMINTINGADDRESS(): 0= OK, minting addr. unchanged. 1= Error occurred. 2= OK, minting addr. has changed!
GetParmRslt Pool_GetMintingAddress ( const bool initial, miningParameters *params )
{
	// (prototype) json request to get the Pool's minting address:
	json11::Json j_request_mintaddr = json11::Json::object({
		{ "jsonrpc", "2.0" },
		{ "method", "getPoolEthAddress" }, 
		{ "id", "cosmic" }
	});

	// send JSON-RPC request:
	const std::string str_response = LibCurlRequest( j_request_mintaddr.dump().c_str() );
	if (!checkErr_a(str_response)) { return GetParmRslt::GETPARAM_ERROR; }  // check for error from libcurlrequest.

	std::string str_perror {""};
	const json11::Json j_response = json11::Json::parse( str_response, str_perror, json11::JsonParse::STANDARD );
	if (!str_perror.empty()) {
		VLOG_F(V_NORM, "Error parsing pool minting address from pool:  %s", str_perror.c_str());  // <- [FIXME] ?
		return GetParmRslt::GETPARAM_ERROR; }  // 1: err
//new [TESTME]. moved here:
	if (!CheckRequestIDString(j_response, "cosmic")) {
		LOG_F(WARNING, " `id` mismatch or missing in reply"); // <--
		return GetParmRslt::GETPARAM_ERROR;  // NEEDS CHECK OF ID KEY STRING <--- [ WIP / FIXME]
	}
//new [TESTME]. moved here
	if (!j_response.object_items().count("result")) {
		VLOG_F(V_NORM, "Pool_Getpool minting address(): key `result` not found.");
		return GetParmRslt::GETPARAM_ERROR; }  // "
	if (!j_response["result"].is_string()) {
		VLOG_F(V_NORM, "Pool_Getpool minting address(): key `result` is not a string.");
		return GetParmRslt::GETPARAM_ERROR; }  // "

//	const std::string str_parsed = j_response["result"].string_value();
//	if (!checkString(str_parsed, 42, true, true)) { /* checkString(): sanity-check, err-check. expects 42-character address preceded by `0x` */
	if (!checkString(j_response["result"].string_value(), 42, true, true)) { /* checkString(): sanity-check, err-check. expects 42-character address preceded by `0x` */
		domesg_verb("Error retrieving minting address from pool.", true, V_DEBUG);
		return GetParmRslt::GETPARAM_ERROR; }  // 1=err

	//std::lock_guard<std::mutex> lockg(mtx_coreparams);	// note: lock so pool mintaddr can't be read while new one's written
	if ( initial || j_response["result"].string_value() != params->mintingaddress_str )
	{ // set the global pool minting address (if new, also on initial minting addr retrieval):
		//
		if (!checkString(j_response["result"].string_value(), 42, true, true))
		{ // must be 42-character hex string- 20 bytes preceded by `0x`.
			LOG_F( INFO, "Bad pool minting address:  %s", j_response["result"].string_value().c_str() );	// <- to log
			return GetParmRslt::GETPARAM_ERROR;			// bad minting address
		}
		LOG_IF_F(INFO, gVerbosity>=V_NORM, "New pool minting address:  %s", j_response["result"].string_value().c_str());
		domesg_verb("New pool minting address: " + j_response["result"].string_value(), true, V_NORM);	// <- ui event
		params->mintingaddress_str = j_response["result"].string_value();
		return GetParmRslt::OK_PARAM_CHANGED;	// OK: new pool minting address set
//
	}
	else { return GetParmRslt::OK_PARAM_UNCHANGED; }	// OK: pool minting address unchanged
}

GetParmRslt Pool_GetTarget( const bool initial, miningParameters *params )
{
	// (prototype) json request to get the Pool's minimum share target:
	json11::Json j_request_target = json11::Json::object({
		{ "jsonrpc", "2.0" },
		{ "method", "getMinimumShareTarget" },
		{ "params", json11::Json::array({ params->mintingaddress_str }) },		/* [note]:  not a key pair. just param 0: miner's eth address.  was: gStr_MinerEthAddress.	*/
		{ "id", "cosmic" }
	});

	// send request:
	const std::string str_response = LibCurlRequest(j_request_target.dump().c_str());
	if (!checkErr_a(str_response)) { return GetParmRslt::GETPARAM_ERROR; }  // if err from libcurlrequest, don't proceed

	std::string str_parse_err{ "" };
	const json11::Json j_response = json11::Json::parse(str_response, str_parse_err, json11::JsonParse::STANDARD);
	if (!str_parse_err.empty()) {
		domesg_verb("Error parsing pool minimum share target: " + str_parse_err, true, V_DEBUG);  // <- [FIXME] ?
		return GetParmRslt::GETPARAM_ERROR; }  // 1: err
//new [TESTME]. moved here:
	if (!CheckRequestIDString(j_response, "cosmic")) {
		LOG_F(WARNING, "`id` mismatch or missing in reply"); // <--
		return GetParmRslt::GETPARAM_ERROR;  // NEEDS CHECK OF ID KEY STRING <--- [ WIP / FIXME]
	}
//new [TESTME]. moved here
	if (!j_response.object_items().count("result")) {
		domesg_verb("Pool_GetTarget(): key `result` not found. ", false, V_NORM);
		return GetParmRslt::GETPARAM_ERROR; }  // "
	if (!j_response["result"].is_string()) {
		domesg_verb("Pool_GetTarget(): key `result` is not a string. ", false, V_NORM);
		return GetParmRslt::GETPARAM_ERROR; }  // "

	//checkString(): sanity-check, err-check. expects 66 characters: "0x" and 32 bytes
	const std::string str_parsed = j_response["result"].string_value();
	if (!checkString(str_parsed, 0, false, false) || str_parsed.length() > 78 || str_parsed.length() < 1)
	{ // [CHECKME / TESTME] should be appropriate maximum length for an unsigned 256-bit decimal number ^
		domesg_verb("Error retrieving pool minimum share target. "/* + responseStr*/, true, V_DEBUG);
		return GetParmRslt::GETPARAM_ERROR; }  // 1: err

	// briefly lock `mtx_coreparams` so Solvers can't read this target string while new one is being written in.
	//std::lock_guard<std::mutex> lockg(mtx_coreparams);
	if ( initial || str_parsed != params->target_str)	/* [TESTME] */
	{ // set the global pool minting address:
		//std::lock_guard<std::mutex> lockg(mtx_coreparams);					// <- here. acquire mutex...
		params->target_str = str_parsed;									//was: gStr_MiningTarget

		LOG_IF_F(INFO, HIGHVERBOSITY, "Incoming Target:	%s", str_parsed.c_str());	//
		domesg_verb("Incoming Target: " + str_parsed, true, V_NORM);		//new min-share target. [todo]: use log callback <-
		return GetParmRslt::OK_PARAM_CHANGED;	// OK: new pool target set
	} else
		return GetParmRslt::OK_PARAM_UNCHANGED;
}



// [TODO]: common difficulty# for both modes
//
SubmitShareResult Pool_SubmitShare( QueuedSolution *p_soln, miningParameters *params )  /* [WIP]  reworking this */
{ //...
	//const std::string str_requestid = "cosmic";		//GetRequestIDString();
	LOG_IF_F(INFO, DEBUGMODE, "Submitting Solution \n" "Nonce: %s \nChallenge: %s \n", p_soln->solution_string.c_str(), p_soln->challenge_string.c_str() );
	LOG_IF_F(INFO, DEBUGMODE, "Difficulty: %" PRIu64 " \nIs Devshare: %s \n", p_soln->poolDifficulty, std::to_string(p_soln->devshare).c_str());	// devshare: true|false.
	
	// (prototype) json request to get the Pool's minimum share target [WORKS]:
	json11::Json j_request_submitshare = json11::Json::object({
		{ "jsonrpc", "2.0" },
		{ "method", "submitShare" },
		{ "params", json11::Json::array({
			p_soln->solution_string, 
			p_soln->devshare ? gStr_DonateEthAddress : params->mineraddress_str, 
			p_soln->digest_string, 
			double(p_soln->poolDifficulty),										/* [note]: json11 handles all numbers as doubles internally */
			p_soln->challenge_string })
		},
		{ "id", "cosmic" }	//{ "id", str_requestid }
	});
	// ^ use the difficulty# the solution was solved for. If VARDIFF Pool has just adjusted the difficulty, nonces not meeting new diff. target must be
	//	 cleared from the queue (they won't be accepted by the pool.)
	//   [FIXME]: ^ a patch, these difficulty #s should be the same unless network issue. (chasing a compatibility bug w / Mvis.ca vardiff pool.) <--
	//
	std::string str_parse_error{ "" };  // parse error (if any)
	std::string str_response = LibCurlRequest( j_request_submitshare.dump().c_str() );
	if (!checkErr_b(&str_response, true)) { //trim "Error: " from string.
		domesg_verb("Network error while submitting share: " + str_response, true, V_NORM);						// <- use log callback [todo].
		LOG_IF_F(INFO, NORMALVERBOSITY, "Network error while submitting share:	%s ", str_response.c_str());	// <-
		//++stat_net_network_errors;	 // <- [wip]
		//++stat_net_badjson_responses;  // <- "
		return SubmitShareResult::SUBMIT_ERROR; }  // error code 1: indicates to calling func _not_ to pop the solution, re-send will be attempted.
//
//	=== parse `str_response`. id and result (bool) must match expected format: ===
	json11::Json j_response = json11::Json::parse( str_response, str_parse_error, json11::JsonParse::STANDARD );
	if (str_parse_error.length()) { //if parsing error occurred
		LOG_F(WARNING, "Error while parsing response to submitShare:	%s", str_parse_error.c_str());			// <--
		domesg_verb("Error while parsing response to submitShare:	" + str_parse_error, true, V_NORM);	// <- _MORE/_DEBUG only?
		return SubmitShareResult::SUBMIT_ERROR; }  // 1:err (see above)
//
	if (!CheckRequestIDString(j_response, "cosmic")) {	// [WIP] check the id of the returned string:
		LOG_F(WARNING, " `id` mismatch or missing in reply"); // <--
		domesg_verb("Pool_SubmitShare(): key `id` mismatch or missing ", false, V_DEBUG);
		return SubmitShareResult::SUBMIT_ERROR; }
//
	if (!j_response.object_items().count("result")) {
		LOG_IF_F(WARNING, NORMALVERBOSITY, "Pool_SubmitShare(): key `result` not found");
		domesg_verb("Pool_SubmitShare(): key `result` not found.", false, V_NORM);
		return SubmitShareResult::SUBMIT_ERROR; }	//condensed
	if (!j_response["result"].is_bool()) {
		LOG_IF_F(WARNING, NORMALVERBOSITY, "Pool_SubmitShare(): key `result` is not a boolean");
		domesg_verb("Pool_SubmitShare(): key `result` is not a boolean.", false, V_NORM);
		return SubmitShareResult::SUBMIT_ERROR; }
	if (j_response["result"].bool_value() == true) { //check result of `submitShare`:
		LOG_IF_F(INFO, NORMALVERBOSITY, "Share accepted by pool (%f ms).", stat_net_lastpoolop_totaltime);					// [TODO]: refactor network stats stuff.
		domesg_verb("Share accepted by pool (" + std::to_string(stat_net_lastpoolop_totaltime) + " ms) ", true, V_NORM);	// [todo]: use log callback.
		return SubmitShareResult::SHARE_ACCEPTED; } // 0: accepted

	LOG_F(WARNING, "Share rejected:	%s ", p_soln->solution_string/*.substr(0,32)*/.c_str());  // [todo]: use log callback instead
	domesg_verb("Share rejected:	" + p_soln->solution_string/*.substr(0,32)*/, true, V_NORM);  // [idea] ^
	return SubmitShareResult::SHARE_REJECTED;	// calling func should pop solution from the queue.
}

