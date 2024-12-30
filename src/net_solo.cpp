#pragma once
// COSMiC V4
// Miner<->Ethereum Node comm for Solo Mining

#include <msclr/marshal.h>			// marshalling native<->managed types
#include "msclr/marshal_cppstd.h"	// "
#include <cinttypes>				// was: #include <stdint.h>
#include <iostream>					// i/o
#include <stdlib.h>					// "
#include <string>
#include <sstream>					// stringstream
#include <mutex>

#include <curl/curl.h>				// network functionality
#include <json11/json11.hpp>		// json parsing and objects
#include <bitcoin-cryptography-library/cpp/Uint256.hpp>	// <- very helpful unsigned 256-bit type
#include <bitcoin-cryptography-library/cpp/Keccak256.hpp>	// a reference Keccak256, from Nayuki bitcoin lib
#include <libsodium/sodium.h>		// sodium cryptography lib
//#include <json11.hpp>			

//#include <secp256k1.h>			// consolidating headers, see net_solo.hpp
//#include <secp256k1_recovery.h>	// 


// [TODO]:
// - finish revising this file
// - consolidate headers?
// = revise functions:
// - Eth_GetTransactionCount()
// - ...


// ===LIBSECP256K1 STUFF===
#include <stdint.h>					// underlying types of ethereum-rlp's custom types (i.e. EthereumTxRequest_signature_r_t)
#include "RLP.h"					// <- [fixme]: <ethereum-rlp/RLP.h>

#include "defs.hpp"
//#include "../Core/mine_events.h"
//
#include "util.hpp"			// for checkErr_a(), _b()
#include "network.hpp"		// common network stuff (both modes). for: Mining Params.
#include "net_solo.h"
//#include "net_rlp.hpp"		// (new)

#define DEF_DEBUG_NETWORK_SOLO

#define NONCEDATA_LENGTH 32			// length in bytes of the nonce data, for signing process  [MOVEME].

extern struct txViewItem gTxViewItems[DEF_TXVIEW_MAX_ITEMS];	// CosmicWind.h <--
extern std::string gStr_SoloNodeAddress;						//
extern std::string gStr_ContractAddress;						//coredefs.hpp <----
//extern int gSolo_ChainID;										//
//extern uint64_t gNum_SolutionCount[CUDA_MAX_DEVICES];			//


using namespace System;					// .NET integer types, etc.
using namespace System::Globalization;	// number format and region stuff
using namespace System::Numerics;		// for BigInteger class
//using namespace System::Diagnostics;

// NET_RLP.CPP	(Native function)
bool AssembleTX(const int solution_no, const Uint256 donation_U256, uint64_t rawTx_u64[24], unsigned int* payload_length);

// === globals for solo mode ===
// [wip]: finish condensing into `params` object.
std::string gTxCount_hex{ "" };					// hex representation (useful?)
long gNum_SoloSolutionsFound{ 0 };

Uint256 gSoloEthBalance256{ Uint256::ZERO };	// (wip)
												// [TODO]: option to auto-stop mining if balance drops below req'd gas cost to submit solution.
Uint256 gSk256 { Uint256::ZERO };				// Skey as uint256
uint64_t gU64_GasPrice{ 3000000000 };			// 3 G-wei price per unit (default)
uint64_t gU64_GasLimit{ 200000 };				// 200,000 max units (default)

std::string gStr_GasPrice_hex{ "" };			// "0x00"
std::string gStr_GasLimit_hex{ "" };			// "0x00"
std::string gStr_SoloNodeAddress{ "" };			// "0x00"
std::string gStr_SoloEthAddress{ "" };			// "0x00"	- derived from privkey (public eth address we're mining with).

unsigned int gSoloNetInterval{ 700 };			// Solo Network access interval, in ms  (todo: make sure this is populated at CosmicWind _Shown time) <--
short gDonationsLeft{ 0 };				// for auto-donation in solo mode. increased when a solo-mode sol'n-tx is confirmed. WIP: consolidate xfers when possible

int gTxView_WaitForTx{ NOT_WAITING_FOR_TX };


// sets `last_node_response` of a  to a NODERESPONSE_ type by error string comparison
// [TODO]: consider using error code instead.
void HandleNodeError(const std::string& error_mesg, const uint64_t error_code, const int txViewItemNo)
{ 
	// let's just compare the error codes or messages verbatim, when available?) <--- [WIP]
	// acquire mutex mtx_txview ! <--- [TODO / FIXME]

	//TODO: use error_code ? 
	if (txViewItemNo < 0) {	/* max? <--- */
		LOG_IF_F(WARNING, DEBUGMODE, "HandleNodeError(): bad txview item # %d", txViewItemNo);
		return;  // only relevant to txview item (solution) submission
	}

	// TODO: use err# instead? (should work for Infura, try other endpoint/node types)
	if (error_mesg.find("insufficient funds") != std::string::npos) {
		gTxViewItems[txViewItemNo].last_node_response = NODERESPONSE_INSUFFICIENT_FUNDS;  // user needs to deposit ether(gas) or sender addr. miscommunicated
		// TODO: set Stop Condition to OUT OF GAS if the setting is enabled. Should be on by default. <------
	}
	else if (error_mesg.find("nonce too low") != std::string::npos) {
		domesg_verb("NONCE TOO LOW: netNonce " + std::to_string(gTxViewItems[txViewItemNo].networkNonce), true, V_DEBUG);  // <-- 
//		gTxViewItems[txViewItemNo].networkNonce += 1;  // this is a hack! <-- (fixme/todo)
//		domesg_verb("increased network nonce of txview item# " + std::to_string(txViewItemNo) + " to " + std::to_string(gTxViewItems[txViewItemNo].networkNonce), true, V_DEBUG);
		gTxViewItems[txViewItemNo].last_node_response = NODERESPONSE_NONCETOOLOW;
	}
	else if (error_mesg.find("replacement transaction underpriced") != std::string::npos) {
		domesg_verb("REPLACEMENT TXN UNDERPRICED: networkNonce " + std::to_string(gTxViewItems[txViewItemNo].networkNonce), true, V_DEBUG);  // <--
		LOG_F(WARNING, "HandleNodeError(): solution # %d has invalid network nonce %" PRIx64 "!", txViewItemNo, gTxViewItems[txViewItemNo].networkNonce);
//		gTxViewItems[txViewItemNo].networkNonce += 1;  // this is a hack! <-- (fixme/todo)
//		domesg_verb("increased network nonce of txview item# " + std::to_string(txViewItemNo) + " to " + std::to_string(gTxViewItems[txViewItemNo].networkNonce), true, V_DEBUG);
		gTxViewItems[txViewItemNo].last_node_response = NODERESPONSE_REPLACEMENT_UNDERPRICED;	// a txn with this nonce already exists
	}
	//else if (error_mesg.find("...")
	//	gLastResponseFromNode = NODERESPONSE_... ;													// TODO: other
	else gTxViewItems[txViewItemNo].last_node_response = NODERESPONSE_OK_OR_NOTHING;

	return;
}



#define DEBUG_NET_PRINTJSONMESGS		0	// print and/or log network requests/replies. spammy.

size_t Solo_JSONRPC_Response_Write_Callback(const char* contents, const size_t size, const size_t nmemb, const void* userp)
{// retrieves JSON-RPC response from node
//	LOG_IF_F(INFO, DEBUGMODE, "Response:	%s ", contents);
	((std::string*)userp)->append((char*)contents, size * nmemb);
	return size * nmemb;
}

// TODO: These operations can take longer. Consider using Curl MULTI interface to avoid doing a long operation
//		 on the main thread, which could hold up the GUI / affect UX.	[WIP]: using backgroundWorkers to prevent this.
std::string Solo_JSON_Request(const std::string& jsonData, const bool strictTimeouts)
{
	CURL* curlHandle = curl_easy_init();
	if (!curlHandle) {
		domesg_verb("LibCURL handle error. Unable to access the network.", true, V_LESS); // always show
		return "Error: Libcurl Init failed";
	}

	CURLcode curlResult = CURLE_OK;	// init
	std::string errorString = "";
	std::string responseBuffer = "";  // stores raw pool response

	// rest of the declarations
	struct curl_slist* headers = NULL;
	headers = curl_slist_append(headers, "Content-Type: application/json");
	curl_easy_setopt(curlHandle, CURLOPT_HTTPHEADER, headers);

	// TODO: error chex of URL, anything possibly not caught by ConfigSoloMining form's code, or bad cfg <-
	curl_easy_setopt(curlHandle, CURLOPT_URL, gStr_SoloNodeAddress.c_str());  // remote eth node
	curl_easy_setopt(curlHandle, CURLOPT_POSTFIELDSIZE, (long)(strlen(jsonData.c_str())));
	curl_easy_setopt(curlHandle, CURLOPT_POSTFIELDS, jsonData.c_str());
	//REF: curl_easy_setopt(curl, CURLOPT_USERPWD, ""); // username:password
	curl_easy_setopt(curlHandle, CURLOPT_USE_SSL, CURLUSESSL_TRY);
	curl_easy_setopt(curlHandle, CURLOPT_WRITEDATA, &responseBuffer);  // where our response will go

	// set timeouts (TODO: user-configurable?)
	if (strictTimeouts) {
		curl_easy_setopt(curlHandle, CURLOPT_DNS_CACHE_TIMEOUT, 2);		// secs, dns only
		curl_easy_setopt(curlHandle, CURLOPT_TIMEOUT_MS, 3500);			// ms, whole request
		curl_easy_setopt(curlHandle, CURLOPT_CONNECTTIMEOUT, 3);		// secs, connection portion (TODO: user-configurable timeouts)
		curl_easy_setopt(curlHandle, CURLOPT_ACCEPTTIMEOUT_MS, 2000);	// ms, waiting for connection acceptance
	} else {
		curl_easy_setopt(curlHandle, CURLOPT_DNS_CACHE_TIMEOUT, 4);		// secs, dns only
		curl_easy_setopt(curlHandle, CURLOPT_TIMEOUT_MS, 7000);			// ms, whole request
		curl_easy_setopt(curlHandle, CURLOPT_CONNECTTIMEOUT, 5);		// secs, connection portion (TODO: user-configurable timeouts)
		curl_easy_setopt(curlHandle, CURLOPT_ACCEPTTIMEOUT_MS, 5000);	// ms, waiting for connection acceptance
	}
	// callback function for getting response, result goes in responseBuffer.
	curl_easy_setopt(curlHandle, CURLOPT_WRITEFUNCTION, Solo_JSONRPC_Response_Write_Callback);

	curlResult = curl_easy_perform(curlHandle);		// perform transfer operation, get result
	curl_slist_free_all(headers);					// free slist
	curl_easy_cleanup(curlHandle);					// clean up libcurl easy handle
	// check the result of the libcurl operation
	if (curlResult == CURLE_OK)
	{
		double timeTaken = 0;  // in seconds
		//str_net_lastcurlresult = "OK";
		//stat_net_lastcurlopsuccess = true;
		
		// check if valid json response [TODO]: revise this <-
		// FIXME: conditionally, parse the response here, check the ID, and pass along reference to the JSON object? What's faster?
		// expect "cosmic" in the "id" field of the response, matching the request. if not in response: likely an error page (cloudflare?)
		if (responseBuffer.find("cosmic") == std::string::npos)
		{	// otherwise: likely not the expected json response
			// [NOTE]:  if infura project is disabled, may get a reply of "account disabled", not json-formatted.
			if (responseBuffer.find("disabled") != std::string::npos) { 
				domesg_verb("Node reports 'account disabled'. Please check your node/APIkey/project.", true, V_LESS);  // always
				return "Error: Node reports 'account disabled'. "; }	// (WIP)
			 else {
				domesg_verb("Received bad JSON-RPC response: " + responseBuffer, true, V_DEBUG); // <--remove
				return "Error: Bad JSON-RPC Response"; }
		}
		else {  /* valid JSON-RPC response */
			LOG_IF_F(INFO, DEBUGMODE && DEBUG_NET_PRINTJSONMESGS, "Valid reply from node:  %s ", responseBuffer.c_str());
			return responseBuffer;  // return string (no error)
		}
	}
	else
	{  // non-OK return code from LibCURL operation:
		const std::string errStr = std::string(curl_easy_strerror(curlResult));
		domesg_verb("LibCurl error: " + errStr, true, V_NORM);
		LOG_F(WARNING, "Libcurl error: %s", errStr.c_str());
		// ... some error stats handling code was here (fixme/todo?)
		return "Error: Network error: " + errStr;
	}

	return "Error: end of function";  // shouldn't reach here.
}

void printbytes_c(const unsigned char inArray[], const size_t len, const char* desc)  /* uint8_t* inArray? also, replace `desc` with a std::string, compile as C++. */
{  // PRINTBYTES_C: Prints a formatted table of bytes `len` long, 8 bytes/row, to stdout. C version.
	if (strlen(desc) >= 1) /*if (desc != "") <- */
		printf("%s  -  %zu bytes (big-endian order): \n", desc, len);  // description first. <-- use std::string
	uint8_t lnbrk_ctr = 0;
	for (uint8_t i = 0; i < len; ++i) {		// print a formatted table of bytes
		++lnbrk_ctr;
		printf("%02x ", inArray[i]);		// print w/ leading zero as needed <-- cast?
		//
		if (lnbrk_ctr >= 8) {
			printf("\n");					// line break every 8 bytes for readability.
			lnbrk_ctr = 0; }
	}
	printf("\n");
}

// === c style function, convert to C++ as needed  [todo] ===
void printbytes_c_v2(const unsigned char* input_bytes, const size_t bytes_length, const std::string desc)
{ // print an unsigned char array that is `length` bytes wide, in a neat table
	unsigned short linebrk = 0;  // line break counter
	printf("\nPrinting bytes of:  %s, length %zu \n", desc.c_str(), bytes_length);
	for (unsigned short i = 0; i < bytes_length; ++i)
	{
		printf("%02x ", input_bytes[i]);    /* print hex string repr'n of byte with leading zero. */
		++linebrk;
		if (linebrk > 7) {                  /* line break every 8 bytes */
			printf("\n");                   /* new line                 */
			linebrk = 0; }                  /* reset                    */
	}
	printf("\n");  // before and after
}


// ==========================

//int wallet_ethereum_assemble_tx_initial( /*const*/ EthereumSignTx* msg, uint64_t* encoded_U64);	//net_rlp.cpp

// SIGNTX: Is passed an unsigned payload in std::string form and returns a concatenated signature std::string with
//		   `r`, `s` and `v` components. ECDSA signs the keccak256() hash of the payload using the SECP256k1 curve.
//
//EcdsaResult SignTx ( secp256k1_context* ctx_both, EthereumSignTx tx, EthereumSig* sig_out, const unsigned char* noncedata_bytes, const size_t nonce_len, const unsigned char sk_bytes[32])
// DEBUG: checking inputs
constexpr auto PUBKEY_LENGTH = 64;
constexpr auto PRIVATE_KEY_LENGTH = 32;		// in bytes

EcdsaResult SignTx(secp256k1_context* ctx_both, /*const*/ EthereumSignTx tx, EthereumSig* sig_out, const unsigned char* noncedata_bytes, const size_t nonce_len)
{ /*... , const unsigned char* sk_bytes, const size_t sk_len) */ /* careful about the length!  fixed-length array arguments like unsigned char nonce[32] legal? */
	print_bytes_uchar(noncedata_bytes, nonce_len, "ecdsa sign nonce");											// <-- remove
	//print_bytes_uchar(sk_bytes, sk_len, "secret key (REMOVE THIS)");											// <--
	//print_bytes_uchar(tx.data_initial_chunk.bytes, tx.data_initial_chunk.size, "tx data initial chunk");		// <--

	secp256k1_ecdsa_recoverable_signature recoverable_sig{};
	secp256k1_ecdsa_signature nonrecoverable_sig{};  // "regular" signature
	unsigned char serialized_recovered_pubkey[65] = { 0 };  // check length. 64? <--  //unsigned char* serialized_pubkey ?

//	=== hash the payload: ===
// This doesn't look right. Hash the transaction bytes, RLP encoded, and the placeholder bytes for V, R, S in that order.
// V will be the network ChainID, R and S will be RLP encoding of a null byte (0x80). Like so, where ChainID=1:	`0x018080`.
// R and S will replace the null bytes, and be preceded by their length prefix, after tx hash is signed.
// update:	Now it works.
	uint64_t rawTx_U64[24]{};
	int pyld_byteslen = -1;
	pyld_byteslen = wallet_ethereum_assemble_tx_initial(&tx, rawTx_U64);	// pass the structure containing the Tx so far, most fields populated (signature not.)

	if (pyld_byteslen < INITIAL_PAYLOAD_MIN_LENGTH || pyld_byteslen > INITIAL_PAYLOAD_MAX_LENGTH) {												/* normally 100+ byte length */
		LOG_F(ERROR, "Bad initial tx length:  %d bytes", pyld_byteslen);
		return EcdsaResult::ErrBadPayloadLen;
	}
	
print_bytes(reinterpret_cast<uint8_t*>(rawTx_U64), pyld_byteslen, "initial tx as bytes");	// (Debug Only, Remove This)
print_bytes(reinterpret_cast<const uint8_t*>(rawTx_U64), pyld_byteslen, "Hashing Tx");		// <-- was: data_initial_chunk. we need the whole initial tx (no signature).
	
	//const size_t pyld_bytelen = ...
	LOG_IF_F(INFO, DEBUGMODE, "Allocating %d bytes for Tx", pyld_byteslen);  // <- DBG
	unsigned char pyld_hash[32]{};     // [Keccak256::HASH_LEN]
	uint8_t* pyld_bytes = (uint8_t*)(malloc(pyld_byteslen));			//// <----- This pointer must be freed !!!	 AW, FUCK. JUST EDIT THE END OF THIS ONE. MAKE IT BE 2 BYTES LONGER IF NEEDED. <----!!
	if (!pyld_bytes) {
		LOG_F(ERROR, "failed to allocate for tx!");
		return EcdsaResult::ErrBadPointer;
	}// <-

	//... now replace the last 2 bytes with 0x80 (null for R, S.) <--- [WIP]
	//if (pyld_byteslen-2 > ...... )
	printf("Copying %d bytes \n", pyld_byteslen);  // <- DBG
	memcpy( pyld_bytes, rawTx_U64, pyld_byteslen-2 ); // <- WIP: the 2 final bytes are there but not correct value. skip copying those.
														// warning LNT1000 <--
	pyld_bytes[pyld_byteslen - 1] = 0x80; //<--
	pyld_bytes[pyld_byteslen - 2] = 0x80; //<--
	print_bytes(pyld_bytes, pyld_byteslen, "payload with signature placeholders");
	// ^ condense this, break up into sub-functions if needed.
	
	const bool hash_success = Keccak256::getHash(pyld_bytes, static_cast<size_t>(pyld_byteslen), pyld_hash);	// out to payload_hash
	free(pyld_bytes);   // impt!
	if (!hash_success) { return EcdsaResult::ErrHashFailed; }

	print_bytes(pyld_hash, Keccak256::HASH_LEN, "payload hash");  //debug

//	=== get context here? ===
	if (!ctx_both)
		return EcdsaResult::ErrContextCreation;         // check that context is valid

//	=== get private key as byte array ===
	unsigned char skey[PRIVATE_KEY_LENGTH]{};			// secret key
	gSk256.getBigEndianBytes(skey);	// global uint256 to local byte-array
	//	skey = GetPrivateKey_Bytes(skey)? [todo]
	if (gVerbosity == V_DEBUG) {
		print_bytes(skey, static_cast<uint8_t>(PRIVATE_KEY_LENGTH), "private key");  // <- Compare. Make sure these are the same :)
		printbytes_c_v2(skey, PRIVATE_KEY_LENGTH, "skey");  // <- COULD PRINT INVALID BYTES? CHECK THIS
	}
	// === verify private key ===
	if (gVerbosity == V_DEBUG) { printf("Verifying private key...\n"); }
	if (!secp256k1_ec_seckey_verify(ctx_both, skey))	/* <--- pubkey w/ pointer to secp256k1_pubkey,				*/
		return EcdsaResult::ErrSKeyVerify;				/* ... or &pubkey to a regular object? <- same for skey?	*/

	// === create public key from private ===
	if (gVerbosity == V_DEBUG) { printf("Computing public key...\n"); }
	secp256k1_pubkey pubkey{};
	if (!secp256k1_ec_pubkey_create(ctx_both, &pubkey, skey))
		return EcdsaResult::ErrPubKeyGenerate;  //err

		// ===  debug stuff: ===
	printbytes_c(pubkey.data, 64, "pubkey");  // <- COULD PRINT INVALID BYTES? CHECK THIS <-

	// === serialize pubkey (important!) ===
	printf("Serializing public key...\n");
	unsigned char serialized_pubkey[65]{};		// check length. 64? <--  //unsigned char* serialized_pubkey ?  [MOVEME]?
	size_t ser_pubkey_len{ 65 };  // <- written into? <- fixme <--
	//
	// write `ser_pubkey_len` (65) bytes to uchar array `serialized_pubkey`:
	if (!secp256k1_ec_pubkey_serialize(ctx_both, serialized_pubkey, &ser_pubkey_len, &pubkey, SECP256K1_EC_UNCOMPRESSED))
		return EcdsaResult::ErrPubKeySerialize;

	// === debug stuff: print uncompressed key ===
	printf("Uncompressed public key:	");
	for (unsigned short i = 0; i < 65; ++i) { printf("%.2x", (unsigned int)serialized_pubkey[i]); } // <- [REF]

	// === sign msg ===
	printf("\nSigning message hash... \n"); /* with some arbitrary data for the nonce function. can also use NULL, default nonce func should be rfc6979. */
	if (!secp256k1_ecdsa_sign_recoverable(ctx_both, &recoverable_sig, pyld_hash, skey, secp256k1_nonce_function_rfc6979, noncedata_bytes))	/* <- [TESTME]: send in nonce data to nonce function? unsigned char[32] array? */
		return EcdsaResult::ErrSignRecoverable;																								/*<- todo/wip. also: default or rfc6979? */

	// === convert from recoverable to regular sig before verifying === 
	printf("Converting signature... \n");
	if (!secp256k1_ecdsa_recoverable_signature_convert(ctx_both, &nonrecoverable_sig, &recoverable_sig))  // find the garbage prototype of this and remove!! <-- FIXME
		return EcdsaResult::ErrConvertSig;

	// === verify signature ===
	printf("Verifying signature... \n");
	if (!secp256k1_ecdsa_verify(ctx_both, &nonrecoverable_sig, pyld_hash, &pubkey))  /* alternatively a context with _VERIFY instead. Pretty sure we want secp256k1's `pubkey` type here and not the trimmed serialized ver. */
		return EcdsaResult::ErrSigVerify;

	// === serialize signature, get recovery id (for computing the transaction's `v` field) ===
	printf("Serializing recoverable signature & getting recovery id... \n");
	unsigned char compact_sig_bytes[64]{ 0 };		  // compact signature from serialization. needed?
	int rec_id{ -1 };	 // fingers crossed.  (for making `v`)
	if (!secp256k1_ecdsa_recoverable_signature_serialize_compact(ctx_both, compact_sig_bytes, &rec_id, &recoverable_sig))  /* <--- this? get the recovery id (valid range: 0-3). */
		return EcdsaResult::ErrSigSerialize;
	// - dbg - 
	printbytes_c_v2(compact_sig_bytes, 64, "compact signature"); // <- ENSURE ACCURATE OUTPUT OF BYTES TO TABLE, WIP/FIXME
	printf("recovery id:  %d \n", rec_id); // <--
	printf("\n** Compare against.. ** \n");
	printbytes_c_v2(recoverable_sig.data, 64, "first 64 bytes of recoverable_sig");
	// - dbg -

	// === recover public key from the new signature, and the message hash ===
	printf("Recovering pubkey from signature ... \n");
	secp256k1_pubkey recovered_pubkey{}; // <-- {}?
	if (!secp256k1_ecdsa_recover(ctx_both, &recovered_pubkey, &recoverable_sig, pyld_hash))
		return EcdsaResult::ErrRecover;

	printbytes_c_v2(recovered_pubkey.data, 64, "recovered public key"); // <--
	LOG_IF_F(INFO, gVerbosity == V_DEBUG, "SignTx():	Serializing public key "); // <- remove
	size_t pubkey_outlen = 65; // <- remove? written into by function call below?
	if (!secp256k1_ec_pubkey_serialize(ctx_both, serialized_recovered_pubkey, &pubkey_outlen, &recovered_pubkey, SECP256K1_EC_UNCOMPRESSED))
		return EcdsaResult::ErrPubKeySerialize;

	// TEST: make sure the last 20 bytes of the keccak256 hash of the _recovered_ public key match the Ethereum address
	//       we're mining/minting with. [todo]: do this in a common helper function. <--
	LOG_IF_F(INFO, gVerbosity == V_DEBUG, "SignTx():	Hashing public key "); // <- remove
	uint8_t pubkey_hash[Keccak256::HASH_LEN]{ 0 };
	Keccak256::getHash(&serialized_recovered_pubkey[1], 64, pubkey_hash);  // <- 64: public key length in bytes. skipping first byte ...
	printbytes_c_v2(pubkey_hash, Keccak256::HASH_LEN, "Keccak256 Hash of Public Key");

	print_bytes(compact_sig_bytes, SIGNATURE_LENGTH, "compact signature");  // <- unsigned char ver? cleaner.
	// [WIP] ^

	memcpy(sig_out->signature_r.bytes, reinterpret_cast<pb_byte_t*>(compact_sig_bytes), 32);		// copy 32 bytes
	memcpy(sig_out->signature_s.bytes, reinterpret_cast<pb_byte_t*>(&compact_sig_bytes[32]), 32);	// "
	sig_out->signature_r.size = static_cast<pb_size_t>(32);  // bytes not characters <--- [wip]
	sig_out->signature_s.size = static_cast<pb_size_t>(32);  //	"
	
	print_bytes(reinterpret_cast<uint8_t*>(sig_out->signature_r.bytes), 32, "signature: `r`");	// <-
	print_bytes(reinterpret_cast<uint8_t*>(sig_out->signature_s.bytes), 32, "signature: `s`");  // <-
	// overwrite these too!! ^^
	LOG_IF_F(INFO, gVerbosity == V_DEBUG, "SignTx():  Cleaning up");
	randombytes_buf(compact_sig_bytes, 64);						//
	randombytes_buf(serialized_recovered_pubkey, 65);				//
	randombytes_buf(pyld_hash, 32);                               //<-- ... anything else? Ensure nonce overwritten in calling func! <-
	// - ...				^^^^ CONDENSE THIS- remove any unneeded tests. Make sure we're hashing and signing the accurate initial payload ('s hash)!! ^^^

	//EthereumSig sig_TEST{}; // <- remove. comparing 2 casting approaches...
	//memcpy(sig_TEST.signature_r.bytes, reinterpret_cast<EthereumTxRequest_signature_r_t*>(compact_sig_bytes), 32);		 // copy 32 bytes (r)	//<---   reinterpret_cast<uint_least8_t*> ?
	//memcpy(sig_TEST.signature_s.bytes, reinterpret_cast<EthereumTxRequest_signature_s_t*>(&compact_sig_bytes[32]), 32);  // copy 32 bytes (s)	//<--- set size?

	sig_out->signature_v = static_cast<uint32_t>((gSolo_ChainID * 2) + 35 + rec_id);		 // `v` byte  <--- sanity check?
	// enforce a max value for chainid? should be unsigned short
	// verify recid/v are valid values! [todo/fixme] <----	
	print_bytes(sig_out->signature_r.bytes, sig_out->signature_r.size, "signature `r` bytes");  // <-- DBG
	print_bytes(sig_out->signature_s.bytes, sig_out->signature_s.size, "signature `s` bytes");	// <-- DBG
	printf("rec_id = %d		v = %" PRIx32 " \n", rec_id, sig_out->signature_v); // <--
	//
	return EcdsaResult::OK;  // no error - return the signed tx as a std::string. (0x? no 0x?)
}


//#include "coredefs.hpp"	// [MOVEME] <----
//#include <mutex>

// [TODO]: use better container/OOP approach for the TxView's items.
// Protect TxViewItems[] from simultaneous asynchronous access by the main/network threads. [FIXME] <--
// void Solo_CheckStalesInTxView (const miningParameters *params) {}


//json11::Json::object foo()
//{ // debug use only!
//	json11::Json j_foo = json11::Json::object({
//		{ "jsonrpc", "2.0" },
//		{ "method", "submitShare" },
//		{ "params", json11::Json::array({
//				p_soln->solution_string,
//				p_soln->devshare ? gStr_DonateEthAddress : params->mineraddress_str,
//				p_soln->digest_string,
//				double(p_soln->poolDifficulty),
//				p_soln->challenge_string
//				})
//		},
//		{ "id", "cosmic" }
//		});
//
//	return j_foo;
//} // debug use only!


System::Numerics::BigInteger Solo_GetMiningReward(const std::string sContractAddr) /* <--- CHECK FOR 0X <---- [WIP] ! */
{ // gets the current Mining Reward (amount in tokens) from the selected token contract
  // [REF]: from hash from keccak256("getMiningReward()" : 0x490203a7...
	System::Numerics::BigInteger reward_bigint{ BigInteger::Zero };
	bool success{ false };

	if (DEBUGMODE) { printf("contract addr: %s \neth addr: %s \n", gStr_ContractAddress.c_str(), gStr_SoloEthAddress.c_str()); } // <-- dbg
	if (!checkString(sContractAddr, 42, true, true) || !checkString(gStr_SoloEthAddress, 42, true, true)) {	/* redundant?				 */
		LOG_F(ERROR, "Bad contract address or Solo mining address in Solo_GetMiningReward()!");				/* should not happen. j.i.c. */
		return BigInteger::Zero;	//NOTE: returning a reward amount of `0` is an error to calling function.
	}

	LOG_IF_F(INFO, DEBUGMODE, "Retrieving current Mining Reward from Contract %s...", sContractAddr.c_str());

	//std::string str1 = "{\"jsonrpc\":\"2.0\", \"method\":\"eth_call\", \"params\": [{\"from\": ";
	//str1 = str1 + "\"" + gStr_SoloEthAddress + "\", \"to\" : \"" + gStr_ContractAddress + "\" , \"data\" : \"";
	//str1 = str1 + "\" }, \"latest\"], \"id\" : \"cosmic_rewardamt\" }";	// TODO: do this w/ JSON object
	const std::string str_data = "0x490203a7000000000000000000000000";  // padded w/ zeroes (?) [FIXME] <----
	json11::Json j_request_getminingreward = json11::Json::object({
		{ "jsonrpc", "2.0" },
		{ "method", "eth_call" },
		{ "params", json11::Json::array({
			json11::Json::object({ { "to", gStr_ContractAddress}, { "data", str_data }, }), "latest"
			}) ,
			//"latest"
		/*})*/ },
		//})
		//},
		//{ "latest" },
		{ "id", "cosmic" }		/* do we need TO, FROM fields too? */
	} );

	//		{ "to", "contract address" },
	//		{ "from", MINER's ADDRESS STRING },		...needed?

	// [REF]:	getting the challenge from the user-specified contract, using balanceOf(address tokenOwner) method (0x4ef37628).
	//std::string str_request = "{\"jsonrpc\":\"2.0\", \"method\":\"eth_call\", \"params\": [{\"from\": ";
	//str_request += "\"" + ethAddress + "\", \"to\" : \"" + gStr_ContractAddress + "\" , \"data\" : \"";
	//str_request += str_data + "\" }, \"latest\"], \"id\" : \"cosmic_tokenbal\" }";  // TODO: use Json object. this is slow.
	const std::string j_request_str = j_request_getminingreward.dump(); ////<--- [new]: using json11 (REMOVE THIS) <--- debug <--

// WIP / FIXME:
	const std::string str_reply = Solo_JSON_Request(j_request_getminingreward.dump(), false);  // send `str_request` to node & get response.
	if (str_reply.empty()) {	//<--- appropriate here? <--- [wip] !
		LOG_IF_F(WARNING, gVerbosity >= V_MORE, "Error: didn't get TRANSACTION COUNT? <---- mining reward (empty reply). ");
		return false;  // err
	}
	// PASTED [FIXME] ^
	//

	// note to self/fixme: use the new helper funcs to process the response, do more(standardized) checks this way, like for
	// length/hex-only characters. ensure everything's checked OK for response from helper func in Solo_JSON_Request() if I haven't already <--

	// condense this:
// parse `result` key <--- (TODO: use ParseKeyFromJson()) ! <--
	const std::string str_reward = ParseKeyFromJsonStr(str_reply, "result", 66, TXVIEW_SOLUTION_NULL, &success);	// expect 66 characters, 0x+64 hex digits. hex check needed:
	if (!success) {
		LOG_F(ERROR, "Bad `result` getting token reward amount from contract!");	// shouldn't reasonably happen, j.i.c.
		LOG_IF_F(ERROR, DEBUGMODE, "the result was:  %s ", str_reward.c_str());
		return BigInteger::Zero;	//err: calling func should check for 0.
	}

 // remove need for this ugly check by adding `expectHex`,
//	if (str_reward[0] != '0' || str_reward[1] != 'x' || str_reward.find_first_not_of(DEF_HEXCHARS, 2, 64) != std::string::npos) {
	if (str_reward[0] != '0' || str_reward[1] != 'x' || str_reward.substr(2).find_first_not_of(DEF_HEXCHARS) != std::string::npos) {
		LOG_IF_F(WARNING, DEBUGMODE, "Bad reward str after parse:	%s ", str_reward.c_str());
		return BigInteger::Zero;	//err
	}

	System::String^ mstr_reward = gcnew System::String(str_reward.substr(2).c_str());	// skip 0x
	if (gVerbosity == V_DEBUG) { printf("Retrieved token contract's reward amount: %s  with length %zu \n", str_reward.c_str(), str_reward.length()); }

	//const Uint256 reward_U256 = Uint256(str_reward.substr(2).c_str());
	IFormatProvider^ iFormat = gcnew Globalization::CultureInfo("en-us");
	//if (!Numerics::BigInteger::TryParse(mstr_reward, reward_bigint)) {	/* system string of 256-bit uint as hex, no `0x` */
	if (!Numerics::BigInteger::TryParse(mstr_reward, Globalization::NumberStyles::HexNumber/* | Globalization::NumberStyles::AllowHexSpecifier*/, iFormat, reward_bigint)) {	/* system string of 256-bit uint as hex, no `0x` */
		LOG_IF_F(ERROR, DEBUGMODE, "Couldn't parse token reward:	%s ", str_reward.c_str());
		return BigInteger::Zero;	//err: calling func should check for 0.
	}

	if (DEBUGMODE) { Console::WriteLine("contract's reward amount (parsed):	{0}	{1}", reward_bigint, reward_bigint.ToString()); }	//<----
	//if (reward_U256 > Uint256::ZERO && reward_U256 < Uint256::Uint256(uint256_max_bytes)) return reward_U256;
	//[wip]: more consistent biginteger/uint256 use. bigint only needed here for division in the calling function

	if (reward_bigint > System::Numerics::BigInteger::Zero)
		return reward_bigint;		// OK: the reward w/ 8 decimals
	 else return BigInteger::Zero;	// 0=err
}


// condensing this function [WIP].
unsigned short Solo_SendDonation(const Uint256 donation256)
{ // submit a solution from the queue, doing an equivalent of eth_sendrawtransaction() to assemble an RLP-encoded
  // transaction payload (as hex byte-string).
	uint64_t payload[24] {};					// consider using regular byte-array
	unsigned int payload_size {0};

	// === assemble transaction ===
	if (!AssembleTX(TXVIEW_SOLUTION_NULL, donation256, /*gStr_SoloEthAddress, */payload, &payload_size)) {
		LOG_F(ERROR, "Error assembling transaction. Please report this bug.");	//dbg
		return 1;
	}

	const std::string str_payloadbytes = uint8t_array_toHexString(reinterpret_cast<uint8_t*>(payload), payload_size); //<-- condensed

//	if (!checkErr_a(str_payloadbytes)) {
	if (static_cast<int>(str_payloadbytes.length()) != payload_size * 2) {  // not trimming off "Error: ", fixme?
		LOG_F(ERROR, "Solo_SendSolution(): code 1, err: expected %d characters, got %d. \n", static_cast<int>(str_payloadbytes.length()), payload_size * 2); //<-- oughtn't happen
		domesg_verb("Error while assembling Tx payload. Please report this bug.", true, V_NORM);  // should not occur.
		return 1;	// abort send
	} else {
		LOG_IF_F(INFO, DEBUGMODE, "RLP-encoded payload:	0x%s ", str_payloadbytes.c_str());	// payload to send
		print_bytes(reinterpret_cast<uint8_t*>(payload), payload_size, "RLP-encoded payload");	//<-- dbg only
	}

	// === sign, verify, and clean up ===
	// === Send raw transaction: ===
	bool sendrawtx_success{ false };
	const std::string result = SendRawTx("0x" + str_payloadbytes, RawTxType::Mint, TXVIEW_SOLUTION_NULL, &sendrawtx_success);  // <-- -1: no txView item# passed. we're only sending tokens.
	if (!checkErr_a(result)) { /* <-- use a bool for the success/fail? then treat the string as err if it's false.  <----------------*/
		LOG_IF_F(WARNING, gVerbosity >= V_NORM, "- SendRawTx() error [donation]:  %s \n", result.c_str());
		//gTxViewItems[t].txHash.clear();  // redundant
		//gTxViewItems[t].errString = result;
		return 1;	//err
	}

	// === Check response to SendRawTx(): ===
	// expecting 64 hex characters prepended by `0x` (txhash). otherwise: treat as error.   [TODO/FIXME]: put the parsed err mesg/code from geth/etc.!! <--
	if (checkString(result, 66, true, true))
	{ //txhash received
		LOG_IF_F(INFO, HIGHVERBOSITY, "Donation Tx hash:	%s ", result.c_str());	//<-- was: V_MORE. Should now log all TxHashes
		//gTxViewItems[t].txHash = result;	// store for receipt requests, UI display and/or logging. <--- [MOVEME]?
		//gTxViewItems[t].errString.clear();  // redundant
// iterate solutions count now? also display UI event. <-- [WIP]
// UI event						<--
		return 0;	//ok
	} else { //store error string if no txhash.		[TODO]: verify parsing functon parses the error name/code from geth/etc. [WIP] <--
		//xViewItems[t].errString = result;													// Leftover  <-- [WIP]: ui event for this func's completion
		domesg_verb("Error submitting donation:	" + result, true, V_MORE);					// >=V_MORE ?
		LOG_IF_F(INFO, HIGHVERBOSITY, "Solution %d: no tx hash, error:  %s ", -1/*t*/, result.c_str());
		return 1;  //err code
	} // ^ [WIP] ^
	// gTxViewItems[txViewItemNum].txHash = result;
}



//
const unsigned char maxval_bytes[32] = { // MOVEME
							0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 
							0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 
							0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 
							0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE };
//
void Solo_DonationCheck(void)
{ // checks if any tokens are supposed to be donated, does so as needed.  
  // should be called at a reasonable interval (every 5-10 minutes?)
  // - one donation = the auto-donation % (of) one mining reward. or they are consolidated (>1 waiting). <-- [WIP]
	if (!gDonationsLeft) { return; }	// no donations waiting

// kludgy use of BigInteger here. prefer uint256 where possible	for neater/more portable code [todo / fixme] <---[WIP]
	LOG_IF_F(INFO, DEBUGMODE, "Donations waiting: %d ", gDonationsLeft);

// slightly wacky sanity check. converting BigInteger to smaller type Uint256, ensure it will fit.
//	const BigInteger MaxVal = BigInteger::Parse("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFE",
//		System::Globalization::NumberStyles::HexNumber | Globalization::NumberStyles::AllowHexSpecifier);
	BigInteger reward = Solo_GetMiningReward(gStr_ContractAddress);  // get the current token reward (see net_solo.cpp)

	if (reward < BigInteger::One) {		/* || reward >= MAX_U256) */
		LOG_IF_F(ERROR, DEBUGMODE, "Solo_DonationCheck(): bad token reward amount");
		return;
	}

	BigInteger donation = reward / 50;				// for a 2% donation per solution minted
	/*const BigInteger donation = reward / 67; */	// for slightly less than 1.5% donation
	if (gDonationsLeft > 1)
		donation *= gDonationsLeft;				// consolidate rewards [TESTME] 
												/* for if a previous donation call was unsuccessful.		*/
												/*	[TODO]: set a threshold, send X consolidated donations 	*/
												/*			 as a single transfer() call when possible		*/

//	Console::WriteLine("Contract mining reward:	{0} satoastis", reward.ToString());
//	Console::WriteLine("Auto-Donation amount:	{0} satoastis	(8 decimals)", donation.ToString()); }
	if (donation < BigInteger::One) {  /* quick sanity check */
		LOG_IF_F(WARNING, HIGHVERBOSITY, "Error performing auto-donation: unexpected donation amount of <1 satoasti");
		return /*0*/;
	}	// [todo]: see its calling function

	msclr::interop::marshal_context mctx;
	const Uint256 donationU256 = Uint256( mctx.marshal_as<std::string>(donation.ToString("X2")).c_str() );		//<-- Sanity check: should be 32 hex bytes [todo]
//
	if (DEBUGMODE) {
		printf("Donation amount:	");
		print_Uint256(donationU256, true, true);	//verify proper operation!! this is the amount that will be sent <------ [WIP]
	}
// -- send donation. if no error, decrement # of donations waiting, if any. RESET TO ZERO if sending consolidated donations as 1 tx --
	if (!Solo_SendDonation(donationU256)) {
		gDonationsLeft = 0;						//<-- WIP: ensure this is not accessed by network BGworker simultaneously. <-- mtx_soloparams ?

//		if (gDonationsLeft <= 1) { gDonationsLeft = 0; }
//		 else { --gDonationsLeft; }
	}
	
	// [WIP]: update the Reward each time the diff changes? this should be rare.
	//		  try to save on # of API calls. it should only change if a reward halving has occurred.
}



// new version. Last 2 params are Outputs (retrieved/parsed transaction count as hex bytes in string representation ( w/out `0x` ) and as 64-bit unsigned int
// Returns `true` if successful, `false` on any error fatal to the operation. If `false`, outputs are not necessarily valid. *Should* only fail on Network Error.
// In such an event, the calling function must react appropriately  (i.e. return err to retry later, use the already-stored gSolo_TxnCount value, etc.)
bool Eth_GetTxCount(const std::string& for_address, const int txview_itemno, Uint256* out_txcount, std::string& out_txcount_hexstr)
{ // [TODO / WIP]: out_txcount is not changed, use the hex string
  // was: Eth_GetTransactionCount_New()
	if (!checkString(for_address, 42, true, true)) { /* expect 0x + 40 hex digits */
		LOG_IF_F(ERROR, gVerbosity == V_DEBUG, "Error: Eth_GetTransactionCount got bad eth address ");
		return false; }

	LOG_IF_F(INFO, gVerbosity == V_DEBUG, "Getting transaction count for address %s... \n", for_address.c_str());
	json11::Json j_request_gettransactioncount = json11::Json::object({
		{ "jsonrpc", "2.0" },
		{ "method", "eth_getTransactionCount" },
		{ "params", json11::Json::array({ for_address, "latest" }) },	/* [note]:  not a key pair- just param 0, the miner's eth address */
		{ "id", "cosmic" }
	});

	const std::string str_reply = Solo_JSON_Request(j_request_gettransactioncount.dump(), false);  // send `str_request` to node & get response.
	if (str_reply.empty()) {
		LOG_IF_F(WARNING, gVerbosity >= V_MORE, "Error: didn't get transaction count (empty reply). ");
		return false;  // err
	}

	// [todo]: useful return type from Solo_JSON_Request would remove need for this check:
	if (!checkErr_a(Solo_JSON_Request(j_request_gettransactioncount.dump(), false))) {
		// err string returned by Solo_JSON_Request(). Probably network error:
		LOG_IF_F(WARNING, gVerbosity >= V_MORE, "Retrieving transaction count:  %s ", str_reply.c_str());  // the network error, etc. from Solo_JSON_Request. [TODO]: use a pointer-to-bool `success` param.
		return false;  // err
	}

	bool parse_success{ false };
	//	if (txview_itemno != -2) { .dump()... } else { .string_value() }  //<- use enum or logically named param.
	std::string str_txncount = ParseKeyFromJsonStr(str_reply, "result", 0, txview_itemno, &parse_success); //no expected length
	if (!parse_success || !checkErr_a(str_txncount)) {
		LOG_IF_F(WARNING, gVerbosity >= V_MORE, "Parsing Error:  %s ", str_txncount.c_str());  //print/log the parsing error
		return false;  //err
	}
//
	if (DEBUGMODE) { printf("str_txncount:  %s \n", str_txncount.c_str()); }	// <--- REMOVE
	if (str_txncount.length() >= 2) {
		if (str_txncount[0] == '0' && str_txncount[1] == 'x')
			str_txncount = str_txncount.substr(2);  //trim `0x` if present.
	}// else {
	 // LOG_IF_F(ERROR, gVerbosity >= V_MORE, "Transaction count hex-string is too short");
	 //   return false; }

	// check evenness, pad w/ zero if needed:
	if (str_txncount.length() % 2 != 0) {
		str_txncount = "0" + str_txncount;  // no 0x (just bytes).
		LOG_IF_F(WARNING, gVerbosity == V_DEBUG, "Uneven length transaction count hexstr- padding with 0");  // <- remove, debug only
	} //else { /* already even length */ }

	out_txcount_hexstr = str_txncount;
	// write value to uint256:  `out_txcount`
	// [todo]^

	if (gVerbosity == V_DEBUG) printf("* str_txncount: %s \n" "* hexstr_txcount: %s \n", str_txncount.c_str(), out_txcount_hexstr.c_str());  // <--- remove
	//
	// `result` string value (hex) cannot be longer than unsigned 256-bit ( 0x + 64 hex digits/32 bytes = 66 characters )
	if (str_txncount.length() < 2 || str_txncount.length() > 64) {
		LOG_IF_F(WARNING, gVerbosity >= V_MORE, "Error parsing transaction count:  Result value has bad length %zu.", str_txncount.length());
		return false;  //err
	}

// ^ Condense this. [WIP]
// the adjusted std::string of hex bytes, with leading zero if needed.  no `0x` specifier.

// === while we're at it, convert the txcount (as hexstr) to a uint, compare to global, update as needed. ===
//    auto txcount256 = Uint256_MaxValue;
//    txcount256 = Uint256( hexstr_txcount.c_str() );  // parse bytes (hex string w/out `0x`) to Uint256.  <- local var!
//    if (txcount256 == Uint256_MaxValue) {
//        LOG_IF_F( ERROR, gVerbosity==V_DEBUG, "Failed to parse txcount to uint256!" ); // <--
//        return false;
//    }
	return true;  // OK
}


GetParmRslt UpdateTxCount(const std::string& for_address, std::string& hex_txcount_out)
{ // retrieve the transaction count for the specified Ethereum account, and update the global value if needed
    LOG_IF_F(INFO, gVerbosity >= V_MORE, "Checking for updated Tx count for account %s.", gStr_SoloEthAddress.c_str());

    //  int txview_itemno{ INT_MAX };				//
    std::string txncount_hexstr{};				// function-local only
	Uint256 txcount_U256{ Uint256::ZERO };		//<--- [WIP]: not used yet

    if (!Eth_GetTxCount(for_address, INT_MAX, &txcount_U256, txncount_hexstr)) {
        LOG_IF_F(WARNING, gVerbosity >= V_MORE, "UpdateTxCount():  failed to retrieve Txn count. Network error?");  // network error? <----
        return GetParmRslt::GETPARAM_ERROR;
	} else
		LOG_IF_F(INFO, gVerbosity >= V_MORE, "Got Tx Count:  %s ", txncount_hexstr.c_str());	// got tx count as string of bytes
																								// <-- should be up-to-date now.  [TESTME]
// === compare txcount hex-string to global, update as needed. ===
	hex_txcount_out = txncount_hexstr;     // for calling function

    if (txncount_hexstr == gTxCount_hex) {
        LOG_IF_F(INFO, DEBUGMODE && DEBUG_NETWORK, "UpdateTxCount():  txn count unchanged  (OK)");  // <- dbg only
        return GetParmRslt::OK_PARAM_UNCHANGED;
    } else { // transaction count is new
        gTxCount_hex = txncount_hexstr;
        //gU256_TxnCount = *txncount_u256;
        LOG_IF_F(INFO, gVerbosity == V_DEBUG, "UpdateTxCount():  new txncount 0x%s ", gTxCount_hex.c_str());  // <--- remove. dbg only
        return GetParmRslt::OK_PARAM_CHANGED;
    }

}


std::string GetAccountBalance_Ether(std::string ethAddress)
{ // Requests Ether balance for address `ethAddress` from API endpoint, parses the reply, returns balance formatted for output
  // The last (least-sigificant) decimal place of the Ether balance is not rounded. (FIXME)
	domesg_verb("Requesting Ether balance for account " + ethAddress + "...", false, V_DEBUG);
	
	// request json in convenient puddle form:
	const std::string str_request = "{\"jsonrpc\":\"2.0\",\"method\":\"eth_getBalance\",\"params\":[\"" + ethAddress + "\",\"latest\"],\"id\":\"cosmic_ethbal\"}";

	msclr::interop::marshal_context marshalctx;  // for marshaling managed/native types
	BigInteger big_wei{ BigInteger::Zero }, remainder{ BigInteger::Zero };
	Decimal dec_gwei{ Decimal::Zero };
	std::string str_result{ "" };
	bool parse_success{ false };
//	ethAddress = AddHexSpecifierIfMissing(ethAddress);	// todo:  use isEthereumAddress() here and elsewhere

	str_result = Solo_JSON_Request(str_request, false);		// `s1`: request str, the result also goes in `s1`.
	if (!checkErr_a(str_result))  { return str_result; }	// return error string, if returned by Solo_JSON_Request() <- [CHECKME]

// get the key `result` from str_result (response from node), if present.
	str_result = ParseKeyFromJsonStr(str_result, "result", 0, -1, &parse_success);  //std::string str_ethbal = jsonReply["result"].dump();
	if (!parse_success) { /* see: error string check in ParseKeyFromJsonStr() */
		printf("GetAccountBalance_Ether(): returning error:  %s \n", str_result.c_str());  // <- remove this
		return "Error parsing result key from response: " + str_result; }		// see checkerr_b() & calling func.
	// no need to trim double-quotes.	... if false, `s1` should be an err string from ParseKeyFromJsonStr().	// nn
	bool bSuccess{ false };
	System::String^ mstr_ethbal_hexstr = gcnew System::String( std::string("0" + Check0x_StringVer(str_result, true, &bSuccess)).c_str() );	// <-- hexstr of eth balance in wei. Prepend '0' because input is unsigned,
//																										//	 so BigInteger::Parse() won't interpret the hex string input as a negative #
	if (!bSuccess) {
		return "Ether Balance:  ?";
	} // ^

// compute & format the ether balance for display:
	IFormatProvider^ iFormatProv = gcnew CultureInfo("en-US");
	if (!BigInteger::TryParse(mstr_ethbal_hexstr, NumberStyles::HexNumber | NumberStyles::AllowHexSpecifier, iFormatProv, big_wei)) { 
		domesg_verb("Parsing error while getting tokens balance. ", false, V_NORM);		// If parse failed...
		return "Ether Balance:  ?"; }													// BigInteger::TryParse() will not throw exceptions.
	// parse mstr to biginteger `big_wei` succeeded:
	//BigInteger big_onebillion = BigInteger(1000000000);
	dec_gwei = (System::Decimal)(BigInteger::DivRem(big_wei, BigInteger(1000000000), remainder));  // biginteger division (wei to gwei)  [TESTME]
	Decimal dec_ether = dec_gwei.Multiply(dec_gwei, (System::Decimal)0.000000001);  // gwei value * how many gwei
	// ^ [CHECKME]. try-catch? shouldn't throw. ^
	if (gVerbosity == V_DEBUG) { Console::WriteLine("dec_ether: " + dec_ether.ToString() + ", dec_gwei: " + dec_gwei.ToString() + ", remainder: " + remainder.ToString()); }
	return "Ether Balance: " + marshalctx.marshal_as<std::string>(dec_ether.ToString()) +
		marshalctx.marshal_as<std::string>(remainder.ToString()->Substring(0, 1));  // formatted string to calling func for display
}
// ^This function has a bug:  result is ? when getting balance on a TEST net like Ropsten. [FIXME] ^



std::string GetAccountBalance_Tokens(std::string ethAddress)
{  // GETACCOUNTBALANCE_TOKENS: Gets an Ethereum address (20 bytes/40 digit hexadecimal string, prepended with `0x`) and
   //							returns the balance expressed as a decimal string for display in UI. Can also debug-print
	msclr::interop::marshal_context marshalctx;
	System::Numerics::BigInteger big_tokenbalance{ BigInteger::Zero };
	std::string str_response{ "" };						// tokens balance
	bool parse_success{ false };
	ethAddress = AddHexSpecifierIfMissing(ethAddress);	// [redundant] ?

// function selector: 0x70a08231  from keccak256("balanceOf(address)"), first four bytes followed by the miner's address 
	//const std::string str_data = "0x70a08231" + gStr_SoloEthAddress.substr(2);
	const std::string str_data = "0x70a08231000000000000000000000000" + gStr_SoloEthAddress.substr(2);  // with padding zero bytes
	domesg_verb("Requesting Token Balance for account " + ethAddress, false, V_DEBUG);

	// getting the challenge from the user-specified contract, using balanceOf(address tokenOwner) method (0x4ef37628).
	std::string str_request = "{\"jsonrpc\":\"2.0\", \"method\":\"eth_call\", \"params\": [{\"from\": ";
	str_request += "\"" + ethAddress + "\", \"to\" : \"" + gStr_ContractAddress + "\" , \"data\" : \"";
	str_request += str_data + "\" }, \"latest\"], \"id\" : \"cosmic_tokenbal\" }";  // TODO: use Json object. this is slow.

// do these globally in Core? update as needed only?
// json11::Json j_balanceOf = json11::Json::object {
//		{ "jsonrpc", "2.0" },  ...
//	const std::string str_balanceOf_TEST = j_balanceOf.dump();  	// replace str_tokens above with j_balanceOf.dump( ... ); <--- WIP

//	FIXME: better checks (like length/hex-only characters. Beef these up in Solo_JSON_Request?) <----
//  [NOTE]:  infura sometimes returns 0x with nothing following it, check for this:
//
// calling token contract function to get tokens balance for the active Eth account:
	//
	if (gVerbosity == V_DEBUG) { printf("token balance request: %s \n", str_request.c_str()); }  // dbg, remove <-
	str_response = Solo_JSON_Request(str_request, false);	  // json request to node
	if (gVerbosity == V_DEBUG) { printf("token balance reply: %s \n", str_response.c_str()); }  // dbg, remove <-
	//
	if (!checkErr_a(str_response)) { return str_response; }  // return the error string, e.g. "Error: Bad JSON-RPC response". <-- [CHECK ME]
	str_response = ParseKeyFromJsonStr(str_response, "result", 66, -1, &parse_success);				// expecting unsigned 256bit value represented as 0x+32-byte hex str. "Quotes" around value are trimmed by
																								//		ParseKeyFromJsonStr(), which also checks for nested error message & code from node if key not found.
	if (!parse_success) { return "Tokens:  ?"; }   // err string from `str_tokens` helpful?
	if (str_response.length() != 66) {			   /* [redundant]: see ParseKeyFromJsonStr() param `expectedLength`. */
		LOG_F(WARNING, "GetAccountBalance_Tokens(): Unexpected result length %zu. Please check node", str_response.length());
		return "Token Balance:  ?";	}
// [TODO / FIXME]: ^ any other checks? ^
//
//  == Parse the node response (expecting hex bytes prepended by `0x` in key `result`) to BigInteger, format for display ==
	System::IFormatProvider^ iFormat = gcnew System::Globalization::CultureInfo("en-US");  // really necessary?
	//System::String^ mstr_tokenbalance = gcnew String(str_tokens.c_str());  // native -> managed str
	// adding 0 before the hex number so BigInteger::Parse doesn't treat it as a negative value (it's unsigned) <-
	System::String^ mstr_tokenbalance = gcnew String(str_response.substr(2).c_str());  // native -> managed str
	if (!BigInteger::TryParse("0" + mstr_tokenbalance, NumberStyles::AllowHexSpecifier | NumberStyles::HexNumber, iFormat, big_tokenbalance))
	{ return "Token Balance:  ?"; }  // if parse failed. (note: BigInteger::TryParse() does not throw exceptions :)
	
//	== Display and Debug stuff ==
	if (gVerbosity == V_DEBUG) { Console::WriteLine("Token Balance (satoastis):  " + big_tokenbalance.ToString("N0")); }  // dbg. w/ thousands place commas
	big_tokenbalance = big_tokenbalance / 100000000;				// biginteger version.  [alt]  big_tokenbalance = big_tokenbalance.Divide(100000000);
																	// shift left 8 decimal places (for 0xBTC). [TODO]:  get # of decimals from contract.
	mstr_tokenbalance = big_tokenbalance.ToString("N0");			// now with thousands separation commas/periods (bignum ver.) [TODO: locale-aware spacing?)
	if (gVerbosity == V_DEBUG) { Console::WriteLine("Token Balance:  " + big_tokenbalance.ToString("N0")); }  // dbg

//	== Return std::string with Tokens balance for GUI's status bar, with comma thousands separation for readability ==
	return "Token Balance: " + marshalctx.marshal_as<std::string>(mstr_tokenbalance);  // <- [todo] move this functionality to a func in CosmicWind.h, save a marshal/convert/etc.
}





// Very specialized helper function and much slower than the other conversion functions. Also something of an experiment.
// Provide function w/ a c-string (with a hex number), address of C byte-array to write to, and length in bytes expected (2 hex characters in str == 1byte)
// ... it will parse and, if possible, write the bytes in big-endian order to the address `outArray` (a pre-allocated buffer of length `byte_len`). Don't precede w/ "0x".
void HexStrToByteArray_b(const char* cstring, uint8_t* outArray, const unsigned int byte_len)
{
	String^ workStr = gcnew String( cstring );				   // instantiate managed string from "c-string" input
	IFormatProvider^ iFormat = gcnew Globalization::CultureInfo("en-us");
	BigInteger workBig = BigInteger( BigInteger::Zero );  // System::Numerics::BigInteger class. Initialize to 0.
	if (BigInteger::TryParse(workStr, NumberStyles::HexNumber | NumberStyles::AllowHexSpecifier, iFormat, workBig))  // &workBig?
	{ /* printf("Parse successful \n"); */ }	
	else {
		printf("Parse error \n");
		return;  }
	cli::array<byte>^ managedBytes = gcnew cli::array<byte>(byte_len);		// make our byte-array
	managedBytes = workBig.ToByteArray();   // out to cli::array `managedBytes` of byte_len length
	System::Array::Reverse(managedBytes);  // little to big endian: reverse the bytes' order
	for (unsigned int i = 0; i < byte_len; ++i)
		outArray[i] = (uint8_t)( managedBytes[i] );  // copy the big-endian bytes to native array

	//printf("HexStrToByteArray_b() test result: \n");
	//const unsigned short outArray_len = sizeof(outArray);  // in bytes
	//print_bytes(outArray, outArray_len);
	//
	// WIP: was write out to outArray successful? 
}

//
// DERIVEKEYFROMPASSPHRASE_WINSDK: Uses Windows SDK (Win7 and up) functionality to derive a key from a user-provided password
//												and a randomly-generated 256bit salt. TODO: make variable algorithms and iterations# user-selectable?
//
//												[FIXME / TODO]: replace 2nd and 3rd arguments with fixed-length arrays? <--

// [WIP / FIXME]
// was:  unsigned short DeriveKeyFromPassPhrase_WinSDK(const std::string& passPhrase, unsigned char* keyOut, unsigned char* saltOut, const bool newSalt);
unsigned short DeriveKeyFromPassPhrase_WinSDK(const /*unsigned*/ char *passPhrase/*_uchar*/, const size_t passPhraseLen, 
	unsigned char *keyOut, unsigned char *saltOut, const bool newSalt)												/* <--- */
{
	//const unsigned int passPhraseLen =
	//	uchar_passphrase;  //(unsigned int)( strlen(passPhrase.c_str() ));
	BCRYPT_ALG_HANDLE providerHandle;

	// use provided salt for derivation if false (for verification/decrypting)
	if (newSalt)
		randombytes_buf(saltOut, SALT_LENGTH);	// TODO: consider using MS crypto provider as CRNG here instead of libsodium
//
	print_bytes(saltOut, SALT_LENGTH, "salt");	// dbg
//
//BCRYPT_ALG_HANDLE_HMAC_FLAG or 0 for final arg: which is actually correct? 
//ref: BCRYPT_PBKDF2_ALGORITHM, BCRYPT_SHA512_ALGORITHM, BCRYPT_SHA512_ALGORITHM ...
	NTSTATUS NT_result{ 0 };
	NT_result = BCryptOpenAlgorithmProvider(&providerHandle, BCRYPT_SHA512_ALGORITHM, NULL, BCRYPT_ALG_HANDLE_HMAC_FLAG);  // last arg is reserved, must be 0.
	if (!BCRYPT_SUCCESS(NT_result)) {
		LOG_F( ERROR, "Error opening algorithm provider:  NTSTATUS 0x%" PRIx32 " ", (uint32_t)NT_result );
		return 1; }

	LOG_IF_F( INFO, gVerbosity > V_NORM, "Generating key from password." );
	//  Using SHA512 algo and PBKDF2_ITERATIONS (900k) iterations of the derivation func.  [TODO]: make configurable by user in "advanced" dropdown panel?
																		//   const size_t passPhraseLen = strlen( passPhrase.c_str() );
																		// ...
	NT_result = BCryptDeriveKeyPBKDF2(providerHandle, (PUCHAR)passPhrase/*_uchar*/, (ULONG)passPhraseLen, (PUCHAR)saltOut,			/* <---- */
		(ULONG)SALT_LENGTH, (ULONGLONG)PBKDF2_ITERATIONS, (PUCHAR)keyOut, DERIVED_KEY_LENGTH, 0);								/* <---- */

	if (!BCRYPT_SUCCESS(NT_result)) {
		LOG_F( WARNING, "Error deriving key:  NTSTATUS 0x%" PRIx32 " ", (uint32_t)NT_result );
		return 1; }

	// close the provider! (WIP)
	NT_result = BCryptCloseAlgorithmProvider(providerHandle, 0);
	if ( !BCRYPT_SUCCESS(NT_result) )
		LOG_F( WARNING, "Error closing algorithm provider:  NTSTATUS 0x%" PRIx32 " ", (uint32_t)NT_result );
	
	// overwrite the passphrase, salt, etc. in the _calling function_. <-- wip
	print_bytes((uint8_t*)keyOut, 32, "OK - key from password");
	return 0;  // no error
}


static void mod_illegal_callback_fn(const char* str, void* data) {
	(void)data;
	LOG_F(WARNING, "[libsecp256k1]  illegal argument ");  // : %s ", str);
//	abort();
}

static void mod_error_callback_fn(const char* str, void* data) {
	(void)data;
	LOG_F(ERROR, "[libsecp256k1]  internal consistency check failed ");  // : %s \n", str);
//	abort();  // <---- [todo] stop mining? should never happen.
}




// [WIP]
/*	Inputs: private_key - The private key to compute the public key for
	Outputs: public_key - Will be filled in with the corresponding public key	*/
//
constexpr auto UNCOMPRESSED_PUBKEY_LENGTH = 65;

bool getAddressFromSKey (secp256k1_context *ctx_both, unsigned char *skey_bytes, std::string& out_ethaddress)
{ // Compute the corresponding public key for a private key. Returns: 1=if key was computed successfully, 0=error occurred
	//uint8_t pubkey_u8arry[PUBLIC_KEY_BYTESLEN] {};	 // 64 bytes long for public key		 <--- unsigned char array?
	uint8_t pubkey_hash[HASH_LENGTH] {};			 // 32 bytes long for keccak256 hash  <--- unsigned char array?
	secp256k1_pubkey secp256k1_public_key;			 // local
	unsigned char unc_public_key[65] {};  // uncompressed ver. (expecting byte 0 to be value `0x04`, bitcoin-style, followed by R and S (64 bytes total)

//	secp256k1_context *ctx_both = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY); // <--
	if (!ctx_both) { /* not a null pointer.  is this appropriate way to check if the context is valid? */
		LOG_F( ERROR, "Null pointer (expected secp256k1 context). Aborting" );
		return false; } // [MOVEME]? to calling function <-

	//unsigned char skey_bytes[32] {};
	//if (!ConvertHexStr256(in_skey.substr(2), skey_bytes))  /* [TESTME] <-- skip 0x hex specifier, output to `skey_bytes` */
	//	return false;  // err: bad input

	// === verify the secret key ===
	if (!secp256k1_ec_seckey_verify(ctx_both, skey_bytes)) {
		LOG_F(ERROR, "secp256k1_context_create() failed!");
		domesg_verb("secp256k1_context_create failed!", true, V_NORM);
		randombytes_buf(skey_bytes, PRIVATE_KEY_BYTESLEN);
		return false; }  //err
	// === compute public key from skey ===
	if (!secp256k1_ec_pubkey_create(ctx_both, &secp256k1_public_key, skey_bytes)) {
		randombytes_buf(skey_bytes, PRIVATE_KEY_BYTESLEN);
		return false; }  //err
	randombytes_buf(skey_bytes, PRIVATE_KEY_BYTESLEN);
	// === write as uncompressed ===
	size_t out_bytelen { UNCOMPRESSED_PUBKEY_LENGTH };
	if (!secp256k1_ec_pubkey_serialize(ctx_both, unc_public_key, &out_bytelen, &secp256k1_public_key, SECP256K1_EC_UNCOMPRESSED)) {
		return false; }  //err
	//dbg
	if (out_bytelen != UNCOMPRESSED_PUBKEY_LENGTH) {
		LOG_IF_F(ERROR, gVerbosity==V_DEBUG, "GetAddressFromSKey(): secp256k1_ec_pubkey_serialize() wrote %zu bytes (expect 65)!", out_bytelen);
		return false; }
	//dbg
	randombytes_buf(skey_bytes, PRIVATE_KEY_BYTESLEN);  //overwrite private key (cleanup)
	// ===  ===
	if (unc_public_key[0] != 0x04) {
		LOG_F(WARNING, "Expected 0x04 for first byte of computed public key");
		return false; }  //err

	//memcpy(pubkey_u8arry, &unc_public_key[1], 64);			// copy 64 bytes

	// clean up

	//memcpy(pubkey_u8arry, reinterpret_cast<uint8_t*>(&unc_public_key[1]), 64);					// copy 64-byte uncompressed pubkey after the `0x04`
	//print_bytes(pubkey_u8arry, PUBLIC_KEY_BYTESLEN, "public key");	// <-- [TESTME] vs. secp256k1 generated pubkey from same sKey. <-
	// === ... ===

	// === hash the Public Key ===
	print_bytes(reinterpret_cast<uint8_t*>(&unc_public_key[1]), PUBLIC_KEY_BYTESLEN, "public key (skipping byte 0 of 0-64.)");	// <-- [TESTME] vs. secp256k1 generated pubkey from same sKey. <-

	//	bool hash_pubkey_rslt = Keccak256::getHash(pubkey_u8arry, PUBLIC_KEY_BYTESLEN, pubkey_hash);  // <- cast to uint8_t array input from char/unsigned char array? standardize <---- INW
	bool hash_pubkey_rslt = Keccak256::getHash( reinterpret_cast<uint8_t*>(&unc_public_key[1]), PUBLIC_KEY_BYTESLEN, pubkey_hash);  // <- cast to uint8_t array input from char/unsigned char array? standardize <---- INW
	if (!hash_pubkey_rslt) { /* ensure hash succeeded */
		LOG_F(ERROR, "Keccak256 hash of public key failed ");
		//randombytes_buf(pubkey_u8arry, PUBLIC_KEY_BYTESLEN);	// 64 bytes long for public key		 <--- unsigned char array?
		return false; }
	//randombytes_buf(pubkey_u8arry, PUBLIC_KEY_BYTESLEN);	// 64 bytes long for public key		 <--- unsigned char array?

	uint8_t ethaddress_bytes[20] {0};				// for the derived address  <--- unsigned char array?
	memcpy(ethaddress_bytes, &pubkey_hash[12], 20);	// get last 20 bytes / 160 bits of the public key hash <---- to UINT8_T ARRAY <---- STANDARDIZE <---
	randombytes_buf(unc_public_key, 65);
	randombytes_buf(pubkey_hash, HASH_LENGTH);				// 32 bytes long for keccak256 hash  <--- unsigned char array?
	randombytes_buf(secp256k1_public_key.data, 65);

	const std::string str_ethAddress = "0x" + HexBytesToStdString( (uint8_t*)ethaddress_bytes, 20 );  //  HexBytesToStdString() takes an unsigned char array				  <----- ! fixme !
	if (!checkString(str_ethAddress, 42, true, true)) { /* check for proper format.  redundant?  */
		LOG_F(ERROR, "CheckString() returned false in getAddressFromSKey().");
		return false; }  // err
	
//	=== if successful, write ethereum addr for calling func, return true. ===
	//if (gVerbosity > V_NORM)  { printf("Got Ethereum Address:  %s \n", str_ethAddress.c_str()); }  // ...but a uint8_t array is being passed. probably fine but [FIXME]. <-
	//if (bSuccess) {
	 out_ethaddress = str_ethAddress;
	 return true;  //OK
	//} else { return false; }  //err
}


// ETH_GETTRANSACTIONRECEIPT(): Request receipt for a submitted tx's txhash (TODO/WIP)
//								Works! Consider doing w/ a BGworker to avoid lagging main thread if TXview has many items pending
std::string Eth_GetTransactionReceipt(const std::string txhash_str, const int txv_itemno)
{
	const std::string str_data_eth_getrecpt = "{ \"jsonrpc\": \"2.0\", \"method\": \"eth_getTransactionReceipt\", \"params\": [ \"" + txhash_str + "\" ], \"id\" : \"cosmic\" }";
	// ^ replace w/ json object.
	std::string str_parseerr{ "" };  // parse serialized json to object, store resulting
	bool theSuccess{false};

	//if (txhash_str.length() != 66)  return "Error: bad txhash";
	//if (txhash_str.substr(0,2) != "0x" || txhash_str.substr(2).find_first_not_of("0123456789abcdefABCDEF") == string::npos)
	//	return "Error: bad txhash";  // old stuff, more checks as needed. <--- (wip)

	if (gVerbosity == V_DEBUG)  printf("Getting Txn Receipt for TxHash %s... \n", txhash_str.c_str());  // TODO: json object instead:
	std::string str_nodereply = Solo_JSON_Request(str_data_eth_getrecpt, false);					/* call eth_getTransactionReceipt for this TxHash */
	// TODO: check response's `id` key ^ [checkme] <--
	if (!checkErr_b(&str_nodereply, true)) {														/* check for any network error. Trim off "Error:". _a()? */
		doEveryCalls_Values[Timings::getTxReceipt] = doEveryCalls_Settings[Timings::getTxReceipt];	/* if net error, try again next run						 */
		return "Error requesting Tx receipt: " + str_nodereply; }									/* return trimmed error string.							 */

//	str_result = jay["result"].dump();					// reuse as scratch string <--- ???
	const auto jsonobj_nodereply = json11::Json::parse(str_nodereply, str_parseerr, json11::JsonParse::STANDARD);
	if (!str_parseerr.empty())
		return "Error parsing Tx receipt response: " + str_parseerr;  // the trimmed errro

	if (!jsonobj_nodereply.object_items().count("result"))
		return "Key `result` not found in node response";

	const std::string str_result = jsonobj_nodereply["result"].dump();  // should contain the receipt's json key/value pairs.
	if (str_result.length() < 1)	{ return "Error parsing result in Eth_GetTransactionReceipt: empty result."; } /* adjust the min. length [todo]. */
//	if (gVerbosity == V_DEBUG)		{ printf("Tx Receipt:  %s \n\n", str_result.c_str()); }

	// TODO/CHECKME: (check for a 'Null' response: see calling func). <---
	// any other checks? [todo]

	return str_result;		// return `result` key value, should be the receipt
//	return "Error: Unknown ";  // TODO: parse out "error" key if useful. Use ParseKeyFromJsonStr()? <--
}



std::string GetGasPrice(void)
{ // GETGASPRICE(): Perform eth_gasPrice on the node
	//REF: '{"jsonrpc":"2.0","method":"eth_gasPrice","params":[],"id":73}'
	std::string s = "{\"jsonrpc\": \"2.0\", \"method\": \"eth_gasPrice\", \"params\": [], \"id\": \"cosmic_gasprice\"}";
	s = Solo_JSON_Request(s, false);  // request via JSONRPC
	if (gVerbosity == V_DEBUG)
		printf("Got GasPrice from network: %s \n", s.c_str());

	return s;
}


std::string SendRawTx (const std::string& str_payload, const RawTxType tx_type, const int tx_soln_no, bool *success)
{ // Submit "raw transaction" from std::string format payload of hexadecimal-represented bytes, already signed
  // Returns a std::string with the txHash for user display/logging if successful. Otherwise, an Error string.
//	[WIP]: param pointer to success bool in calling function to idicate result, only use the output string if TRUE.
//	[TODO]: use byte-arrays wherever possible, instead of hex strings.
	if (tx_type == RawTxType::Mint) {
		if (DEBUGMODE && DEBUG_NETWORK) { printf("SendRawTx(): submit mint() call Tx payload: %s \n\n", str_payload.c_str()); }	// DBG <---
		if (tx_soln_no < 0) {	/* [FIXME]: min negative code for edge case uses of the function for non-mint raw tx submitting... */
			LOG_F(ERROR, "SendRawTx(): Bad txview item # %d! " BUGIFHAPPENED, tx_soln_no);
			return "Error: bad txview item!";
		}
		// ...
	} else if (tx_type == RawTxType::Transfer) {	// remove this
		if (DEBUGMODE && DEBUG_NETWORK) { LOG_IF_F(INFO, "SendRawTx(): submit transfer() call Tx payload: %s ", str_payload.c_str()); }	// DBG <---
		// ...
	} else { 
		LOG_F(ERROR, "Bad RawTxType in SendRawTx()! " BUGIFHAPPENED);
		return "Error: bad RawTxType in SendRawTx(). " BUGIFHAPPENED;		//err (empty string instead?)
	}

//std::string s1 = "{\"jsonrpc\": \"2.0\", \"method\": \"eth_sendRawTransaction\", \"params\": [\"" + str_payload + "\"" + " ], \"id\": \"cosmic_sendrawtx\"}";
	json11::Json j_request_sendrawtx = json11::Json::object({
		{ "jsonrpc", "2.0" },
		{ "method", "eth_sendRawTransaction" },
		{ "params", json11::Json::array({ str_payload }) },	/* [note]:  not a key pair- just param 0, the payload */
		{ "id", "cosmic" }									/* or: GetRequestIDString(). */
	});
	//const std::string str_request = j_request_transfer.dump();

	if (DEBUGMODE && DEBUG_NETWORK) { printf("SendRawTx(): sending json request: %s \n\n", j_request_sendrawtx.dump().c_str()); }
	std::string str_response = Solo_JSON_Request(j_request_sendrawtx.dump(), false);  // request to node, result to std::string `s`.
	if (DEBUGMODE && DEBUG_NETWORK)	{ printf("Raw Tx send result: %s \n", str_response.c_str()); }
	
	// handle if a LibCurl or other error string was returned by Solo_JSON_Request(). We want a Tx Hash.
	if (!checkErr_a(str_response)) { /* if we have a descriptive error string: */
		domesg_verb("Error submitting sol'n: " + str_response, true, V_NORM);  // V_MORE?
		LOG_IF_F(WARNING, NORMALVERBOSITY, "SendRawTx():	%s", str_response.c_str());
		return "Error: " + str_response;	// return the error str.
	}

	// parse out value of "result" string- should be Tx Hash from node:
	bool parse_success{ false };
	const std::string str_txhash = ParseKeyFromJsonStr(str_response, "result", /*66*/ 0, tx_soln_no, &parse_success);	 // expect txhash or error string.
	if (!parse_success) {
		LOG_F(WARNING, "Error in SendRawTx(): %s", str_txhash.c_str());
		return "Error: " + str_txhash; }  // new [TESTME]

	if (!checkErr_a(str_txhash))  { return str_txhash; }  // return error string <- using checkErr for space. [TESTME] <--- CheckString with the length 66 and hex/0x req'd? <-- WIP/FIXME
	domesg_verb("Node response: " + str_txhash, true, V_DEBUG);  // <- remove
	//
	// Tx Hash should be 
	if (!checkString(str_txhash, 66, true, true))	/* expect length 66: `0x` followed by 32-byte txhash as hex */
		return "Error: bad txhash length or format ";
	
	// NOTE: ParseKeyFromJson will check key 'error' for a node err if the requested key isn't found.
	*success = true;
	domesg_verb("Submitted solution! Tx Hash: " + str_txhash, true, V_NORM);
	return str_txhash;
}
	


// assemble RLP-encoded payload:
  //EthereumSignTx esigntx {};
  //EthereumSig ethereum_sig{};
  //EncodeEthereumSignTx ethereum_encodesigntx;
  //EncodeEthereumTxRequest ethereum_txrequest;

//[note to self]:  getting context for ecdsa operations   [WIP]: initing the context & cleaning up in function which calls SignTx(), for easy cleanup in the event of error.

//std::string uint8t_array_toHexString(const uint8_t* data, const int len);

//#include "RLP_utils.hpp"
//#include "RLP_utilsC.h"
extern "C" void int8_to_char(uint8_t* buffer, int len, char* out); // <-

//WIP: condensing this function
unsigned short Solo_SendSolution(const unsigned int t)		/* t: txView (solutions view) item # */
{ // submit a solution from the queue, doing an equivalent of eth_sendrawtransaction() to assemble an RLP-encoded
  // transaction payload (as hex byte-string).

//	unsigned char noncedata[32] {0};	// <-- WIP. Duplicate? cleanup after signtx().  pass by VALUE?
//	unsigned char skbytes[32] {0};		// <-- WIP. Duplicate? cleanup after signtx(). pass by VALUE?
//	char rawTx_char[256] = {};  // <-- variable length? type?
	unsigned int payload_byteslen{ 0 }; // <---- fixme?
	uint64_t rawTx_u64[24]{};

// === WIP: use a json object, avoid the use of strings where possible ===					<-------
// === assemble transaction ===  // rawTx_u64
    if (!AssembleTX( t, Uint256::ZERO, rawTx_u64, &payload_byteslen)) { /*gStr_SoloEthAddress, rawTx_char,*/ 
        LOG_F( ERROR, "Error assembling transaction. Please report this bug." );	//dbg
        return 1; }

	const std::string str_payloadbytes = uint8t_array_toHexString(reinterpret_cast<uint8_t*>(rawTx_u64), payload_byteslen);

//	if (!checkErr_a(str_payloadbytes)) {
	if (static_cast<unsigned int>(str_payloadbytes.length()) == payload_byteslen*2) {  // not trimming off "Error: ", fixme? <---
		LOG_IF_F(INFO, DEBUGMODE, "RLP-Encoded Payload:		0x%s ", str_payloadbytes.c_str());	// payload to send
		if (DEBUGMODE) print_bytes(reinterpret_cast<uint8_t*>(rawTx_u64), payload_byteslen, "RLP-encoded payload");
	} else {
		LOG_F(ERROR, "Solo_SendSolution(): code 1, err: expected %zu characters, got %d. \n", str_payloadbytes.length(), payload_byteslen * 2); //<-- oughtn't happen
		domesg_verb("Error while assembling Tx payload." BUGIFHAPPENED, true, V_LESS);  // should not occur.
		return 1;	// abort send
	}
//
// === sign, verify, and clean up ===
//
	bool success{ false };
	const std::string result = SendRawTx( "0x" + str_payloadbytes, RawTxType::Mint, t, &success );
	if (!success) {
		LOG_IF_F(WARNING, HIGHVERBOSITY, "- SendRawTx() error:  %s \n", result.c_str());
		gTxViewItems[t].txHash.clear();  // redundant
		gTxViewItems[t].errString = result;
		return 1;	//err
	}

// === Check response to sendrawtx() ===
// expecting 64 hex characters prepended by `0x` (txhash). otherwise: treat as error.   [TODO/FIXME]: put the parsed err mesg/code from geth/etc.!! <--
	if (checkString(result, 66, true, true))
	{ //txhash matches format, store it
		domesg_verb("Solution #" + std::to_string(t) + " Tx Hash:	" + result, true, V_NORM);	// ui event	(V_MORE?)
		LOG_IF_F(INFO, NORMALVERBOSITY, "Solution %d Tx Hash:		%s ", t, result.c_str());	//<-- was: V_MORE. Should now log all TxHashes
		gTxViewItems[t].txHash = result;	// store for receipt requests, UI display and/or logging. <--- [MOVEME]?
		gTxViewItems[t].errString.clear();  // redundant
		// iterate solutions count? <------ [WIP]
		return 0;	//ok
	} else { //store error string if no txhash.	[WIP]: hook up the func that parses the error name/code from geth/etc. <--
		domesg_verb("Solution #" + std::to_string(t) + "  Error: " + result, true, V_NORM);  // event
		gTxViewItems[t].errString = result;  // [MOVEME]?
		LOG_IF_F(INFO, HIGHVERBOSITY, "Solution %d:	No Txn hash, error:	%s ", t, result.c_str());
		return 1;  //err code
	}
	// gTxViewItems[txViewItemNum].txHash = result;
}


//	[TODO]: use type MiningParameters, phase out globals gU64_DifficultyNo` and `gStr_SoloEthAddress`.
// [WIP]. function selector: 0x17da485f0
uint64_t Solo_GetDifficultyNumber( const bool isInitial, bool *parse_success /*... wip */ )
{ // build request (TODO: use json object).
	uint64_t scratchU64{ 0 };
	parse_success = false;	// [redundant]
	
//	LOG_IF_F(INFO, DEBUGMODE && DEBUG_NETWORK, "Retrieving mining difficulty from contract...");
	LOG_IF_F(INFO, DEBUGMODE||DEBUG_NETWORK, "Retrieving mining difficulty from contract...");

// [TODO] / [FIXME]: this will be "slow", use JSON object. <--
	std::string str_data = "0x17da485f0000000000000000000000000000000000000000000000000000000000000000";
	str_data = "{\"jsonrpc\":\"2.0\", \"method\":\"eth_call\", \"params\": [{\"from\": ";
	str_data += "\"" + gStr_SoloEthAddress + "\", \"to\" : \"" + gStr_ContractAddress + "\" , \"data\" : \"";
	str_data += str_data + "\" }, \"latest\"], \"id\" : \"cosmic_diff\" }";

	std::string s1 = Solo_JSON_Request(s1, isInitial);	//via libcurl

	// parse result (from string `s1`:)  REVISE THIS: parse a number instead of a string, neater approach.
	s1 = ParseKeyFromJsonStr(s1, "result", 66, -1, parse_success);  // parsing a 256-bit number (represented as string)

	// error checks  [TODO]: anything else to check?
	if (parse_success && checkString(s1.substr(2), 66, true, true))  /* skip `0x` [wip] */
	{  // pool returns decimal. solo mode: expect a uint256 in hex string representation.
		// [idea]:
		const Uint256 scratchU256 = Uint256( s1.c_str() );  // uint256 constructor to parse hexstr to scratch uint256.
		uint8_t diffnumber64_bytes[32]{ 0 };		// output byte-array
		uint64_t parsed_diffU64{ 0 };				// for the parsed U64 difficulty#.

		scratchU256.getBigEndianBytes(diffnumber64_bytes);
		memcpy(&parsed_diffU64, diffnumber64_bytes, 8);					// <- get the correct 64 bits from bignum...[WIP]
		if (parsed_diffU64 > 0 && gU64_DifficultyNo != scratchU64) {
			printf("Got new difficulty number:  %s ! \n", s1.c_str());	// New Difficulty #!
			gU64_DifficultyNo = parsed_diffU64;							// set internally
			if (gVerbosity == V_DEBUG) { printf("Set gU64_DifficultyNo (integer) # to:  %" PRIu64 " \n", gU64_DifficultyNo); }
		}
		return parsed_diffU64;
	}
	else
	{ // `s1` should contain an error string because `parse_success`==false
		if (gVerbosity > V_NORM) { printf("Error retrieving the Difficulty Number. \n"); }			// 
		if (gVerbosity == V_DEBUG) { printf("Got: %s \n", s1.c_str()); }							// dbg
		//
		if (isInitial) { return 1; }	// if error getting initial mining params, mining won't start.
		  else { return 0; }			// 0: not a valid target (err).
	}
} // ^function is WIP. call only if computing mining target from maxtarg, exponent & diff#. otherwise just parse contract's Target.


//first draft stuff, revise and condense this function [TODO]. <--
// [NOTE]: in Solo Mode, the Minting Address is not retrieved. Instead it's derived from the loaded account
bool Solo_GetMiningParameters(const bool isInitial, const bool compute_target, miningParameters *params)
{
 // currently gets both target and difficulty#. we really only need one. use the `compute_target` arg. [TODO]
	std::string s1{ "" };	//initialization redundant
	unsigned short errorCount{ 0 };
	bool newChallenge{false};	// [solo mode] if `newChallenge` is true: _do_ check for new difficulty/target. [TESTME] <--

	//if (params->mineraddress_str.length() != 42) {
	if (!checkString(params->mineraddress_str, 42, true, true)) {
		domesg_verb("Solo_GetMiningParameters(): no account loaded. ", true, V_MORE);
		domesg_verb("Miner's Eth Address: " + params->mineraddress_str, true, V_DEBUG);
		return false;	//err (mining has probably ended)
	}

// [WIP]: generic Eth_Call() function. <---
// getting the challenge from the user-specified contract, using getChallengeNumber() method. // [TODO]: replace w/ generic Eth_Call() function, Json11 object <--
	std::string str_data = "0x4ef376280000000000000000000000000000000000000000000000000000000000000000";
	s1 = "{\"jsonrpc\":\"2.0\", \"method\":\"eth_call\", \"params\": [{\"from\": ";
	s1 += "\"" + params->mineraddress_str + "\", \"to\" : \"" + gStr_ContractAddress + "\" , \"data\" : \"";  // [TESTME] ensure output is correct (replace w/ json obj.)
	s1 += str_data + "\" }, \"latest\"], \"id\" : \"cosmic\" }";

// Exception could happen HERE: mutex-lock protection needed on a resource? Probably gTxViewItems somehow... hmmm <--- WIP / FIXME
	s1 = Solo_JSON_Request(s1, isInitial);				// access node, send our JSON request, result to `s1`
	if (!checkErr_a(s1)) {
		LOG_IF_F(WARNING, HIGHVERBOSITY, "Error retrieving Challenge:	%s", s1.c_str());
		return false;	//err
	}
	
	bool parse_success{ false };	//to decrease # of API requests to endpoint, [TESTME].
	s1 = ParseKeyFromJsonStr(s1, "result", 66, -1, &parse_success);	 // <--Parse JSON: func expects 66 characters: 0x + 64 hexadecimal (32 bytes, 256-bit) challenge
	
	// CheckString(): does various sanity checks. Expecting uint256 challenge as 66-character hex string, 0x+32 bytes
	// the error string check is redundant, see check of `parse_success`.	[wip]
	if (parse_success && checkString(s1, 66, true, true))
	{
		if (s1 != params->challenge_str) {			/* [todo]: re-used string `s1` has ambiguous name */
			params->challenge_changed = true;
			params->params_changing = true;
			params->challenge_str = s1;				//save the new challenge (hex string)
			newChallenge = true;					//target/difficulty will be retrieved this run
			LOG_IF_F(INFO, NORMALVERBOSITY, "New Challenge:	%s ", params->challenge_str.c_str());
			domesg_verb("New Challenge:	" + params->challenge_str, true, V_NORM);	//to ui. [TODO]: use log callback. <--
		} //will check for new difficulty target next	[WIP] <---
	} else { /* unsuccessful: `s1` should be a semi-helpful error string */
		LOG_IF_F(WARNING, HIGHVERBOSITY, "Error retrieving the Challenge:	%s", s1.c_str());	// spawn event for err with severity=1, no dialog to inform user/do stdout
		if (isInitial) return false;	//err
		 else ++errorCount;
	}	// if error getting initial parameters, abort mining start. Otherwise: keep retrieving params.
		// [FIXME / TODO]: re-try for auto-start [TODO/FIXME] <--

// if the challenge hasn't changed, the token contract's difficulty target should also be unchanged.	[TESTME] <--
	if (!newChallenge && !isInitial) {
		LOG_IF_F(INFO, DEBUGMODE && DEBUG_NETWORK, "(Solo Mode)	Challenge unchanged: not checking for new diff target.");	// [remove] <--
		return true;	//OK
	}

//	=== challenge changed: ===		// [TODO]: use json11 object here.
	LOG_IF_F(INFO, HIGHVERBOSITY, "Challenge changed. Retrieving/computing mining target.");	//DEBUGMODE
// [WIP]: don't get difficulty # if "Compute Target Locally" is not enabled	[TESTME]. <--
// getting Difficulty Number from the user-specified contract:
	if (compute_target) {
		LOG_IF_F(INFO, DEBUGMODE, "Computing target using retrieved difficulty# and maxtarget:");	//DEBUGMODE

	//	const uint64_t old_difficulty = params->difficultyNum;
		const uint64_t new_difficulty = Solo_GetDifficultyNumber(isInitial, &parse_success);	// [TESTME] <---
	//	gU64_DifficultyNo = Solo_GetDifficultyNumber(isInitial, &parse_success);				// [old]
		if (new_difficulty != params->difficultyNum) {	// or `old_difficulty`
			params->difficultyNum = new_difficulty;		//<-- ACQUIRE COREPARAMS MUTEX FIRST? <---- [FIXME] / [TODO].
			params->difficulty_changed = true;
			params->params_changing = true;		//<--
			LOG_IF_F(INFO, NORMALVERBOSITY, "Got New Difficulty:	%" PRIu64 ".", params->difficultyNum);
		}
	}
	else { //getting mining target from contract via the node:

		LOG_IF_F(INFO, DEBUGMODE, "Retrieving target from contract:");	//DEBUGMODE
		str_data = "0x32e997080000000000000000000000000000000000000000000000000000000000000000";	//getting Target from contract via getChallengeNumber() method (0x32e99708)
		s1 = "{\"jsonrpc\":\"2.0\", \"method\":\"eth_call\", \"params\": [{\"from\": ";
		s1 += "\"" + params->mineraddress_str + "\", \"to\" : \"" + gStr_ContractAddress + "\" , \"data\" : \"";	//was gStr_DerivedEthAddress
		s1 += str_data + "\" }, \"latest\"], \"id\" : \"cosmic\" }";  // <--- [TODO]: replace with json11 object. <--

		LOG_IF_F(INFO, DEBUGMODE, "Requesting Mining Target from contract");
		s1 = Solo_JSON_Request(s1, isInitial);		//send request to node. result out to string `s1`
		if (!checkErr_a(s1)) {
			LOG_IF_F(WARNING, HIGHVERBOSITY, "Error retrieving mining target");
			return false;	//err
		}

		parse_success = false;	// [redundant]: also re-set to false in ParseKeyFromJsonStr().
		s1 = ParseKeyFromJsonStr(s1, "result", 66, -1, &parse_success);		// <-- Expects 66 characters: 0x + 64 hexadecimal (32 bytes, 256-bit) from TokenPool (TARGET)
		if (parse_success && checkString(s1, 66, true, true)) {				/* func contains various sanity checks, and "Error" check. expects length of 66 and 0x  */
			if (s1 != params->target_str) { //check against string version of current mining target:
				params->target_str = s1;
				params->target_changed = true;	//<---
				params->params_changing = true;	//<---
				LOG_IF_F(INFO, HIGHVERBOSITY, "Got new mining target:	%s ", params->target_str.c_str());
			}
		}
		else {
			LOG_IF_F(WARNING, HIGHVERBOSITY, "Error parsing mining target");	//use log callback [TODO]
			if (isInitial)
				return false;	//error getting initial parameters, mining will not start
			++errorCount;	//otherwise
		}

	}
// [WIP]: if !compute_target, gU64_DifficultyNo will be 0? <---

	LOG_IF_F(INFO, DEBUGMODE, "Solo_GetMiningParameters():	%u errors", errorCount);	// [REMOVE]
	return (errorCount > 0);	//True=OK, False=Error(s).
}


void Erase_gSK(void)
{ //For Solo Mining support. Overwrite private key with random bytes
	uint8_t random_bytes[32] = { 0 };	 // buffer 32 bytes long
	randombytes_buf(random_bytes, 32);  // fill with randoms
	gSk256 = Uint256(random_bytes);	 // overwrite pkey <-- write to the value[]'s? <-- [wip]
	
	gStr_SoloEthAddress = "0x00";	// require acct be loaded again.
}


void ClearTxViewItemsData(void)
{ // clear TxView (Solutions View) items' data from gTxViewItems
	LOG_IF_F(INFO, HIGHVERBOSITY, "Clearing txView data... ");	//<-- event message?

	// [TODO]: use a container object instead.
	for (unsigned int i = 0; i < DEF_TXVIEW_MAX_ITEMS; ++i)
		ClearTxViewItem(i);  // clear individual index

	gTxViewItems_Count = 0;		// <-- WIP/FIXME
	gTxView_WaitForTx = NOT_WAITING_FOR_TX;  // null
}

void ClearTxViewItem(const int itemNum)
{ // clears an individual txview item (solution-tx) in the array.
  // init all members.  [TODO]: just use memset?
  // [TODO / IDEA]  make a new container and place the txViewItems into it- make life a little easier :D
	gTxViewItems[itemNum].deviceOrigin = UINT8_MAX;  // _MAX = null, CUDA devices indexed from 0
	gTxViewItems[itemNum].deviceType = DeviceType::Type_NULL;
	gTxViewItems[itemNum].errString.clear();
	gTxViewItems[itemNum].last_node_response = NODERESPONSE_OK_OR_NOTHING;
	gTxViewItems[itemNum].networkNonce = UINT64_MAX;
	gTxViewItems[itemNum].slot_occupied = false;
	gTxViewItems[itemNum].solution_no = -1;
	gTxViewItems[itemNum].status = TXITEM_STATUS_EMPTYSLOT;
	gTxViewItems[itemNum].str_challenge.clear();;
	gTxViewItems[itemNum].str_digest.clear();
	gTxViewItems[itemNum].str_signature_r.clear();
	gTxViewItems[itemNum].str_signature_s.clear();
	gTxViewItems[itemNum].str_signature_v.clear();
	gTxViewItems[itemNum].str_solution.clear();
	gTxViewItems[itemNum].submitAttempts = 0;
	gTxViewItems[itemNum].txHash.clear();

	if (gTxViewItems_Count - 1 >= 0)  gTxViewItems_Count -= 1;
	else  gTxViewItems_Count = 0;
}
