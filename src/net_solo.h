#pragma once
// COSMiC V4
// Miner<->Ethereum Node Comm for Solo Mining

// === standard library stuff ===
#include <cinttypes>					// was: #include <stdint.h>
#include <string>
//#include <mutex>

// === libs ===
#include <bitcoin-cryptography-library/cpp/Uint256.hpp>
#include <secp256k1.h>					// 
#include <secp256k1_recovery.h>			// 
#include <ethereum-rlp/include/RLP.h>	// [new]
//#include <libsodium/sodium.h>			// sodium cryptography lib
//#include <curl/curl.h>				// network functionality

// === cosmic headers ===
#include "network.hpp"					// cosmic common network stuff (both modes)
//#include "util.h"						// checkErr_a(), _b()

//=== keystore defs ===					// [TODO]: user-configurable encryption settings.
//#define CIPHERTEXT_LEN (crypto_secretbox_xchacha20poly1305_MACBYTES + MESSAGE_LEN)
//#define MESSAGE_LENGTH_STRING	64			// 32 bytes, or 64 hex digits. was at one point a 66 character text string of hex plus `0x`
//#define MESSAGE_LEN				32		// bytes
//#define SALT_LENGTH				32		// "
//#define DERIVED_KEY_LENGTH		32		// "
//#define PUBLIC_KEY_LENGTH	64				// 
//#define PBKDF2_ITERATIONS	900000			// for key derivation from password (todo: user-configurable?)

constexpr auto PRIVATE_KEY_BYTESLEN = 32;  // length in bytes:32, length as hex string:64
constexpr auto PUBLIC_KEY_BYTESLEN = 64;			// MOVEME <--

#define DEFAULT_GASPRICE	3000000000		// 3 billion wei (3 gwei)
#define DEFAULT_GASLIMIT	200000			// units max

constexpr auto KECCAK256_HASHLENGTH = 32;  // keccak256 hash, 32 bytes long
constexpr auto PAYLOAD_MINLEN_BYTES = 100;	// <- wip
constexpr auto PAYLOAD_MAXLEN_BYTES = 400;  // <- wip
constexpr auto SIGNATURE_LENGTH = 64;  // <-- just the 2 256-bit parts (r, s)
//constexpr auto PUBKEY_LENGTH = 64;
//constexpr auto PRIVATE_KEY_LENGTH = 32;		// in bytes
constexpr auto INITIAL_PAYLOAD_MIN_LENGTH = 75;
constexpr auto INITIAL_PAYLOAD_MAX_LENGTH = 400;
constexpr auto TX_PAYLOAD_MIN_LENGTH = 100;
constexpr auto TX_PAYLOAD_MAX_LENGTH = 600;

// #define GETCOLORFOR_DEVICETEMP 1			// color for gpu temp reading
// #define GETCOLORFOR_DEVICETACH 2			// color for tachometer reading


// === only if included by a file compiled with /clr: ===
#if (_MANAGED) && (_M_CEE)
using namespace System;
using namespace System::Numerics;

//#include <msclr/marshal.h>			// marshalling native<->managed types
//#include "msclr/marshal_cppstd.h"		// "


//unsigned short Solo_SendDonation(System::Numerics::BigInteger reward256);  // <-- todo: return signature as std::string?
//unsigned short Solo_SendDonation(System::Numerics::BigInteger reward256);
System::Numerics::BigInteger /*Uint256*/ Solo_GetMiningReward(const std::string sContractAddr);
//System::Int64 Eth_GetTransactionCount(void);
#endif


enum class EcdsaResult { /* was `enum` */
	OK = 0, ErrContextCreation = 1, ErrSKeyVerify = 2, ErrPubKeyGenerate = 3, ErrPubKeySerialize = 4, ErrSignRecoverable = 5, 
	ErrSigVerify = 6, ErrConvertSig = 7, ErrSigSerialize = 8, ErrSigParse = 9, ErrRecover = 10, ErrBadPayloadLen = 11, 
	ErrUnevenPayload = 12, ErrBadPointer = 13, ErrHashFailed = 14
}; //[WIP]:  very granular error codes for debugging

void Solo_CheckStalesInTxView(const miningParameters* params);	// [NOTE]: function definition is in network.cpp
																// because net_solo.cpp is still compiled with /clr
void Solo_DonationCheck(void);

std::string GetAccountBalance_Ether(std::string ethAddress);

std::string GetAccountBalance_Tokens(std::string ethAddress);

void HexStrToByteArray_b(const char* cstring, uint8_t* outArray, const unsigned int byte_len);  // move to `util`?

unsigned short DeriveKeyFromPassPhrase_WinSDK(const /*unsigned*/ char *passPhrase, const size_t passPhraseLen, unsigned char *keyOut, unsigned char *saltOut, const bool newSalt);

bool getAddressFromSKey(secp256k1_context *ctx_both, unsigned char *skey_bytes, std::string& out_ethaddress);

EcdsaResult SignTx(secp256k1_context* ctx_both, const EthereumSignTx tx, EthereumSig* sig_out, const unsigned char* noncedata_bytes,
	const size_t nonce_len/*, const unsigned char* sk_bytes, const size_t sk_len*/);  //params wip. fixed-size arrays?  [todo]

std::string Solo_JSON_Request(const std::string& jsonData, const bool strictTimeouts);

std::string Eth_GetTransactionReceipt(const std::string txhash_str, const int txv_itemno);

std::string GetGasPrice(void);

void HandleNodeError(const std::string& error_mesg, const uint64_t error_code, const int txViewItemNo);

std::string SendRawTx(const std::string& str_payload, const RawTxType tx_type, const int tx_soln_no, bool* success);
//std::string SendRawTx_Donate(const std::string& str_payload);  //only one arg

unsigned short Solo_SendSolution(const unsigned int t);

void Erase_gSK(void);

int wallet_ethereum_assemble_tx_initial( /*const*/ EthereumSignTx* msg, uint64_t* encoded_U64);	//net_rlp.cpp

void ClearTxViewItemsData(void);
void ClearTxViewItem(const int itemNum);

//std::string Web3_Sha3(const std::string mesg);

// === exterrnals for solo mode ===
extern std::string gStr_SoloNodeAddress;			// api endpoint. could be localhost, infura url, etc.
extern unsigned int gSoloNetInterval;				// Solo Network access interval, in ms  (todo: make sure this is populated at CosmicWind _Shown time) <--
extern std::string gStr_SoloEthAddress;				// public eth address to mint with. (derived from private key)
extern Uint256 gSoloEthBalance256;					// temp storage of eth balance (wei): option to stop mining if < the required gas to submit a solution. [TODO] <--

extern std::string gTxCount_hex;					// 
//extern Uint256 gTxnCount_uint256;					// transaction count for the eth address in use, supporting Tx nonce values from 0 to UINT256_MAX - 1.

extern Uint256 gSk256;
extern std::string gStr_GasPrice_hex;				// "
extern std::string gStr_GasLimit_hex;				// "
extern uint64_t gU64_GasPrice;						// 3 G-wei default
extern uint64_t gU64_GasLimit;						// 200,000 max default

extern int gTxView_WaitForTx;						// wait for this tx# to be confirmed or fail
extern short gDonationsLeft;						// for auto-donation in solo mode. increased when a solo-mode sol'n-tx is confirmed.
													// [WIP]: consolidating xfers when possible to save gas.

