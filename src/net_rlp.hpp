#pragma once
// should be compiled as C++, native (without the /clr option.)

#include <string>
#include <iomanip>
#include <sstream>

#include <ethereum-rlp/include/RLP.h>
//#include <ethereum-rlp/include/RLP_test.h>
//#include <RLP_utils.hpp>
//#include "ethereum-rlp/include/RLP_utils.hpp"        // note: hybrid C/C++ file.
// -or-  enum GetParmRslt { OK_PARAM_UNCHANGED = 0, GETPARAM_ERROR = 1, OK_PARAM_CHANGED = 2 };

#include <json11/json11.hpp>
#include <loguru/loguru.hpp>
#include <libsodium/sodium.h>       //<-

#include "network.hpp"
#include "net_solo.h"  /* -or- */   //<-

#include "util.hpp"
#include "defs.hpp"
//#include "net_RLPutils.hpp"
#include "coredefs.hpp"         //or, extern int gSolo_ChainID;    //<---

//using namespace System;
//using namespace System::Globalization;
using namespace json11;
using namespace loguru;


// static char rawTx[256] {};
constexpr auto ECDSANONCE_BYTESLEN = 32;    //NONCEDATA_LENGTH
constexpr auto ETH_ADDR_SIZE_IN_BYTES = 20;
constexpr auto DATA_SIZE_IN_BYTES = 56;
constexpr auto funcselect_mint_str = "1801fbe5";
uint8_t funcselect_transfer[4]{ 0xa9, 0x05, 0x9c, 0xbb };

//unsigned char donateaddr_bytes[20] {// 0x57C906Fce9527C5cD9fE415FC6ECEC33A67e708E
//    0x57, 0xC9, 0x06, 0xFc, 0xe9, 0x52, 0x7C, 0x5c, 0xD9, 0xfE, 0x41, 0x5F, 0xC6, 0xEC, 0xEC, 0x33, 0xA6, 0x7e, 0x70, 0x8E };

// === externals [wip] ===
extern std::string gStr_SoloEthAddress;  // net_solo.h
extern std::string gTxCount_hex;     // moveme. defined in: net_solo.
//extern ushort gVerbosity;
//extern Uint256 gU256_TxnCount;                                                          // for the network nonce

extern struct txViewItem gTxViewItems[DEF_TXVIEW_MAX_ITEMS];    //<---
//extern std::string gStr_ContractAddress;	//<-- or include "coredefs.hpp"



// === C++ functions ===
std::string uint8t_array_toHexString(const uint8_t* data, const int len);                   // or include "util.h".
std::string Solo_JSON_Request(const std::string& jsonData, const bool strictTimeouts);      // or include "net_solo.h"
GetParmRslt UpdateTxCount(const std::string& for_address, std::string& hex_txcount_out);

//std::string Solo_JSON_Request(const std::string jsonData, const bool strictTimeouts);

// === C functions ===
extern "C" int wallet_ethereum_assemble_tx(EthereumSignTx* msg, EthereumSig* tx, uint64_t* rawTx);      // RLP_test <--
//
extern "C" bool wallet_encode_element(pb_byte_t* bytes, pb_size_t size,
    pb_byte_t* new_bytes, pb_size_t* new_size, bool remove_leading_zeros);                              // RLP.c
//
extern "C" void wallet_encode_int(uint32_t singleInt, pb_byte_t* new_bytes);                            // RLP.c

extern "C" int wallet_encode_list(EncodeEthereumSignTx* new_msg, EncodeEthereumTxRequest* new_tx, uint64_t* rawTx);

extern "C" int hex2byte_arr(const char* buf, const int len, uint8_t* out, int outbuf_size);             // RLP_utils

extern "C" int size_of_bytes(int str_len);                                                              // RLP_utils

extern "C" void int8_to_char(uint8_t* buffer, int len, char* out);                                     // RLP_utils




std::string RLP_EncodePayload_Donation(const std::string input_str);        // { /* skeleton func. */ }  // [WIP].

int wallet_ethereum_assemble_tx(EthereumSignTx* msg, EthereumSig* sig_in, uint64_t* rawTx);

// New. Encode the Transaction with placeholder signature components. V=ChainID, R=null (0x80), S=null (0x80).
//      This RLP-encoded message will be keccak256 hashed and signed w/ secp256k1.
//      Signature components will be added to the Tx before it's sent (actual V, R, S).
// [testme].
int wallet_ethereum_assemble_tx_initial( /*const*/ EthereumSignTx* msg, uint64_t* encoded_U64);


// === UNFINISHED / PHASING OUT: ===
//unsigned char* RLP_EncodeData(const std::string& minedNonce, const std::string& solnDigest);  /*<- return type? [wip]. */
std::string RLP_EncodeData_StringVer(/*const*/ std::string& minedNonce, const std::string& solnDigest, bool* success);
/* [TEST] 3rd param extraneous? */

// Byte-arrays version. Use this one!
void RLP_EncodeData_Transfer(const pb_byte_t dest_address[20], const pb_byte_t amount_sats256[32], 
    pb_byte_t* data_out, unsigned int* data_size, bool* success);

//AssembleTX() - Regular version.
bool AssembleTX(const int solution_no, const Uint256 donation_U256, uint64_t rawTx_u64[24], unsigned int* payload_length);
