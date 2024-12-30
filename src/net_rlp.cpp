

#include "net_rlp.hpp"
//using namespace System;
//using namespace System::Globalization;
//using namespace loguru;
//using namespace json11;

int wallet_ethereum_assemble_tx(EthereumSignTx* msg, EthereumSig* sig_in, uint64_t* rawTx)
{
    EncodeEthereumSignTx new_msg;
    EncodeEthereumTxRequest new_tx;
    memset(&new_msg, 0, sizeof(new_msg));
    memset(&new_tx, 0, sizeof(new_tx));

    // NOTE: input `msg` and `tx`, to output `new_msg` and `new_tx`:
    if (!wallet_encode_element(msg->nonce.bytes, msg->nonce.size, new_msg.nonce.bytes, &(new_msg.nonce.size), false))                   { return 0; }
    if (!wallet_encode_element(msg->gas_price.bytes, msg->gas_price.size, new_msg.gas_price.bytes, &(new_msg.gas_price.size), false))   { return 0; }
    if (!wallet_encode_element(msg->gas_limit.bytes, msg->gas_limit.size, new_msg.gas_limit.bytes, &(new_msg.gas_limit.size), false))   { return 0; }
    if (!wallet_encode_element(msg->to.bytes, msg->to.size, new_msg.to.bytes, &(new_msg.to.size), false))                               { return 0; }
    if (!wallet_encode_element(msg->value.bytes, msg->value.size, new_msg.value.bytes, &(new_msg.value.size), false))                   { return 0; }
    if (!wallet_encode_element(msg->data_initial_chunk.bytes, msg->data_initial_chunk.size, new_msg.data_initial_chunk.bytes, &(new_msg.data_initial_chunk.size), false))   { return 0; }
//  wallet_encode_int(tx->signature_v, &(new_tx.signature_v));
    wallet_encode_int(sig_in->signature_v, (pb_byte_t*)(&(new_tx.signature_v) ));                                                                              /* V */

    if (!wallet_encode_element(sig_in->signature_r.bytes, sig_in->signature_r.size, new_tx.signature_r.bytes, &(new_tx.signature_r.size), true))    { return 0; }   /* R */
    if (!wallet_encode_element(sig_in->signature_s.bytes, sig_in->signature_s.size, new_tx.signature_s.bytes, &(new_tx.signature_s.size), true))    { return 0; }   /* S */

    // the bytes of the payload we signed, and the signature bytes, are RLP encoded, serialized into the raw Tx bytes we will submit.
    const int length = wallet_encode_list(&new_msg, &new_tx, rawTx);  // out to `rawTx` bytes
    return length;
}


// New. Encode the Transaction with placeholder signature components. V=ChainID, R=null (0x80), S=null (0x80).
//      This RLP-encoded message will be keccak256 hashed and signed w/ secp256k1.
//      Signature components will be added to the Tx before it's sent (actual V, R, S).
// [testme].
int wallet_ethereum_assemble_tx_initial( /*const*/ EthereumSignTx *msg, uint64_t *encoded_U64 )
{
    EncodeEthereumSignTx new_msg{}; // 
    EncodeEthereumTxRequest new_tx{}; //

    memset(&new_msg, 0, sizeof(new_msg));   // redundant?
    memset(&new_tx, 0, sizeof(new_tx));     // "
    //memset(&new_tx, 0, sizeof(new_tx));
    // set V to the chainID: <--- [WIP/TODO]
    //new_msg.chain_id = gSolo_ChainID;   // <--
    //-or-
    wallet_encode_int(static_cast<uint32_t>(gSolo_ChainID), reinterpret_cast<pb_byte_t*>(&new_tx.signature_v)); // <-- new: placeholder `V`
    new_tx.has_signature_v = true;  // V: the chain ID (placeholder)! we've set the value.
    new_tx.has_signature_r = true;  // R/S: these have no value set, but let's pretend they do
    new_tx.has_signature_s = true;  //      and i expect a damn null byte 0x80!!
    //new_msg.has_chain_id = true;


    wallet_encode_element(msg->nonce.bytes, msg->nonce.size,
        new_msg.nonce.bytes, &(new_msg.nonce.size), false);
    wallet_encode_element(msg->gas_price.bytes, msg->gas_price.size,
        new_msg.gas_price.bytes, &(new_msg.gas_price.size), false);
    wallet_encode_element(msg->gas_limit.bytes, msg->gas_limit.size,
        new_msg.gas_limit.bytes, &(new_msg.gas_limit.size), false);
    wallet_encode_element(msg->to.bytes, msg->to.size, new_msg.to.bytes,
        &(new_msg.to.size), false);
    wallet_encode_element(msg->value.bytes, msg->value.size,
        new_msg.value.bytes, &(new_msg.value.size), false);
    wallet_encode_element(msg->data_initial_chunk.bytes,
        msg->data_initial_chunk.size, new_msg.data_initial_chunk.bytes,
        &(new_msg.data_initial_chunk.size), false);

    
    // encode a placeholder r and s (0x80) byte in an elegant way
    // or do a kludge for now and streamline after next test build. <--
    //wallet_encode_int(initialSignTx->signature_v, &(new_tx.signature_v));    // <-- orig
    //new_tx.signature_v = static_cast<uint32_t>(gSolo_ChainID);  // <- since we know this will be a single byte in the 00-7f range..
    
    //pb_byte_t rs_kludge[1]{0};
    //pb_size_t rs_kludge_size{};
    //wallet_encode_int(0, rs_kludge);
    //wallet_encode_element(rs_kludge, 1, ...);
    new_tx.signature_r.size = 1; // <-- results in 0x00, not 0x80 (null byte)
    new_tx.signature_s.size = 1; // <-- "



    int length = wallet_encode_list( &new_msg, &new_tx, encoded_U64 );
    //new.. placeholders for the `r` and `s`. Figure out how to make ethereum-rlp encode these without unnecessary contraptions. <--
    new_tx.has_signature_r = true;
    new_tx.has_signature_s = true;
    //new_tx.has_signature_v = true;
    //memcpy(new_tx.signature_r.bytes, source, howmany...
    //new_tx.signature_r.size...
    //

    return length;
}


// UNFINISHED
//unsigned char* RLP_EncodeData(const std::string& minedNonce, const std::string& solnDigest)  /*<- return type? [wip]. */
//{
//    // once assembled, should be 68 bytes total (32 byte nonce, 32 byte challenge digest, four byte function select).
//    // 0xb8 = 0xb7 + size of length in bytes(1), 44 = hex length of data field(?)
//
//    //uint8_t prefix_bytes[2] = { 0xb8, 0x44 };                 // RLP prefix: 70 bytes as 140 hex characters
//    //uint8_t  or  char* ?
//    unsigned char minedNonce_bytes[32]{}; /*<- type? */
//    unsigned char solnDigest_bytes[32]{}; /*<- type? */
//    Uint256 minedNonce256 = Uint256(minedNonce.c_str()); // temp kludge
//
//
//    // Make function selector user-configurable. or load abi? [todo].
//    // [todo/wip]: ProxyMint() call with the token contract address as a parameter also. <-
//    unsigned char data[68]{         /* <-- type? see calling func. */
//        0x18, 0x01, 0xfb, 0xe5      // function selector:   first 4 bytes of keccak256 hash of ` mint(uint256,bytes) `
//    };  // the remainder of data[] will be inited to 0
//
//    //if (checkString(minedNonce, ...)) // <---- wip
//     // memcpy (&data[4], // copy the two 256bit uint parameter arguments to mint() <--- WIP...
//
//
//
//    //const std::string str_data = /*"b844" +*/ Check0x_StringVer(gStr_Set_MintFunctionSelector, true) +          /* prefix will be taken care of by ethereum-rlp lib :) */
//    //    Check0x_StringVer(minedNonce, true) + Check0x_StringVer(solnDigest, true);
//
//    //  wip: change this to char array ^
//    //
//    //printf("? str_data: %s \n", str_data.c_str());
//
//     //if (str_data.length() != 140) /*&& str_data.length() != 72 <-- for ERC-918 style */  // sanity check: 68 bytes * 2 hex characters in string
//     //{
//     //   printf("Error in RLP_EncodeData: expected 70 bytes, got: %s \n", str_data.c_str());
//     //   return "Error: invalid string length for `data` field: expected 70 bytes ";		 // plus 2 bytes of RLP field description
//     //}
//
//    return data;  // was: return str_data;
//}


//constexpr auto funcselect_mint_str = "1801fbe5";
// PHASING OUT:
std::string RLP_EncodeData_StringVer (/*const*/ std::string& minedNonce, const std::string& solnDigest, bool* success)  /* [TEST] 3rd param extraneous? */
{ // Note: first-draft version, use the bytes version!
    // sanity-check the inputs: 0x specifier? <--- [wip/fixme!]
    //if (checkString(minedNonce, 64, false, true)) || Check0x(solnDigest)) { /* remove need for this, check calling function. [wip] */
    if (!checkString(minedNonce, 66, true, true) || !checkString(solnDigest, 66, true, true)) {
        LOG_F(ERROR, "RLP_EncodeData_StringVer(): Bad input nonce or digest!" BUGIFHAPPENED);
        return "";
    }

    if (DEBUGMODE) { printf("minedNonce:  %s \nsolnDigest: %s \n", minedNonce.c_str(), solnDigest.c_str()); }
    //   LOG_IF_F(ERROR, gVerbosity == V_DEBUG, "RLP_EncodeData_StringVer():  Unexpected `0x` in input."); // <- remove
    //    return; }

    const std::string output_str = funcselect_mint_str + minedNonce.substr(2) + solnDigest.substr(2);  // <- remove `0x` if present. should it have been there in the first place? <--
    if (output_str.length() == 136) { /* 4-byte function selector.  2 32-byte arguments.    total: 68 bytes, or 136 characters. */
        LOG_IF_F(ERROR, DEBUGMODE, "RLP_EncodeData_StringVer()  OK: %s, length %zu \n", output_str.c_str(), strlen(output_str.c_str()));  //<-- DBG only <--
        *success = true;    // <-
        return output_str;
    } else {
        *success = false;   // <-
        LOG_F(ERROR, "RLP_EncodeData_StringVer()  Err: bad `data` field length %zu", output_str.length());  //<- err
        return "";
    }
}



// Byte-arrays version. Use this one!
 //uint8_t funcselect_transfer[4]{ 0xa9, 0x05, 0x9c, 0xbb };
//unsigned char donateaddr_bytes[20] {// 0x57C906Fce9527C5cD9fE415FC6ECEC33A67e708E
//    0x57, 0xC9, 0x06, 0xFc, 0xe9, 0x52, 0x7C, 0x5c, 0xD9, 0xfE, 0x41, 0x5F, 0xC6, 0xEC, 0xEC, 0x33, 0xA6, 0x7e, 0x70, 0x8E };
//constexpr auto ETH_ADDR_SIZE_IN_BYTES = 20;
//constexpr auto DATA_SIZE_IN_BYTES = 56;

// [todo]: consider compacting this into AssembleTx()! <--
void RLP_EncodeData_Transfer(const pb_byte_t dest_address[20], const pb_byte_t amount_sats256[32], pb_byte_t* data_out, unsigned int* data_size, bool* success)
{   // encode the "data" field of a transaction for calling transfer() in token contract. takes destination address
    // as byte array, donation amount with 8 decimal places (Uint256), and returns `data` field as bytes
    *data_size = DATA_SIZE_IN_BYTES;
    memset(data_out, 0, 56);

// assembling the data field (without strings)
// 56 bytes: 4-byte function select, (arg0): 20-byte destination address, (arg1): 256-bit/32-byte amount w/ 8 decimals.
    unsigned char amount_bytes[32] {};
    print_bytes(reinterpret_cast<uint8_t*>(amount_bytes), 32, "make sure these are identical #1/2");               // <--
    
// Make function selector user-configurable. or load abi? [todo].
    memcpy(data_out, funcselect_transfer, 4);       // 
    memcpy(&data_out[4], dest_address, 20);         // arg 0:  (address)       where to send the tokens
    memcpy(&data_out[24], amount_sats256, 32);      // arg 1: (uint256/uint)  amount as bytes. the amount in satoastis (8 dec.) <--

    print_bytes(data_out, 56, "donation transfer() data field");
    *success = true;
}

bool AssembleTX(const int solution_no, const Uint256 donation_U256, uint64_t rawTx_u64[24], unsigned int* payload_length)
{ // ARGS: `t`: txview solution#: Input. If this is a transfer() call this is unused (-1: null solution no.)
  //       `donation_U256`: Input. Amount in satoastis w/ 8 dec. places. for transfer() only: 0 when minting solutions
  //        `rawTx_u64`: Output. Payload as uint64_t*.  `payload_length`: the length in bytes.

    pb_byte_t donate_addr[20]{ /* 0x57C906Fce9527C5cD9fE415FC6ECEC33A67e708E */ /* [MOVEME] */
        0x57, 0xC9, 0x06, 0xFc, 0xe9, 0x52, 0x7C, 0x5c, 0xD9, 0xfE, 0x41, 0x5F, 0xC6, 0xEC, 0xEC, 0x33, 0xA6, 0x7e, 0x70, 0x8E };
    EthereumSignTx signTx{};          // populate and sign this...
    EthereumSig sig{};                // output signature to here.  
    std::string s_txcount{ "" };  // updated now
  //uint64_t raw_tx_bytes[24]{};      // 192 bytes

    const GetParmRslt rslt_gettxcount = UpdateTxCount(gStr_SoloEthAddress, s_txcount);   // should return a valid hex str, w/out 0x, even length (pad w/ 0 if needed) <-- [wip]
    if (rslt_gettxcount == GetParmRslt::OK_PARAM_CHANGED)                                // less checks here- for clarity.
        LOG_IF_F(INFO, gVerbosity >= V_MORE, "Updated Tx count:  %s ", s_txcount.c_str());
    else if (rslt_gettxcount == GetParmRslt::OK_PARAM_UNCHANGED) { /* nothing  */ }
    else { /*  such as GetParmRslt::GETPARAM_ERROR.                            */
        LOG_F(WARNING, "AssembleTX(): Error getting Txn Count for this address while sending solution");
        return false; } //err

    signTx.nonce.size = size_of_bytes(static_cast<int>(s_txcount.length()));  // -or-  signTx.nonce.size = size_of_bytes((int)strlen(nonce)); 
    hex2byte_arr(s_txcount.c_str(), static_cast<int>(s_txcount.length()), signTx.nonce.bytes, static_cast<int>(signTx.nonce.size));     // <-- [testme]
//  hex2byte_arr(nonce, (int)strlen(nonce), signTx.nonce.bytes, signTx.nonce.size);
    if (signTx.nonce.size == 1 && signTx.nonce.bytes[0] == 0x0) {
        LOG_IF_F(INFO, DEBUGMODE, "New Ethereum account? TxCount 0, encoding placeholder byte 0x80...");
        signTx.nonce.size = 0;  // to get 0x80
    }
// ^ check the type of the input/output buffers, and length/outlength params/args in this call to hex2byte_arr()! ^
// - hex2byte_arr's output buffer type is `uint8_t` and `signTx` (type: EthereumSignTx)'s .nonce.bytes is type `pbytes_t` (uint_least8_t)? <-
// - hex2byte_arr(s_txcount.c_str(), txcount_hexstr_len, signTx.nonce.bytes, signTx.nonce.size);  // <--[old]
// === moveme & implement in ConfigSolo handler etc. ===
//std::mutex mtx_solosettings;
//std::lock_guard<std::mutex> ulock(mtx_solosettings);

// === gas price (in gwei: 1B wei) per unit gas ===
    if (gStr_GasPrice_hex.length() < 2 || gStr_GasPrice_hex.empty() || !checkString(gStr_GasPrice_hex, 0, false, true)) { /* <- use byte-arrays only where possible */
        LOG_F(ERROR, "AssembleTx(): Bad gas price."); //<-
        return false; }
    if (gStr_GasLimit_hex.length() < 2 || gStr_GasLimit_hex.empty() || !checkString(gStr_GasLimit_hex, 0, false, true)) { /* <- obviate need for 0x check. do min-max params to checkString instead. */
        LOG_F(ERROR, "AssembleTx(): Bad gas price."); //<-
        return false; }
    if (!IfEthereumAddress(gStr_ContractAddress)) {
        LOG_F(ERROR, "assembleTx(): Bad Contract address."); //<--
        return false; }

    const std::string gasprice_kludge = gStr_GasPrice_hex;  // <- could be changed while mining, use mutex? <--
    signTx.gas_price.size = size_of_bytes((int)strlen(gasprice_kludge.c_str()));  // <---
    hex2byte_arr(gasprice_kludge.c_str(), (int)strlen(gasprice_kludge.c_str()), signTx.gas_price.bytes, signTx.gas_price.size);  // <---

// - gas limit -
    const std::string gaslimit_kludge = gStr_GasLimit_hex;  // <- could be changed while mining, use mtx_coreparams? <--
    signTx.gas_limit.size = size_of_bytes((int)strlen(gaslimit_kludge.c_str()));
    hex2byte_arr(gaslimit_kludge.c_str(), (int)strlen(gaslimit_kludge.c_str()), signTx.gas_limit.bytes, signTx.gas_limit.size);

    bool b_success{ false };

// - `to` address: -
    const std::string to_kludge = Check0x_StringVer(gStr_ContractAddress, true, &b_success);
    if (!b_success) {
        LOG_F(ERROR, "Error encoding `to` field of transaction!");
        return false;
    } //condense this

    signTx.to.size = size_of_bytes((int)strlen(to_kludge.c_str())); // <- redundant?
    //hex2byte_arr(gStr_ContractAddress, (int)strlen(to), signTx.to.bytes, signTx.to.size);
    hex2byte_arr(to_kludge.c_str(), (int)strlen(to_kludge.c_str()), signTx.to.bytes, signTx.to.size);  // <--

// - value -
    signTx.value.size = 0;  // sending no eth.  we want a 'null' byte of 0x80 for value

// === data (function call w/ arguments:  mined solution nonce and its keccak256 digest) ===
 // call to mint(): the function sig and 2 params:  mined solution nonce and its keccak256 digest, uint256's, as hex-strings. W/out `0x` prefix
    b_success = false;  // <- dbg only
    std::string str_data_kludge{ "" };  /* string version */ /* [TODO] replace this. */
    if (donation_U256 == Uint256::ZERO) { /* mint() call: nothing being donated */
        str_data_kludge = RLP_EncodeData_StringVer( gTxViewItems[solution_no].str_solution, gTxViewItems[solution_no].str_digest, &b_success );
        if (b_success) {
            LOG_IF_F(INFO, DEBUGMODE, "SignTx():  initial `data` chunk is %zu bytes in size", (size_t)signTx.data_initial_chunk.size);  // <-- dbg only, remove!
            signTx.data_initial_chunk.size = size_of_bytes(static_cast<int>(strlen(str_data_kludge.c_str())));                         // <-- check 
            hex2byte_arr(str_data_kludge.c_str(), static_cast<int>(strlen(str_data_kludge.c_str())), signTx.data_initial_chunk.bytes, signTx.data_initial_chunk.size);  // <-- this!
        } else { // argh
            LOG_F(ERROR, "Error encoding `data` field of transaction!");
            return false;
        }
    }
    else { /* transfer() call: */   /* raw bytes only version */
        pb_byte_t donation_amount_bytes[32]{ 0 };
        donation_U256.getBigEndianBytes(static_cast<uint8_t[32]>(donation_amount_bytes));   // <-- legal?
        print_bytes(static_cast<uint8_t[32]>(donation_amount_bytes), 32, "Donation Amount as Bytes (Satoastis, Uint256)");  // <-- Dbg only
        RLP_EncodeData_Transfer(donate_addr, donation_amount_bytes, signTx.data_initial_chunk.bytes, &signTx.data_initial_chunk.size, &b_success);     // compacting...
          //signTx.data_initial_chunk.size = 56;
        if (!b_success) {
            LOG_F(ERROR, "Error encoding `data` field of auto-donate transaction!");
            return false;
        }
    }

    // ===sign the Tx with placeholder R, S, V=chainID===   [wip]
    LOG_IF_F(INFO, DEBUGMODE, "Creating secp256k1 context");
    secp256k1_context* secp256k1_ctx_both = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);  // for both. <----------------- CONTEXT
    if (!secp256k1_ctx_both){ return 1; }   //check for valid context <--- [checkme]

    unsigned char skbytes[32]{};
    unsigned char ecdsa_noncedata[ECDSANONCE_BYTESLEN]{};     //<- ecdsa signing nonce can be arbitrary length?  [todo]
    randombytes_buf(ecdsa_noncedata, ECDSANONCE_BYTESLEN);   //<- get random bytes. destroy this (and sk bytes if possible) after use.

    // ===SIGN, VERIFY===
    EcdsaResult signtx_result = SignTx(secp256k1_ctx_both, signTx, &sig, ecdsa_noncedata, ECDSANONCE_BYTESLEN/*, skbytes, 32*/);  // <- streamline this w/ byte-arrays. Should save on overhead using hex bytestrings <--

//  ===CLEANUP===
    randombytes_buf(ecdsa_noncedata, ECDSANONCE_BYTESLEN);	// overwrite the arbitrary data for ecdsa sign
    randombytes_buf(skbytes, 32);					// overwrite skey byte-array
    secp256k1_context_destroy(secp256k1_ctx_both);  // redundant? check <---
    // any other cleanup?  <--
    // ...

    // === sign, verify, and clean up ===
    if (signtx_result != EcdsaResult::OK)  /* if the operations did not complete */
        return false;  // err
// === end of signing stuff ===  (rem) <-

// === assemble signed, RLP-encoded transaction [TESTME] ===
    //int length = wallet_ethereum_assemble_tx( &signTx, &sig, raw_tx_bytes );  // raw tx bytes: 24*uint64_t = 192 bytes max written? (pointer). <--
    const int length = wallet_ethereum_assemble_tx(&signTx, &sig, rawTx_u64);  // raw tx bytes: 24*uint64_t = 192 bytes max written? (pointer). <--
    *payload_length = length;   // in bytes

    uint8_t rawTx_u8[256]{};                 // <--- Debug: view bytes of rawTx <---- [WIP]                                                         // DEBUGGER USE ONLY, REMOVE <---
    memcpy(rawTx_u8, reinterpret_cast<uint8_t*>(rawTx_u64), 192); // max source length 192 bytes (uint64_t*24), max destination length 256 bytes.   // for viewing raw tx bytes <--


    if (length <= 0) { /* if (length >= 1) */
        LOG_F(ERROR, "Error in wallet_ethereum_assemble_tx. Possible null pointer ");  // <-- dbg
        return false;  // Err
    }
    
    char rawTx_char[256]{};
    int8_to_char((uint8_t*)rawTx_u64, length, rawTx_char);
//
    LOG_IF_F(INFO, DEBUGMODE, "Raw tx payload bytes (not null-terminated. length: %d):  %s \n", length, rawTx_char); //<-- TEST
    print_bytes((uint8_t*)rawTx_u64, length, "Raw Tx Payload, cast from uint64_t* to uint8_t* byte-array");
//
    return true;  // OK
} //AssembleTX() - Regular version.


std::string RLP_EncodePayload_Donation(const std::string input_str) {
    // skeleton func
    // ...
    return "";  // [TODO].
}