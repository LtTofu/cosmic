// util.hpp : General helper functions
// 2020 LtTofu unless otherwise noted (see also: docs/ for addit'l licenses.)
#pragma once

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <fcntl.h>
#include <io.h>

#include <string>
#include <loguru/loguru.hpp>
#include <bitcoin-cryptography-library/cpp/Uint256.hpp>
#include <iomanip>
//#include <sstream>

#include "defs.hpp"


#define TEST_BEHAVIOR  // <--

// [ MOVEME ]
#define DEF_POOL_MAX_PARSED_LENGTH 200	// was 150
#define DEF_JSON_PARSE_MAXLENGTH 2000  // raised to accommodate transaction receipts


//std::queue<std::string> gEventsQueue;

// TODO: consider moving these to Core. and replacing with time values
extern unsigned int doEveryCalls_Settings[TIMINGS_COUNT];
extern unsigned int doEveryCalls_Values[TIMINGS_COUNT];


std::string AddHexSpecifierIfMissing(	const std::string inputStr	);

// TODO: optimize this function.
bool checkString(const std::string& theStr, 
				const size_t expectedLength, 
				const bool requireHexSpecifier, 
				const bool expectHex );

bool checkErr_b ( std::string* io_str, const bool trim_error );

bool checkErr_a ( const std::string& str_in );

bool checkDoEveryCalls ( const unsigned short whichEvent );

bool IfEthereumAddress (const std::string inputStr );

// conversion functions
// Thx: Infernal Toast, Zegordo, authors of 0xbitcoin-miner!

std::string uint8t_array_toHexString(	const uint8_t* data, const int len	);

bool HexToBytes_b(	const std::string& theHex, uint8_t* bytes_out	);

uint8_t* hexstr2uint8(	const char *cstring	);  // [old]

bool cstring_to_uchar_array(	const char* cstring, 
								const size_t bytes_len, 
								unsigned char* out_buf	);	// <- [New]: Outputs to uchar array `out_buf` of length `out_buflen`.  [Note]: the length of `cstring` in hex digits/characters
																									//			  should be twice the size of `out_buflen` in bytes  (2 hex digits=1 byte).
																									//    [example]:	Input "a0b1c2" (len: 6 chars) writes 3 bytes to output buffer:  { 0xa0, 0xb1, 0xc2 }.

std::string HexBytesToStdString( const unsigned char* inBuf, 
								 unsigned short length	);  // stringstream version

std::string HexBytesToStdString_u8(	const uint8_t* inBuf,
									const unsigned short length	);  // ditto, uint8_t array param

std::string ParseKeyFromJsonStr( const std::string& str_input,
								 const std::string& str_keyname,
								 const size_t expected_len,
								 const int txview_itemno,
								 bool *success );

/*__inline*/ bool Check0x ( std::string& strIn );  // TESTME

std::string Check0x_StringVer(	const std::string& in_str, 
								const bool trim_0x, 
								bool *success	);  // [FIXME]?

// from net_rlp_utils.hpp:
void toHex(void* data, const size_t dataLength, std::string& destStr);

//std::string Uint256_to_StdString(const Uint256 input256);

std::string bytes_to_string(const uint8_t* byteArray, const unsigned short length);


static const char* const ascii[] = {
	"00","01","02","03","04","05","06","07","08","09","0a","0b","0c","0d","0e","0f",
	"10","11","12","13","14","15","16","17","18","19","1a","1b","1c","1d","1e","1f",
	"20","21","22","23","24","25","26","27","28","29","2a","2b","2c","2d","2e","2f",
	"30","31","32","33","34","35","36","37","38","39","3a","3b","3c","3d","3e","3f",
	"40","41","42","43","44","45","46","47","48","49","4a","4b","4c","4d","4e","4f",
	"50","51","52","53","54","55","56","57","58","59","5a","5b","5c","5d","5e","5f",
	"60","61","62","63","64","65","66","67","68","69","6a","6b","6c","6d","6e","6f",
	"70","71","72","73","74","75","76","77","78","79","7a","7b","7c","7d","7e","7f",
	"80","81","82","83","84","85","86","87","88","89","8a","8b","8c","8d","8e","8f",
	"90","91","92","93","94","95","96","97","98","99","9a","9b","9c","9d","9e","9f",
	"a0","a1","a2","a3","a4","a5","a6","a7","a8","a9","aa","ab","ac","ad","ae","af",
	"b0","b1","b2","b3","b4","b5","b6","b7","b8","b9","ba","bb","bc","bd","be","bf",
	"c0","c1","c2","c3","c4","c5","c6","c7","c8","c9","ca","cb","cc","cd","ce","cf",
	"d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","da","db","dc","dd","de","df",
	"e0","e1","e2","e3","e4","e5","e6","e7","e8","e9","ea","eb","ec","ed","ee","ef",
	"f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","fa","fb","fc","fd","fe","ff"
};

static uint8_t fromAscii( uint8_t c );

static uint8_t ascii_r( uint8_t a, uint8_t b );

//
void print_bytes(const uint8_t inArray[], const uint8_t len, const std::string desc);				// uint8_t* inArray?

//
void print_bytes_uchar(const unsigned char inArray[], const size_t len, const std::string desc);	// uint8_t* inArray?

//
bool std_string_to_uchar_array (const std::string hex_string, const size_t bytes_len, unsigned char* out_buf);  // [CHECKME]

//
void ClearEventsQueue(void);

//
bool GetEventStr(std::string* str_out);

//
void AddEventToQueue(const std::string theText);

//
void print_Uint256(const Uint256 input, const bool as_hexstr, const bool as_bytes);

//phasing out
/*_inline_*/ void domesg_verb (const std::string& to_print, const bool make_event, const unsigned short req_verbosity);
