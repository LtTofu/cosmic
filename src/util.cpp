// util.cpp : General helper functions
// 2020 LtTofu unless otherwise noted (see also: docs/ for addit'l license info)
#pragma once

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <fcntl.h>
#include <io.h>

#include <string>
#include <iomanip>
#include <sstream>
#include <loguru/loguru.hpp>
#include <json11/json11.hpp>
using namespace json11;

//#include "../Core/Core.h"  // optionally
#include "defs.hpp"
#include "net_solo.h"
//#include "coredefs.hpp"
extern struct txViewItem gTxViewItems[DEF_TXVIEW_MAX_ITEMS];	//<--

#define TEST_BEHAVIOR  // <--
#include "util.hpp"
//#include "types.hpp"

// todo: longer interval for updating minting addr?
unsigned int doEveryCalls_Settings[TIMINGS_COUNT]	{};  // set when CosmicWind form inits
unsigned int doEveryCalls_Values[TIMINGS_COUNT]		{};  // all to zero

//extern unsigned short gVerbosity;



std::string ParseKey_NotFound(json11::Json j_input, const int txview_itemno)
{ // called if parsing a key that was not found in provided json payload. look for `error`.
	// ^^ adapt for json11.. make sure we can pass a json object like this and how it's created. NO PROBLEM <--
	std::string errStr{ "" }, parseErrStr{ "" };// for (possible) parsing error from json11::Json::parse()
	std::string error_mesg{ "" };
	int error_code{ 0 };

	//	== parse just the key `error` to Json object ==
	// [WIP] probably a faster/more direct way to do this with json11.
	if (j_input.object_items().count("error"))
		errStr = j_input["error"].dump();						// serialized text of key `error`, if present in network response.
	 else { return "Error: no `error` key found in response "; }	// [WIP]:  see calling func! <--

	// parse the error as json object, get message and code if present:
	const auto j_error = Json::parse(errStr, parseErrStr, JsonParse::STANDARD);
	if (!parseErrStr.empty())
		return "Error parsing message/code: " + parseErrStr;		// if an errStr from Json::parse()

//  == if `error` exists in the response, get `message` string: ==
	std::string str_output {};
	if (j_error.object_items().count("message") && j_error["message"].is_string()) {
		error_mesg = j_error["message"].string_value();  // the `message` key in `error` replacing scratch contents.
		str_output = error_mesg;
	} else { domesg_verb("parsing network reply: no error message present. ", false, V_DEBUG); }  // <-- useful?

//	== and get the error code if available: ==
	if (j_error.object_items().count("code") && j_error["code"].is_number()) {
		error_code = (int)( j_error["code"].number_value() );
		str_output += " (" + std::to_string(error_code) + ")";
	} else { LOG_IF_F(WARNING, DEBUGMODE, "While parsing network reply: no error code present "); }

	if (str_output.length())	LOG_IF_F( WARNING, HIGHVERBOSITY, "Couldn't parse JSON, got `error`: %s", str_output.c_str() );
	 else	LOG_IF_F( WARNING, HIGHVERBOSITY, "Couldn't parse error from response in ParseKey_NotFound() " );
//
// exception could happen HERE		[WIP] / [FIXME]  *****
//
	gTxViewItems[txview_itemno].last_node_response = NODERESPONSE_OTHER;
	// gTxViewItems[txview_itemno].lastresponse_errstr = "?";
	// gTxViewItems[txview_itemno].lastresponse_errcode = 0 ;
	// gNodeLastError_Code = ?? ;		//
	// gNodeLastError_Mesg = ?? ;		// save to solution item in txview? or just one globally?

	if (txview_itemno >= 0)	HandleNodeError(error_mesg, error_code, txview_itemno);
	 else LOG_IF_F(INFO, DEBUGMODE, "ParseKey_NotFound(): Not calling HandleNodeError(). txview_itemno is %d.", txview_itemno);

	return str_output;
}


constexpr auto DEF_ALREADY_IN_TXVIEW = -2;

// [TODO / FIXME]: function param "expect hex" or "base10/16" switch... expect 0x or not. etc. enforce that on the result.
std::string ParseKeyFromJsonStr( const std::string& str_input, 
								 const std::string& str_keyname,			/* <-- std::string& */
								 const size_t expected_len, 
								 const int txview_itemno, 
								 bool *success )
{ // sets `success` so calling function knows if returned std::string is a parsed value, or a (hopefully useful) error message.
  // [WIP]: function param indicating what type expected? if string, the "double quotes" are removed.
	*success = false;				// [Redundant]?
	std::string str_parse_err{""};

// error checks  [TODO]: anything else?	 (possibly redundant- see caller.)
	if (!checkErr_a(str_input))
		return str_input;			// an error string (libcurl err, etc.)
	
	if (str_input.length() > DEF_JSON_PARSE_MAXLENGTH) {	 /* || str_input.find_first_not_of(DEF_JSON_PARSE_VALIDCHARS) != std::string::npos) { */
		LOG_IF_F(WARNING, DEBUGMODE, "err parsing: %s with length %zu. Too long or not JSON format.", str_input.c_str(), str_input.length());
		return "Too long or not JSON format while parsing string ";
	}

// === parse: ===
	const auto j_input = json11::Json::parse(str_input, str_parse_err, json11::JsonParse::STANDARD);  //
	if (!str_parse_err.empty()) {  // ^ REDUNDANT? 
		domesg_verb("Couldn't parse key '" + str_keyname + "' from JSON input:  " + str_input + ": " + str_parse_err, true, V_DEBUG); //<--
		return "Error parsing to json object "; }
	
	if (!j_input.object_items().count(str_keyname))			/* [TESTME] */
		return ParseKey_NotFound(j_input, txview_itemno);	/* parse out `Error`: `code` and `message`, if present.	*/
	
	std::string str_value{""};
	// [WIP]  check for unexpected "double quotes". <- 
	if (txview_itemno != DEF_ALREADY_IN_TXVIEW) {
		if (j_input[str_keyname].is_string()) { str_value = j_input[str_keyname].string_value(); }
	} else { str_value = j_input[str_keyname].dump(); } // <- in what cause would I want it in quotes?

	if (expected_len > 0 && (str_value.length() != expected_len))
		return "Value of key " + str_keyname + " has length " + std::to_string(str_value.length()) + ", expected " + std::to_string(expected_len);

	if (DEBUG_NETWORK) { domesg_verb("Parsed key `" + str_keyname + "`:  " + str_value, false, V_DEBUG); }
	*success = true;	// for the calling function. (true): txhash, etc.	(false): error string.
	return str_value;	// ok: the value of the key
}

std::string AddHexSpecifierIfMissing(const std::string inputStr)
{
	if (inputStr.length() >= 2) {
		if (inputStr.substr(0, 2) != "0x")
			return "0x" + inputStr;
	}
	return inputStr;  // ... or return input string unchanged
}

// TODO: optimize this function.
bool checkString( const std::string& theStr, const size_t expectedLength, 
	const bool requireHexSpecifier, const bool expectHex )
{
	if (!gSoloMiningMode)
	{ // pool mode: check length before parsing, rejecting things like cloudflare error pages
		if (theStr.length() > DEF_POOL_MAX_PARSED_LENGTH) { /* <-- not relevant in generic use (other than pool requests) */
			if (gVerbosity == V_DEBUG) { printf("checkString(): pool response too long \n"); }
			LOG_IF_F(WARNING, gVerbosity==V_DEBUG, "checkString(): input is too long. expected %zu, got %zu, max %d",
				expectedLength, theStr.length(), DEF_POOL_MAX_PARSED_LENGTH); // <-
			return false;
		}
	}

	if (theStr.empty()) {
		LOG_F(WARNING, "checkString(): empty string");
		return false; }

	if (theStr.length() >= 5) { /* don't out-of-range if string length is <5 */
		if ((theStr[0] == 'E' || theStr[0] == 'e') && theStr[1] == 'r' && theStr[2] == 'r' && theStr[3]=='o' && theStr[4]=='r') {
			LOG_IF_F(WARNING, DEBUGMODE, "CheckError(): %s ", theStr.c_str());
			return false;
		}
	}

	if (expectedLength) {						 /* if 0, no specific length is expected	*/
		if (theStr.length() != expectedLength) {
#ifdef TEST_BEHAVIOR								 /* show the input string in the log		*/
			LOG_F(WARNING, "checkString(): input: %s  has length %zu, expected length %zu.", theStr.c_str(), theStr.length(), expectedLength);
#endif
			domesg_verb("Network response length is " + std::to_string(theStr.length()) + ", expected " + std::to_string(expectedLength), false, V_DEBUG);
			return false;
		}
	}
	
	// check for `0x` (if string is >=2 
	bool has_0x{ false };
	if (theStr.length() >= 2) {
		has_0x = (theStr[0] == '0' && theStr[1] == 'x');
		if (has_0x && !requireHexSpecifier) {
			LOG_F(ERROR, "checkString(): unexpected hex specifier `0x`");	// <-	version of this function which logs 'WARNING'
			return false; }
		if (!has_0x && requireHexSpecifier) { //		returns an adjusted string with 0x added?	[todo].
			LOG_F(ERROR, "checkString(): expected hex specifier `0x`");		// <-	version of this function which logs 'WARNING'
			return false; }
	}

	if (has_0x && theStr.length() < 3) {
		LOG_F(ERROR, "CheckString():  hex specifier `0x` followed by no data!");	// <- abort. needs to be addressed, if it happens.
		return false; }

	//std::string temp_str{theStr};
	//if (temp_str.substr(0, 2) == "0x")
	//	temp_str = temp_str.substr(2);	// remove 0x specifier from string (if present)

	// check the length first? Don't out-of-range.
	if (expectHex) { /* input must be hexadecimal.  [todo]: check evenness here too? redundant? */
		if (theStr.find_first_not_of(DEF_HEXCHARS, 2) != std::string::npos) { /* search from offset:2 (after `0x`) */
			LOG_F(WARNING, "illegal character(s) in expected hex: %s ", theStr.c_str());				// 0-9, A-F, a-f
			return false; }		// expect base16 in string form
	} else { /* base10 decimal: */
		if (theStr.find_first_not_of(DEF_NUMBERS) != std::string::npos) {		/* read the number from character */
			LOG_F(WARNING, "illegal character(s) in expected base10 decimal: %s ", theStr.c_str());	// 0-9
			return false; }		// expect base10 in string form
	}

	return true;  // meets requirements
}

bool checkErr_b(std::string* io_str, const bool trim_error)
{ // returns false if string is an error or empty "". returns true if NOT an error.
	if (io_str->empty())
		return false;  // empty string, considered an error

	if (io_str->length() >= 5)
	{ // ^ don't out-of-range getting substring
		if (io_str->substr(0, 5) == "Error")
		{
			if (trim_error) { *io_str = io_str->substr(7); }  // trim off "Error: " (incl. the space)
			return false;		// string is an error
		}
		else { return true; }	// not an error
	}
	else { return true; } // short string no longer treated as an error  (TESTME, see _a() version also)
}

bool checkErr_a(const std::string& str_in)
{ // returns false if string is an error or empty "". returns true if NOT an error.
	if (str_in.empty()) {
		LOG_IF_F(WARNING, gVerbosity >= V_MORE, "checkErr_a(): empty input string. ");
		return false; }  // empty string, considered an error
	
	if (str_in.length() >= 5)
	{ // false: string is an error.
		if (str_in.substr(0, 5) == "Error") {
			domesg_verb(/*"checkErr_a():" + */ str_in, false, V_NORM);							  // add mining event for listbox_events
			LOG_IF_F( WARNING, gVerbosity >= V_MORE, "%s ", str_in.c_str() );  // write to log if higher verbosity levels
			return false;
		} else { return true; }  // true: OK
	}
	else { return true; }  // OK
}

// CHECKDOEVERYCALLS: manages the timing for certain events. Easily checked, returns bool (whether the calling function
//					  should perform the operation or not. If returning false, the matching counter is iterated in _Values[]
//					  If returning true, reset the counter. (todo: any speed benefit to inline?)
bool checkDoEveryCalls(const unsigned short whichEvent)
{
	if (doEveryCalls_Values[whichEvent] >= doEveryCalls_Settings[whichEvent]) {
		doEveryCalls_Values[whichEvent] = 0;	// reset
		return true; }
	else {
		doEveryCalls_Values[whichEvent] += 1;	// iterate.
		return false; }
}

bool IfEthereumAddress(const std::string inputStr)
{	// returns true if input string appears to be a valid Ethereum-style address. otherwise false.
	// (todo) verify checksum via capitalization?  [TODO]
	//if (inputStr.length() <= 2)  return false;
	if (inputStr.length() != 42)  // todo: accept 40-length if no `0x`
		return false;
	if (!(inputStr[0]=='0' && inputStr[1]=='x'))  // must start with hex specifier
		return false;
	if (inputStr.substr(2).find_first_not_of("0123456789ABCDEFabcdef") != std::string::npos)
		return false;  // must contain only hex characters after the hex specifier

	return true;
}

//
// === conversion functions ===
//
template<typename T>
auto Different_BytesToString(T const buffer) -> std::string const
{
	std::string output;
	output.reserve(buffer.size() * 2 + 1);

	for (auto byte : buffer)
		output += ascii[byte];

	return output;
}

// from: 0xBitcoin-miner
static uint8_t fromAscii(uint8_t c)
{
	if (c >= '0' && c <= '9')
		return (c - '0');
	if (c >= 'a' && c <= 'f')
		return (c - 'a' + 10);
	if (c >= 'A' && c <= 'F')
		return (c - 'A' + 10);
#if defined(__EXCEPTIONS) || defined(DEBUG)
	throw std::runtime_error("fromAscii(): invalid character. ");
#else
	return 0xff;
#endif
}

// from: 0xBitcoin-miner
static uint8_t ascii_r(uint8_t a, uint8_t b)
{
	return fromAscii(a) * 16 + fromAscii(b);
}

//
bool HexToBytes_b(const std::string& theHex, uint8_t *bytes_out)
{ // HEXTOBYTES: Converts std::string `theHex` to uint8_t `byteArray` of variable length
	if (theHex.length() % 2) {
		LOG_F(ERROR, "uneven length %zu input to HexToBytes()", theHex.length());
		return false; }
	if (theHex.find_first_not_of(DEF_HEXCHARS) != std::string::npos) {
		LOG_F(ERROR, "non-hex character(s) in input to HexToBytes()");
		return false; }

	for (std::string::size_type i=0, j=0; i < theHex.length(); i += 2, ++j)
		bytes_out[j] = ascii_r(theHex[i], theHex[i + 1]);  // Thx: 0xBitcoin-Miner!

	return true;
}

// multiple approaches, not all used. settle on one, though (TODO)
std::string uint8t_array_toHexString(const uint8_t* data, const int len)
{
	std::stringstream ss;
	ss << std::hex;
	for (int i = 0; i < len; ++i)
		ss << std::setw(2) << std::setfill('0') << (int)data[i];
	return ss.str();
}


//
//bool Check0x(std::string& in_str) /* FIXME */
bool Check0x(std::string& in_str) /* FIXME */
{ // ...
	if (in_str.empty())
		return false;  // string can't be empty

	if (in_str[0] == '0' && in_str[1] == 'x')
	{ // string begins with 0x hex specifier:
		if (in_str.length() < 3)
			return false;  // not suitable: `0x` with no data following it.
//		if (in_str.length > x)
//			return false;  // not suitable: string exceeds maximum length of `x`.
//		if(in_str.find_first_not_of(DEF_HEXCHARS) != std::string::npos)
//			return false;  // not suitable: non-hex character(s).
		return true;  // looks good
	}
	// specifier not present, expect hex:
	if (in_str.find_first_not_of(DEF_HEXCHARS) != std::string::npos)
		return false;  // not suitable: non-hex character(s).
	if (in_str.length() % 2 != 0)
	{
		LOG_IF_F(INFO, gVerbosity == V_DEBUG, "CheckFor0x: padding uneven hex string");
		in_str = "0" + in_str;
	}
	return true;


	// ...

	return false;
}

//
std::string Check0x_StringVer(const std::string& in_str, const bool trim_0x, bool *success) /* FIXME */
{ // ...
	bool has_0x {false};
	if (in_str.empty())
		return "";  // string can't be empty

	if (in_str[0] == '0' && in_str[1] == 'x')
	{ // string begins with `0x`:
		if (in_str.length() < 3)
			return "";  // not suitable: `0x` with no data following it.
		has_0x = true;
//		if (in_str.length > `x`)
//			return false;  // not suitable: string exceeds maximum length of `x`.
//		if(in_str.find_first_not_of(DEF_HEXCHARS) != std::string::npos)
//			return false;  // not suitable: non-hex character(s).
	}

	// `0x` specifier not present, expect hex:
	if (has_0x) {
		if (in_str.find_first_not_of(DEF_HEXCHARS, 2) != std::string::npos)	/* check after 0x */
			return "";  // err
	} else {
		if (in_str.find_first_not_of(DEF_HEXCHARS) != std::string::npos) /* check all characters */
			return "";  // err
	}
	
	if (in_str.length() % 2 != 0) {	/* === pad with 0 as needed for even length. min length: 1 byte | 2 characters. */
		LOG_IF_F(INFO, gVerbosity==V_DEBUG, "CheckFor0x: padding uneven hex string");
		*success = true;				// <- output from the function is good.
		return "0" + in_str;
	}


// [checkme]
	if (in_str.length() < 2) { /* jic, shouldn't happen. ensure appropriate length before .substr() below. */
		LOG_F(ERROR, "Check0x_StringVer(): unexpected hex string, returning null.");
		return "";	// output is null.
	}
// [checkme]

	*success = true;				// <- output from this function is good.
	if (has_0x && trim_0x)
		return in_str.substr(2);	// return characters after 0x
	 else { return in_str; }		// or return string as-is.
}


// Converts a C-style string (char array) of Hexadecimal characters into an array of bytes represented as uint8_t
// [WIP]: Phasing out
uint8_t* hexstr2uint8(const char* cstring)
{
	//if (gVerbosity == V_DEBUG)  printf("- DEBUG: hexstr2uint8(): input string %s \n", cstring);
	if (cstring == NULL) {//<--
		printf("error: input string to hexstr2uint8() is null \n");
		return NULL; }  //<--

	//const std::string hexCheck = std::string(cstring);
	if ( std::string(cstring).find_first_not_of("0123456789abcdefABCDEF") != std::string::npos) {
		printf("error: hexstr2uint8(): input contains illegal character(s).  problem string: %s \n", cstring);
		return NULL; }  //<--

	size_t slength = strlen(cstring);
	if ((slength % 2) != 0) {
		printf("error: input string to hex_str_to_uint8() not even \n");  // pad with zero?
		return NULL; }  //<--

	size_t dlength = slength / 2;
	uint8_t* data = (uint8_t*)malloc(dlength);  // <--- req's freeing
	memset(data, 0, dlength);  // fill with dlength of zeroes

	size_t index = 0;
	while (index < slength) {
		char c = cstring[index];
		int value = 0;
		if (c >= '0' && c <= '9')
			value = (c - '0');
		else if (c >= 'A' && c <= 'F')
			value = (10 + (c - 'A'));
		else if (c >= 'a' && c <= 'f')
			value = (10 + (c - 'a'));
		else {
			LOG_IF_F(ERROR, gVerbosity>=V_MORE, "error: invalid character in string, in hex_str_to_uint8()\n");  // <- Verify no calls actually produce this log message,
			return NULL; //<--																					 //	   even if the input to the function is bad.  [TESTME] <--
		}
		data[(index / 2)] += value << (((index + 1) % 2) * 4);
		index++;
	}
	// Impt. Note: ensure malloc'd space is freed this function or
	//			   immediately after. this function being phased out
	return data;
}


// Converts a C-style string (char array) of Hexadecimal characters into an array of `unsigned char` of size `out_buflen` bytes. Pass in a cstr _without_ `0x` prefix.
bool cstring_to_uchar_array ( const char* cstring, const size_t bytes_len, unsigned char* out_buf )
{ // [CHECKME]
	if (gVerbosity == V_DEBUG) { printf("- DEBUG: cstring_to_uchar_array(): input string %s \n", cstring); }  //<---

  // verify length, check against expected (2 hex digits characters input = 1 byte output)
	const size_t slength = strlen(cstring);
	const size_t dlength = slength / 2;						/* 2 hex digits to a byte.		*/
	if ( (slength % 2) != 0 || slength != (bytes_len*2)) {
		LOG_IF_F(ERROR, gVerbosity>V_NORM, "cstring_to_uchar_array(): input hexstr with uneven or unexpected length %zu", strlen(cstring));
		return false; }  //err

	if (dlength != bytes_len) {								/* <-- redundant? see above.	*/
		LOG_IF_F(ERROR, gVerbosity>V_NORM, "cstring_to_uchar_array(): input hexstr with uneven bytes length %zu", dlength);
		return false; }
	
//	input cstr must be appropriate length, and not null.
	if (!cstring) {
		LOG_IF_F(ERROR, gVerbosity>V_NORM, "cstring_to_uchar_array():  null input or unexpected length");
		return false; }  //err

//	check for illegal (non-hex) characters.  [TODO/FIXME]:  for speed, obviate need for this conversion to std::string.
	const std::string hex_check = std::string(cstring); // <--
	if (hex_check.find_first_not_of("0123456789abcdefABCDEF") != std::string::npos) {
		LOG_IF_F(ERROR, gVerbosity>V_NORM, "hex_str_to_uchar_array():  illegal character(s) in input hexstr");
		printf("problem string follows: %s \n", cstring);
		return false; }  //err
	// ^


	unsigned char *data = (unsigned char*)(malloc(dlength));  // <--- requires freeing! <--
	memset( (void*)data, 0, dlength );

	size_t index = 0;
	while (index < slength) {
		char c = cstring[index];
		int value = 0;
		if (c >= '0' && c <= '9')		{ value = (c - '0'); }
		 else if (c >= 'A' && c <= 'F')	{ value = (10 + (c - 'A')); }
		 else if (c >= 'a' && c <= 'f')	{ value = (10 + (c - 'a')); }
		 else {
			printf("error: invalid character in string, in hex_str_to_uchar_array()\n");
			return NULL; }
		//
		data[(index / 2)] += value << (((index + 1) % 2) * 4);
		++index;
	}
	memcpy((void*)out_buf, data, dlength);
	free(data);  // important!
	return true;  // OK
}


//
std::string HexBytesToStdString(const unsigned char* inBuf, unsigned short length)  /* <- make 2nd param `const` */
{  // HEXBYTESTOSTDSTRING: Takes an unsigned char buffer of `length` bytes long and returns a std::string
   //					   of the hexadecimal bytes representation, keep correct length/zeroes
	std::stringstream sst;
	for (unsigned short i = 0; i < length; ++i)
		sst << std::hex << std::setw(2) << std::setfill('0') << (unsigned short)inBuf[i]; // <- why the cast? [TODO]
	return sst.str();
}

// HEXBYTESTOSTDSTRING: version whose first param takes a uint8_t array instead
std::string HexBytesToStdString_u8(const uint8_t* inBuf, const unsigned short length)
{
	std::stringstream sst;
	for (unsigned short i = 0; i < length; ++i)
		sst << std::hex << std::setw(2) << std::setfill('0') << inBuf[i];
	return sst.str();
}


// New
// [WIP]: check control paths
// Converts a std::string of hex (without `0x` prefix) into a byte array (of type `unsigned char`, and size `out_buflen` bytes)
bool std_string_to_uchar_array(const std::string hex_string, const size_t bytes_len, unsigned char* out_buf)
{ // [CHECKME]
	if (gVerbosity == V_DEBUG)
		printf("- DEBUG: std_string_to_uchar_array():  input string %s \n", hex_string.c_str());
	//
	// verify length, check against expected (2 hex digits characters input = 1 byte output)
	const size_t slength = hex_string.length();
	if (((slength % 2) != 0) || slength != (slength * 2)) {
		LOG_IF_F(ERROR, gVerbosity > V_NORM, "std_string_to_uchar_array():  input hexstr with uneven or unexpected length %zu", hex_string.length());
		return false;
	} //err

	const size_t dlength = slength / 2;						/* 2 hex digits to a byte.		*/
	if (dlength != bytes_len) {								/* <-- redundant? see above.	*/
		LOG_IF_F(ERROR, gVerbosity > V_NORM, "std_string_to_uchar_array():  input hexstr with uneven bytes length %zu", dlength);
		return false;
	}

	//	input cstr must be appropriate length, and not null.
	if ( hex_string.empty() || hex_string.length() != dlength) {
		LOG_IF_F(ERROR, gVerbosity > V_NORM, "std_string_to_uchar_array():  null input or unexpected length");
		return false;
	} //err

//	check for illegal (non-hex) characters
	if (hex_string.find_first_not_of("0123456789abcdefABCDEF") != std::string::npos) {
		LOG_IF_F(ERROR, gVerbosity > V_NORM, "std_string_to_uchar_array():  illegal character(s) in input hexstr");
		return false;
	} //err
// ^


	unsigned char* data = (unsigned char*)(malloc(dlength));  // free this
	memset((void*)data, 0, dlength);

	size_t index = 0;
	while (index < slength) {
		char c = hex_string[index];
		int value = 0;
		if (c >= '0' && c <= '9') { value = (c - '0'); }
		else if (c >= 'A' && c <= 'F') { value = (10 + (c - 'A')); }
		else if (c >= 'a' && c <= 'f') { value = (10 + (c - 'a')); }
		else {
			printf("error: invalid character in string, in hex_str_to_uchar_array()\n");
			return NULL;
		}
		//
		data[(index / 2)] += value << (((index + 1) % 2) * 4);
		++index;
	}
	memcpy((void*)out_buf, data, dlength);
	memset((void*)data, 0, dlength); // <------ new. use in equivalent functions too? or overwrite w/ random bytes?
	free(data);  // freed here
	return true;  // ok
}

void print_bytes(const uint8_t inArray[], const uint8_t len, const std::string desc)  /* uint8_t* inArray? */
{  // PRINT_BYTES: Prints a formatted table of bytes `len` long, 8 bytes/row, to stdout
	if (!DEBUGMODE || !DEBUG_PRINTBYTES) { return; }

	if (desc != "") { printf("%s  %d bytes (big-endian order): \n", desc.c_str(), len); }
	uint8_t i, lnbrk_ctr{ 0 };
	for (i = 0; i < len; ++i) {				// prints a formatted table of bytes
		++lnbrk_ctr;
		printf("%02x ", inArray[i]);		// print w/ leading zero as needed
		//
		if (lnbrk_ctr >= 8) {
			printf("\n");					// line break every 8 bytes for readability
			lnbrk_ctr = 0;
		}
	}
	printf("\n");
}

void print_bytes_uchar(const unsigned char inArray[], const size_t len, const std::string desc)  /* [NEW].  uint8_t* inArray? */
{  // PRINT_BYTES: Prints a formatted table of bytes `len` long, 8 bytes/row, to stdout. Input: `unsigned char*` with length `len`.
	if (!DEBUGMODE)	{ return; }
	if (desc != "")
		printf("%s  %zu bytes (big-endian order): \n", desc.c_str(), len);  // description first
	unsigned short i, lnbrk_ctr{ 0 };
	for (i = 0; i < len; ++i) {				// prints a formatted table of bytes
		++lnbrk_ctr;
		printf("%02x ", inArray[i]);		// print w/ leading zero as needed
		//
		if (lnbrk_ctr >= 8) {
			printf("\n");					// line break every 8 bytes for readability
			lnbrk_ctr = 0;
		}
	}
	printf("\n");
}

void print_Uint256(const Uint256 input, const bool as_hexstr, const bool as_bytes)
{ // (for debugging purposes.) Prints a Uint256 to stdout as hex
  // string or formatted table of bytes, in big endian.
	if (gVerbosity < V_DEBUG) { return; }

	uint8_t tempbytes[32]{};
	input.getBigEndianBytes(tempbytes);	// out to byte array
	if (as_hexstr) {
		printf("0x");
		for (unsigned short i = 0; i < 32; ++i) {
			printf("%02x", tempbytes[i]);
		}
		printf("\n");
	}

	if (as_bytes) { print_bytes(tempbytes, 32, "Uint256"); }	//a neat table of bytes
}


// DOMESG_VERB: verbosity-based. adds a message to the mining events log in the main window.
// (phasing out in favor of more elegant approach)
/*_inline*/ void domesg_verb(const std::string& to_print, const bool make_event, const unsigned short req_verbosity)
{ // for messages to the user which don't need logging.
	if (gVerbosity < req_verbosity)  return;
	printf("%s \n", to_print.c_str());
	if (make_event) { AddEventToQueue(to_print); }		// to Events View w/ timestamp
}

bool GetEventStr(std::string* str_out)
{ //
	const std::lock_guard<std::mutex> lock(mtx_events);
	if (q_events.empty())
		return false;				// no events to get

	*str_out = q_events.front();	// the event text for calling function
	q_events.pop();					// frontmost event removed from queue
	return true;					// a mining event's text was retrieved
}


constexpr auto DEBUGGING_EVENTLOG = 1;		// Debug Only!
//
void ClearEventsQueue(void)
{ // Ensures the events queue for the "Mining Events" listbox is empty and initialized. (may be slightly redundant.)
	std::lock_guard<std::mutex> lock(mtx_events);  // must get 
	LOG_IF_F(INFO, DEBUGMODE, "Clearing event log queue (contains %zu items).", q_events.size());	// <-- remove this

	size_t howmany{ 0 };									// <-- this
	while (!q_events.empty()) {
		if (DEBUGMODE && DEBUGGING_EVENTLOG) { ++howmany; }	//<-- this
		q_events.pop();
	}
	LOG_IF_F(INFO, DEBUGMODE && DEBUGGING_EVENTLOG, "Popped %zu events from queue.", howmany);		// <-- and this.
}


void AddEventToQueue(const std::string theText)
{ // push string into `q_events` (fifo queue) for display in `listbox_events`
	std::lock_guard<std::mutex> lock(mtx_events);
	theText.length() > 120 ? q_events.push(theText.substr(0, 120)) : q_events.push(theText);  // max len 120 (adjust this)
}


// from: net_rlp_utils.cpp
std::string bytes_to_string(const uint8_t* byteArray, const unsigned short length)
{ // Convert array of bytes (uint8_t's) to std::string for use by the RLP_ funcs in this file.
	std::stringstream sst;
	sst << std::hex << std::setfill('0');
	for (unsigned short i = 0; i < length; ++i)
		sst << std::setw(2) << static_cast<unsigned>(byteArray[i]);

	if (gVerbosity == V_DEBUG) { printf("bytestostring:  %s", sst.str().c_str()); }
	return sst.str();
}

//std::string Uint256_to_StdString(const Uint256 input256)
//{
//	unsigned char bytes_scratch[32]{};
//	input256.getBigEndianBytes(bytes_scratch);
//	return HexBytesToStdString(bytes_scratch, 32);
//}


void toHex(void* data, const size_t dataLength, std::string & destStr)
{ // Converts a block of data to a hex string  (Credit: tweex.net)!
	unsigned char* byteData = reinterpret_cast<unsigned char*>(data);
	std::stringstream hexStringStream;

	hexStringStream << std::hex << std::setfill('0');
	for (size_t index = 0; index < dataLength; ++index)
		hexStringStream << std::setw(2) << static_cast<int>(byteData[index]);
	destStr = hexStringStream.str();
}

