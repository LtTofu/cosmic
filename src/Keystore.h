#pragma once
#ifndef KEYSTORE
#define KEYSTORE
#pragma message("Including KEYSTORE (" __FILE__ "), last modified " __TIMESTAMP__ ".")
#include <string>
#include <msclr/marshal.h>
#include <stdio.h>			// [old]

//#include <bitcoin-cryptography-library/cpp/Uint256.hpp>
#include <libsodium/sodium.h>
#include "util.hpp"
#include "net_solo.h"

// TODO: consolidate (see: net_solo.h/defs.h/etc.)
#define SAVE_PKEY_TOCONFIG	1
#define SAVE_PKEY_TOKEYFILE 0
#define SALT_LENGTH 32

constexpr auto PUBKEY_LENGTH = 32;				// length, in bytes, of a key used to decrypt keystore (from user-supplied password)
constexpr auto CIPHERTEXT_BYTESLENGTH = 48;		// 80;

// [REF]: constexpr auto CIPHERTEXT_STRINGLENGTH = 160;	// for ref only


ref class Keystore
{
//private:
//	Keystore(void);
//	~Keystore(void);

public:
	static std::string DecryptKeystore(const std::string& encryptedKey, const unsigned int encKeyLength, const std::string& nonceStr, 
		const std::string& passPhrase, const int kdfAlgo, const std::string& pwSalt);

public:
	static System::Boolean LoadSKeyFromConfig(const std::string str_password);

};

#else
#pragma message("Not re-including KEYSTORE.")
#endif
