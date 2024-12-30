#include "Keystore.h"

using namespace System;	//
using namespace System::Windows::Forms;
using namespace System::Configuration;
#include <msclr/marshal.h>
#include <msclr/marshal_cppstd.h>
//#include <msclr/marshal_windows.h>

bool isLibSodiumInited(void)
{ // [TODO/FIXME]: COSMiC should close if LibSodium or LibCurl can't init. never seen this error!
	if (gLibSodiumStatus == -1) {
		LOG_F(ERROR, "LibSodium not initialized (status: %d). Aborting EncryptKeystore().", gLibSodiumStatus);
		MessageBox::Show("LibSodium could not be initialized. Please restart the application, or re-extract COSMiC if you receive this error again.",
			"COSMiC", MessageBoxButtons::OK, MessageBoxIcon::Asterisk);
		return false;
	}

	return true;  // OK
}


// === member functions of class Keystore ===

std::string Keystore::DecryptKeystore(const std::string& encryptedKey, const unsigned int encKeyLength, 
	const std::string& nonceStr, const std::string& passPhrase, const int kdfAlgo, const std::string& pwSalt)
{ // libsodium init done at program start (see Cosmic.cpp):															// fixme: unsigned char array? ^
	if (!isLibSodiumInited())
		return "Error: Libsodium not inited";

	//
	// [ WIP / FIXME]:  get the passphrase from the relevant SecureString, then clear it :)          <-----

	// The encrypted key (and info required to decrypt it, except for passphrase) are stored in the Configuration (Cosmic.exe.Config)
	//		as base16 (hex) text representing byte arrays.  "Should" be 96 characters long (48 bytes):  a 32-byte message, and 16-byte authentication tag.

// Debug only, Remove this
	const size_t size_of_encrypted_message_in_config = encryptedKey.length();			// <-- CHECK ME in debugger ! <--------------
// Debug only, Remove this
	if (encryptedKey.length() != (CIPHERTEXT_BYTESLENGTH * 2))
		if (encryptedKey.length() != (CIPHERTEXT_LEN * 2))
			return "Error decrypting account:  encryptedPK has unexpected length";
	if (nonceStr.length() != (crypto_secretbox_xchacha20poly1305_NONCEBYTES * 2))
		return "Error decrypting account:  encryptedPKnonce has unexpected length";

	unsigned char nonce_bytes[crypto_secretbox_xchacha20poly1305_NONCEBYTES]{};
	unsigned char salt_bytes[32]{};
	unsigned char ciphertext_bytes[CIPHERTEXT_BYTESLENGTH]{};									// decrypting the message ENCRYPTED_MESSAGE_LEN long...
	cstring_to_uchar_array(nonceStr.c_str(), crypto_secretbox_xchacha20poly1305_NONCEBYTES, nonce_bytes);		// convert stored nonce to byte-array
	cstring_to_uchar_array(pwSalt.c_str(), SALT_LENGTH, salt_bytes);							// string of hex bytes to `unsigned char` byte-array
	cstring_to_uchar_array(encryptedKey.c_str(), CIPHERTEXT_BYTESLENGTH, ciphertext_bytes);
	//
	unsigned char decrypted_bytes[MESSAGE_LEN]{};  // <-- CIPHERTEXT_LEN? crypto_secretbox_xchachapoly1305_MACBYTES + MESSAGE__LEN? actual key is 0x+64 hex digits=66. 2 characters to a byte.
	unsigned char key_bytes[32]{};
	//-dbg-
	print_bytes((uint8_t*)nonce_bytes, crypto_secretbox_xchacha20poly1305_NONCEBYTES, "decryption nonce"); // DBG: print formatted table of bytes (DEBUG verbosity only.)
	print_bytes((uint8_t*)salt_bytes, SALT_LENGTH, "salt");										// DBG: print formatted table of bytes (DEBUG verbosity only.)
	//print_bytes((uint8_t*)ciphertext_bytes, CIPHERTEXT_BYTESLENGTH /*CIPHERTEXT_LEN*/, "ciphertext (zeroes)");	// DBG 
	//print_bytes((uint8_t*)decrypted_bytes, MESSAGE_LEN, "decrypted mesg (key)");				// DBG: expect this to be all zeroes
	print_bytes((uint8_t*)key_bytes, 32, "decrypted mesg (key)");								// DBG: this too.
	printf("passphrase: %s,  len: %zu.  IMPT: is length expected? \n", passPhrase.c_str(), passPhrase.length());	// DBG. MOVEME ?
//-dbg-

// derive key from user-provided password:
	LOG_IF_F(INFO, gVerbosity == V_DEBUG, "Computing key from passphrase.");  // out to `key`.
	//if (DeriveKeyFromPassPhrase_WinSDK(reinterpret_cast<const unsigned char*>(passPhrase.c_str()), passPhrase.length(), 
	if (DeriveKeyFromPassPhrase_WinSDK(/*reinterpret_cast<const unsigned char*>(*/passPhrase.c_str()/*)*/, passPhrase.length(),
		reinterpret_cast<unsigned char*>(key_bytes), reinterpret_cast<unsigned char*>(salt_bytes), false) != 0)  /* `false`: don't generate new salt: it's passed in. */
	{ // ^ [TESTME]:  convert passphrase to ascii bytes first? ^ [TESTME]
		printf("Error deriving key from passphrase. \n");
		return "Error deriving key from passphrase ";
	}

	//const unsigned char *encrypted = (const unsigned char*)hexstr2uint8(encryptedKey.c_str());
	//unsigned char decrypted[MESSAGE_LEN] = { 0 };  // <-- CIPHERTEXT_LEN? crypto_secretbox_xchachapoly1305_MACBYTES + MESSAGE__LEN? actual key is 0x+64 hex digits=66. 2 characters to a byte.

	LOG_IF_F(INFO, HIGHVERBOSITY/*DEBUGMODE?*/, "Decrypting stored account from Config");
	if (crypto_secretbox_xchacha20poly1305_open_easy(decrypted_bytes, ciphertext_bytes, CIPHERTEXT_LEN, nonce_bytes, key_bytes)) {
		// nonzero result: error
		LOG_IF_F(WARNING, HIGHVERBOSITY, "Couldn't decrypt stored account. Bad key?");
		return "Error: decrypt failed, probable bad key ";
	}
//
	print_bytes(reinterpret_cast<uint8_t*>(decrypted_bytes), MESSAGE_LEN, "decrypted");  // (debug mode only).  print_bytes() expects uint8_t bytearray <--
	printf("decrypted:  %s   (bytes length:  %d) \n", HexBytesToStdString(decrypted_bytes, MESSAGE_LEN).c_str(), MESSAGE_LEN); // <-- dbg
//
//	...return "Error: decrypt succeeded with unexpected result ";
	const std::string str_scratch = HexBytesToStdString(decrypted_bytes, MESSAGE_LEN);
	if (gVerbosity == V_DEBUG)								//   DBG
		printf("str_scratch:  %s \n", str_scratch.c_str());	//<- DBG

	gSk256 = Uint256(reinterpret_cast<uint8_t*>(decrypted_bytes));	// store global (TODO: rework this?) <----- [TODO/FIXME].
	//gSk256 = Uint256(str_scratch.c_str());	// store global (TODO: rework this?) <----- [TODO/FIXME].
	//gSk256 = Uint256();	// store global (TODO: rework this?) <----- [TODO/FIXME].
	//
// do this 2nd. <---

	// check for junk bytes at end, & check buffer lengths!   [WIP / TESTME] <--
	//
	std::string ethAddress{ "0x" };  /* "" */ /* populated by getAddressFromSKey(), expecting 0x + 20 bytes. */
	//
	secp256k1_context* ctx_both = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY); // <--
	if (!ctx_both) { /* context ok? */
		LOG_F(ERROR, "DecryptKeystore():  Couldn't create secp256k1 context!  Aborting.");
		return "Error: could not create ecdsa context";
	}

	// === get ethereum address from private key ===
	const bool getaddr_success = getAddressFromSKey(ctx_both, decrypted_bytes, ethAddress);

	randombytes_buf(decrypted_bytes, MESSAGE_LEN);  // <------------- Make the message length 64? Or have 0x+the private key bytes (as hex) as the
	secp256k1_context_destroy(ctx_both);

	// ^
	// === clean up ===
	//OverwriteAndFree(...);  [idea]
	// overwrite SK (decrypted_bytes)?? <-------[wip] 
													// encrypted message, but strip that off & just pass raw bytes as array of type `unsigned char`
	randombytes_buf(salt_bytes, SALT_LENGTH);  // <--- ?
	randombytes_buf(nonce_bytes, crypto_secretbox_xchacha20poly1305_NONCEBYTES);  // <--- ?
	randombytes_buf(key_bytes, PUBKEY_LENGTH);  // <--- ?if (getaddr_success)
	// anything else?

	if (getaddr_success)
	{ // ethAddress should now contain `0x` + a 20-byte(40 hex digit) Ethereum address, or an "Error: " string.
		if (IfEthereumAddress(ethAddress)) {
			printf("Got Solo mode Eth address:  %s \n", ethAddress.c_str());
			gStr_SoloEthAddress = ethAddress;  // store globally
		}
		else { printf("Err: invalid derived ethAddress %s \n", ethAddress.c_str()); }

		return ethAddress;
	}
	else { return "Error getting Ethereum address from sKey "; }
}



System::Boolean Keystore::LoadSKeyFromConfig(const std::string str_password)
{	// Solo mining-specific. Load encrypted account key from the Configuration, using
	//	the supplied password to decrypt and load it for use
	auto hConfig = ConfigurationManager::OpenExeConfiguration(ConfigurationUserLevel::None);	//get handle to the Configuration
																								// (used?)
	msclr::interop::marshal_context marshalctx;

	int kdfAlgo{ -99 };  // null
	unsigned int encKeyLength{ 0 };
	std::string str_encKey{ "" }, str_encNonce{ "" }, str_salt{ "" };

	// if the key exists
	// TODO: (refined) error checking for bad config file
	LOG_IF_F(INFO, HIGHVERBOSITY, "Reading encryptedPKbytes from Config.");
	if (ConfigurationManager::AppSettings["encryptedPKbytes"])
		str_encKey = marshalctx.marshal_as<std::string>(ConfigurationManager::AppSettings["encryptedPKbytes"]);

	LOG_IF_F(INFO, HIGHVERBOSITY, "Reading encryptedPKnonce from Config.");
	if (ConfigurationManager::AppSettings["encryptedPKnonce"])
		str_encNonce = marshalctx.marshal_as<std::string>(ConfigurationManager::AppSettings["encryptedPKnonce"]);

	LOG_IF_F(INFO, HIGHVERBOSITY, "Reading kdfAlgo from Config.");	// <--- [WIP]: always using value `2`.
	if (ConfigurationManager::AppSettings["kdfAlgo"]) {
		//	kdfAlgo = marshalctx.marshal_as<int>(ConfigurationManager::AppSettings["kdfAlgo"]);
		String^ kdfAlgoMStr = ConfigurationManager::AppSettings["kdfAlgo"];
		kdfAlgo = Convert::ToInt16(kdfAlgoMStr, 10);  // from base 10 <------- use tryparse instead? (fixme)
		Console::WriteLine("Got kdfAlgo String^: " + kdfAlgoMStr + " wrangled as base10 decimal to integer: " + Convert::ToString(kdfAlgo));
	}

	//	printf("Reading encryptedPKlength from Config... \n");
	//	if (ConfigurationManager::AppSettings["encryptedPKlength"]) {
	//		//	encKeyLength = marshalctx.marshal_as<unsigned int>(ConfigurationManager::AppSettings["encryptedPKlength"]);
	//		encKeyLength = Convert::ToUInt16(ConfigurationManager::AppSettings["encryptedPKlength"], 10);  // from base 10. <------ use tryparse()!
	//		if (gVerbosity == V_DEBUG) {
	//			Console::WriteLine("got encryptedPKlength String^: " + ConfigurationManager::AppSettings["encryptedPKlength"] +
	//				" wrangled as base10 decimal to integer: " + Convert::ToString(encKeyLength)); }
	//	}

	LOG_IF_F(INFO, HIGHVERBOSITY, "Reading salt from Config.");
	if (ConfigurationManager::AppSettings["salt"]) {
		str_salt = marshalctx.marshal_as<std::string>(ConfigurationManager::AppSettings["salt"]);  // password salt for kdf
		LOG_IF_F(INFO, gVerbosity == V_DEBUG, "Got salt from config:  %s ", str_salt.c_str());
	}

	// [WIP / FIXME]: Check the inputs
	// ===
	// [REF]: unsigned int encKeyLength{ 0 };	// <--- no longer needed? just check the input length? <--
	// [REF]: int kdfAlgo = -99;				// the kdfAlgo needs to be checked also <--- [TODO]
	//
	if (!checkString(str_encKey, CIPHERTEXT_LEN * 2, false, true) ||		/* must be CIPHERTEXT_LEN (*2?) characters long. hex, no 0x specifier.  [TESTME] */
		!checkString(str_encNonce, crypto_secretbox_xchacha20poly1305_NONCEBYTES * 2, false, true) ||		/* must be CIPHERTEXT_LEN long, hex, no hex specifier.  [TESTME] */
		!checkString(str_salt, SALT_LENGTH * 2, false, true))
	{ //if input(s) not in the expected format:
		LOG_F(ERROR, "Bad stored account data in Configuration. Not proceeding. \n");
		MessageBox::Show("No Ethereum account is stored. Please import one \n"
			"to configure Solo mining.", "COSMiC - Keystore Error", MessageBoxButtons::OK, MessageBoxIcon::Information);
		return false;
	}

	//inputs look good:
	std::string ethAddress = "0x00";  // TODO: check the kdfalgo. enckeylength no longer needed?
	ethAddress = DecryptKeystore(str_encKey, encKeyLength, str_encNonce, str_password, kdfAlgo, str_salt);
	if (checkErr_a(ethAddress)) {
		gStr_SoloEthAddress = ethAddress;  // set Solo mode Eth address.
		LOG_IF_F(INFO, HIGHVERBOSITY, "Setting Solo Eth Address (string): %s", ethAddress.c_str());
		return true;  // OK: loaded account
	}
	else {
		LOG_IF_F(ERROR, gVerbosity == V_DEBUG, "%s ", ethAddress.c_str());
		MessageBox::Show("Unable to decrypt keystore. Is the password correct?", "COSMiC - Keystore Error", MessageBoxButtons::OK, MessageBoxIcon::Asterisk);
		return false;	//error
	}

	// set the key/pad/nce/etc. also? 
	//}
	//else { }
}
