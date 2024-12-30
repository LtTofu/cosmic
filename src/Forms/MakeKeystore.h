// MakeKeystore.h : Windows form that allows user to easily store an existing Ethereum
//					account key, in encrypted form, using the Configuration.
// 2020-2021 LtTofu
#pragma once

#ifndef FORM_MAKEKEYSTORE
// ...
#define FORM_MAKEKEYSTORE

//#include <msclr/marshal_windows.h>
#include <msclr/marshal_cppstd.h>
#include <msclr/marshal.h>
#include <libsodium/sodium.h>

#include "defs.hpp"
#include "net_solo.h"
#include "util.hpp"				//conversion functions

// see: defs.h
constexpr auto PRIVATE_KEY_LENGTH = 64;
constexpr auto PRIVATE_KEY_LENGTH_WITH_0X = 66;

constexpr auto PKSTR_MINRANGE = 0;
constexpr auto PKSTR_0X_MINRANGE = 2;
constexpr auto PKSTR_0X_MAXRANGE = PRIVATE_KEY_LENGTH_WITH_0X - 1;		// 65
constexpr auto PKSTR_MAXRANGE = PRIVATE_KEY_LENGTH - 1;					// 63

constexpr auto MAX_PASSWORD_LENGTH = 100;
constexpr auto PASSWORD_ARRAY_LENGTH = MAX_PASSWORD_LENGTH;				// MAX?

#define SAVE_PKEY_TOKEYFILE	0
#define SAVE_PKEY_TOCONFIG	1
#define PKEY_FORMATINFO "\n(Private key should be 64 hex digits, possibly prefixed with '0x')."


// #define MESSAGE_LEN				32			// 32 bytes, or 64 hex digits.  was: 66 (included `0x`.)
// #define MESSAGE_STRING_LENGTH	64			// 32 bytes, or 64 hex digits.  was: 66 (included `0x`.)
// #define SALT_LENGTH				32			// in bytes
// #define DERIVED_KEY_LENGTH		32			// "
// #define CIPHERTEXT_LEN (crypto_secretbox_xchacha20poly1305_MACBYTES + MESSAGE_LEN)	// is ciphertext's length in BYTES or CHARACTERS? I think it's characters <--- [checkme]
																						// for xchacha20/poly1305
//using namespace System;
//using namespace System::Windows;
//using namespace System::Windows::Forms;
//using namespace System::Security;
//using namespace System::Security::Cryptography;
//	using namespace msclr::interop;														// for marshalling native<->managed


namespace Cosmic {

	using namespace System;
	//using namespace System::Windows;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Drawing;
	using namespace System::Security;
	using namespace System::Text;
	using namespace System::Security::Cryptography;
	using namespace System::Globalization;
	using namespace System::Runtime::InteropServices;
	using namespace System::Configuration;
	//using namespace System::Data;
	//using namespace msclr::interop;													// for marshalling native<->managed

	/// <summary>
	/// Summary for MakeKeystore
	/// </summary>
	public ref class MakeKeystore : public System::Windows::Forms::Form
	{
	private:
		//System::Numerics::BigInteger big_pk;
		String^ mstr_over = "0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";	//garbage-collecting after
																									//local System::String goes out of scope
		IFormatProvider^ format;
		Uint256* temp_sk256;
		Security::SecureString^ secureString_pass = gcnew Security::SecureString();					// [WIP] / [FIXME]. <--

	private:
		System::Windows::Forms::ToolTip^ toolTip1;
		System::Windows::Forms::TextBox^ active_textfield;				// reference to the text field which was right-clicked
																		// to open the context menu (paste/clear)
	public:
		MakeKeystore(void)
		{ // Form constructor
			temp_sk256 = nullptr;  // __nullptr?
			LOG_IF_F(INFO, gVerbosity == V_DEBUG, "Allocating a uint256 w/ libsodium.");
			temp_sk256 = static_cast<Uint256*>( sodium_malloc( sizeof(Uint256) ));  //<-- reinterpret_cast ?

			//big_pk = System::Numerics::BigInteger::Zero;
			format = gcnew CultureInfo("en-US");
			InitializeComponent();
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~MakeKeystore()
		{ // === destructor ===
			print_bytes_uchar(reinterpret_cast<unsigned char*>(temp_sk256), 32, "test_u256 (before overwriting)"); // <--
			//
			randombytes_buf(temp_sk256, 32);  // cast?
			//
			print_bytes_uchar(reinterpret_cast<unsigned char*>(temp_sk256), 32, "test_u256 (after overwriting)."); // <--
			//big_pk   <-- make sure big_pk is erased, GC'd
			// .. anything else to overwrite?

			if (components)
			{
				delete components;
			}
		}

	private:
		System::Void Force_GC()
		{ // [WIP]: force a garbage collection event - called after crucial handling of private parameters in managed code
		  //		(such as MakeKeystore)
			if (gVerbosity == V_DEBUG) { Console::WriteLine("Forcing Garbage Collection"); }
			System::GC::Collect();
			if (gVerbosity == V_DEBUG) { Console::WriteLine("Waiting for Full GC to complete"); }
			System::GC::WaitForFullGCComplete();  // <- [CHECKME]
			if (gVerbosity == V_DEBUG) { Console::WriteLine("Waiting for Pending Finalizers"); }
			System::GC::WaitForPendingFinalizers();
			if (gVerbosity == V_DEBUG) { Console::WriteLine("Waiting for Pending Finalizers"); }
			System::GC::WaitForPendingFinalizers();  // <-- req'd?
			if (gVerbosity == V_DEBUG) { Console::WriteLine("End of ForceGC()."); }
		};


	public:
		String^ GetPassword(void)
		{ // a simple getter for the main form to get text from the pass field.  [WIP]: reworking 	
			return textbox_pass->Text;
			// ...
		};

	private:
		// SAVEPRIVATEKEY(): Writes encrypted private key and associated parameters to the Configuration.	// [MOVEME] ?
		unsigned short SaveEncPrivateKey(const unsigned short saveWhere, const std::string& encPKey, const std::string& nonce,
			const unsigned int encPKeyLen, const int kdfAlgo, const std::string& salt)
		{
			LOG_IF_F(INFO, gVerbosity == V_DEBUG, "SaveEncPrivateKey(): got nonce  %s  and len: %u ", nonce.c_str(), encPKeyLen);	//<--- [DEBUG], REMOVE
			if (saveWhere == SAVE_PKEY_TOKEYFILE) {
				// [TODO]: save to dedicated keystore file.
				return 1;
			}
			else if (saveWhere == SAVE_PKEY_TOCONFIG)  // save to typical configuration
			{
				try {  // [WIP]: proper error handling for use of ConfigurationManager class:
					System::Configuration::Configuration^ hConfig = ConfigurationManager::OpenExeConfiguration(ConfigurationUserLevel::None);	//config handle
					String^ mngdStr_CryptedKey = gcnew String(encPKey.c_str());		// encrypted message (from a crypto_secretbox_xchacha20poly1305)
					String^ mngdStr_Nonce = gcnew String(nonce.c_str());			// to System::String

					// only remove key if it exists
					if (ConfigurationManager::AppSettings["encryptedPKbytes"])
						hConfig->AppSettings->Settings->Remove("encryptedPKbytes");
					hConfig->AppSettings->Settings->Add("encryptedPKbytes", mngdStr_CryptedKey);
					// 
					if (ConfigurationManager::AppSettings["encryptedPKnonce"])
						hConfig->AppSettings->Settings->Remove("encryptedPKnonce");
					hConfig->AppSettings->Settings->Add("encryptedPKnonce", mngdStr_Nonce);
					// 
					if (ConfigurationManager::AppSettings["encryptedPKlength"])
						hConfig->AppSettings->Settings->Remove("encryptedPKlength");
					hConfig->AppSettings->Settings->Add("encryptedPKlength", Convert::ToString(encPKeyLen));
					//
					if (ConfigurationManager::AppSettings["kdfAlgo"])
						hConfig->AppSettings->Settings->Remove("kdfAlgo");
					hConfig->AppSettings->Settings->Add("kdfAlgo", Convert::ToString(kdfAlgo));
					//
					if (ConfigurationManager::AppSettings["salt"])
						hConfig->AppSettings->Settings->Remove("salt");
					hConfig->AppSettings->Settings->Add("salt", gcnew String(salt.c_str()));

					// save out the configuration
					hConfig->Save(ConfigurationSaveMode::Modified);
					ConfigurationManager::RefreshSection("appSettings");
					LOG_IF_F(INFO, /*HIGH?*/NORMALVERBOSITY, "Updated config.");
				}
				catch (...) {

					return 2;  // Some error happened. Disk full? Config file read-only? <--
				}

				return 0;  // OK
			}
			else
			{
				LOG_F(ERROR, "Default in SaveEncPrivateKey()");
				return 3;  // Err
			}
		}

	//public:
	private:
		unsigned short StoreKey(const unsigned char* privKey, const /*unsigned*/ char* passPhrase, const size_t passLen)
		{
			unsigned char pwSalt[32]{};		// 256bit
			unsigned char key[32]{};		// output to this array from DeriveKeyFromPassPhrase_WinSDK().

			//DBG
			if (gVerbosity == V_DEBUG) {
				printf("Unsigned Char Array of passphrase (check ASCII vs. binary), %zu bytes:  \n", passLen);  // <---
				for (unsigned int i = 0; i < passLen; ++i)
					printf("%02x ", passPhrase[i]);
				printf("\n\n");
			}
			//DBG

			// `key` will be written into by this function (assuming it's succesful.)
			if (DeriveKeyFromPassPhrase_WinSDK( passPhrase, passLen, key, pwSalt, true) != 0) {
				MessageBox::Show( "An error occurred deriving an encryption key from the password. " "Please make sure no Cryptography-related "
					"Windows services are disabled.", "COSMiC - Error", MessageBoxButtons::OK, MessageBoxIcon::Asterisk );
				printf("Error deriving key from passphrase. \n");
				return 1;
			}

			print_bytes(pwSalt, 32, "password salt");  // debug verbosity only

			// Encrypt the message (private key) with a new nonce and key (generated from user-provided password).  Result out to byte-array `ciphertext`
			unsigned char nonce[crypto_secretbox_xchacha20poly1305_NONCEBYTES]{};  // nonce[crypto_secretbox_NONCEBYTES] = {0};
			unsigned char ciphertext[CIPHERTEXT_LEN] = {};

			randombytes_buf(nonce, crypto_secretbox_xchacha20poly1305_NONCEBYTES);						// buffer of random bytes of appropriate size (libsodium)
			crypto_secretbox_xchacha20poly1305_easy(ciphertext, privKey, (unsigned long long)MESSAGE_LEN, nonce, key);		// out to `ciphertext` <--

			// convert encrypted message encryption nonce byte-arrays to string, represented as hex  (for storage)
			const std::string str_encPK = HexBytesToStdString(ciphertext, CIPHERTEXT_LEN);								// the encrypted message from the user-provided private key
			const std::string str_nonce = HexBytesToStdString(nonce, crypto_secretbox_xchacha20poly1305_NONCEBYTES);	// nonce used	(these are stored in the Configuration)

			if (str_encPK.length() != CIPHERTEXT_LEN * 2 || str_nonce.length() != crypto_secretbox_xchacha20poly1305_NONCEBYTES * 2)
			{ /* Bad String Lengths?  [TODO] / [FIXME] ? */
				LOG_F(ERROR, "Bad encPK or encNonce length!");
				return 2;  // err!
			}  // <---

			LOG_IF_F(INFO, HIGHVERBOSITY, "Saving new encrypted account to Configuration.");

			if (SaveEncPrivateKey(SAVE_PKEY_TOCONFIG, str_encPK, str_nonce, CIPHERTEXT_LEN, 2, HexBytesToStdString(pwSalt, SALT_LENGTH))) {
				LOG_F(ERROR, "Error(s) saving encrypted account to the Configuration!");
				MessageBox::Show("Error(s) occurred while saving account to the Configuration.\n"
					"Disk full? Is COSMiC's folder read-only?", "COSMiC", MessageBoxButtons::OK, MessageBoxIcon::Error);  // <--
				return 3;  // err
			}

			// 
			// === clean up (overwrite buffers, strings) ===
			randombytes_buf(key, PRIVATE_KEY_BYTESLEN);			//overwrite arrays contents w/ random bytes

//			randombytes_buf(passPhrase, passLen);				// overwrite password!! <-- [FIXME]

			randombytes_buf(pwSalt, SALT_LENGTH);				// ...
			randombytes_buf(nonce, crypto_secretbox_xchacha20poly1305_NONCEBYTES);
			randombytes_buf(ciphertext, CIPHERTEXT_LEN);

			// pkStr  						  //<--- overwrite the skey byte array here? or in calling func. <- WIP/FIXME
			// anything need to be freed?
			return 0;  // OK
		};


	private: 
	System::Void pasteToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e)
	{
//		marshal_context^ mctx = gcnew marshal_context();
		if (!active_textfield) { /* null: not pointing to a textbox */
			Console::WriteLine("Error: No text field is active!");
			LOG_F(WARNING, "Error: No text field is active!");
			return; }

		// Pasting into either the `Private Key` or `Password` field:
		if (active_textfield == textbox_pk)
			MakeKeystore::PasteToSKeyField();
		else if (active_textfield == textbox_pass)
			MakeKeystore::PasteToPasswordField();
		else { /* [WIP]: Get the relevant textbox (in the ContextMenuStrip opening event?) and don't require focus. */
			Console::WriteLine("No textbox selected.");
			return; }
		
		// === cleanup: ===
//		delete mctx;
		return;				// clean up in calling function. (anything else to do here?)
		
	} //;

	private: System::Void keystoreContextMenu_Opening(System::Object^ sender, System::ComponentModel::CancelEventArgs^ e)
	{
		//active_textfield = static_cast<System::Windows::Forms::TextBox^>(static_cast<System::Windows::Forms::ContextMenuStrip^>(sender)->SourceControl);	<- condensed ver.
		System::Windows::Forms::ContextMenuStrip^ contextmenu = static_cast<System::Windows::Forms::ContextMenuStrip^>(sender);
		active_textfield = static_cast<System::Windows::Forms::TextBox^>(contextmenu->SourceControl);
		if (gVerbosity == V_DEBUG) {
			Console::WriteLine("Sender:  {0}", sender->ToString());
			Console::WriteLine("Should be context menu:  {0}", contextmenu->Name);
			Console::WriteLine("Sender's SourceControl:  {0}", active_textfield->Name);
		}

	}

	private: System::Windows::Forms::PictureBox^ pictureBox1;
	private: System::Windows::Forms::PictureBox^ pictureBox2;
	private: System::Windows::Forms::TextBox^ textbox_pk;
	private: System::Windows::Forms::GroupBox^ groupBox1;
	private: System::Windows::Forms::TableLayoutPanel^ tableLayoutPanel1;
	private: System::Windows::Forms::Label^ lbl_passkey;
	private: System::Windows::Forms::Label^ label_key;
	private: System::Windows::Forms::TextBox^ textbox_pass;
	private: System::Windows::Forms::CheckBox^ checkbox_hidepw;
	private: System::Windows::Forms::Button^ button1;
	private: System::Windows::Forms::Button^ button2;
	private: System::Windows::Forms::TextBox^ textBox3;
	private: System::Windows::Forms::Button^ button4;
	private: System::Windows::Forms::ContextMenuStrip^ keystoreContextMenu;
	private: System::Windows::Forms::ToolStripMenuItem^ pasteToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ clearToolStripMenuItem;
	private: System::ComponentModel::IContainer^ components;
	private:
		/// <summary>
		/// Required designer variable.
		/// </summary>


#pragma region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		void InitializeComponent(void)
		{
			this->components = (gcnew System::ComponentModel::Container());
			System::ComponentModel::ComponentResourceManager^ resources = (gcnew System::ComponentModel::ComponentResourceManager(MakeKeystore::typeid));
			this->pictureBox1 = (gcnew System::Windows::Forms::PictureBox());
			this->pictureBox2 = (gcnew System::Windows::Forms::PictureBox());
			this->textbox_pk = (gcnew System::Windows::Forms::TextBox());
			this->keystoreContextMenu = (gcnew System::Windows::Forms::ContextMenuStrip(this->components));
			this->pasteToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->clearToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->groupBox1 = (gcnew System::Windows::Forms::GroupBox());
			this->checkbox_hidepw = (gcnew System::Windows::Forms::CheckBox());
			this->tableLayoutPanel1 = (gcnew System::Windows::Forms::TableLayoutPanel());
			this->lbl_passkey = (gcnew System::Windows::Forms::Label());
			this->label_key = (gcnew System::Windows::Forms::Label());
			this->textbox_pass = (gcnew System::Windows::Forms::TextBox());
			this->button1 = (gcnew System::Windows::Forms::Button());
			this->button2 = (gcnew System::Windows::Forms::Button());
			this->textBox3 = (gcnew System::Windows::Forms::TextBox());
			this->button4 = (gcnew System::Windows::Forms::Button());
			this->toolTip1 = (gcnew System::Windows::Forms::ToolTip(this->components));
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox2))->BeginInit();
			this->keystoreContextMenu->SuspendLayout();
			this->groupBox1->SuspendLayout();
			this->tableLayoutPanel1->SuspendLayout();
			this->SuspendLayout();
			// 
			// pictureBox1
			// 
			this->pictureBox1->Anchor = System::Windows::Forms::AnchorStyles::None;
			this->pictureBox1->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"pictureBox1.Image")));
			this->pictureBox1->Location = System::Drawing::Point(6, 3);
			this->pictureBox1->Name = L"pictureBox1";
			this->pictureBox1->Size = System::Drawing::Size(16, 20);
			this->pictureBox1->TabIndex = 0;
			this->pictureBox1->TabStop = false;
			// 
			// pictureBox2
			// 
			this->pictureBox2->Anchor = System::Windows::Forms::AnchorStyles::None;
			this->pictureBox2->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"pictureBox2.Image")));
			this->pictureBox2->Location = System::Drawing::Point(6, 31);
			this->pictureBox2->Name = L"pictureBox2";
			this->pictureBox2->Size = System::Drawing::Size(16, 19);
			this->pictureBox2->TabIndex = 1;
			this->pictureBox2->TabStop = false;
			// 
			// textbox_pk
			// 
			this->textbox_pk->Anchor = System::Windows::Forms::AnchorStyles::Left;
			this->textbox_pk->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->textbox_pk->CharacterCasing = System::Windows::Forms::CharacterCasing::Upper;
			this->textbox_pk->ContextMenuStrip = this->keystoreContextMenu;
			this->textbox_pk->Location = System::Drawing::Point(102, 3);
			this->textbox_pk->MaxLength = 66;
			this->textbox_pk->Name = L"textbox_pk";
			this->textbox_pk->PasswordChar = '*';
			this->textbox_pk->ShortcutsEnabled = false;
			this->textbox_pk->Size = System::Drawing::Size(340, 20);
			this->textbox_pk->TabIndex = 1;
			this->textbox_pk->WordWrap = false;
			this->textbox_pk->TextChanged += gcnew System::EventHandler(this, &MakeKeystore::textbox_pk_TextChanged);
			this->textbox_pk->KeyUp += gcnew System::Windows::Forms::KeyEventHandler(this, &MakeKeystore::textbox_pk_KeyUp);
			// 
			// keystoreContextMenu
			// 
			this->keystoreContextMenu->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(2) {
				this->pasteToolStripMenuItem,
					this->clearToolStripMenuItem
			});
			this->keystoreContextMenu->Name = L"keystoreContextMenu";
			this->keystoreContextMenu->Size = System::Drawing::Size(144, 48);
			this->keystoreContextMenu->Opening += gcnew System::ComponentModel::CancelEventHandler(this, &MakeKeystore::keystoreContextMenu_Opening);
			// 
			// pasteToolStripMenuItem
			// 
			this->pasteToolStripMenuItem->Name = L"pasteToolStripMenuItem";
			this->pasteToolStripMenuItem->ShortcutKeys = static_cast<System::Windows::Forms::Keys>((System::Windows::Forms::Keys::Control | System::Windows::Forms::Keys::V));
			this->pasteToolStripMenuItem->Size = System::Drawing::Size(143, 22);
			this->pasteToolStripMenuItem->Text = L"Paste";
			this->pasteToolStripMenuItem->Click += gcnew System::EventHandler(this, &MakeKeystore::pasteToolStripMenuItem_Click);
			// 
			// clearToolStripMenuItem
			// 
			this->clearToolStripMenuItem->Name = L"clearToolStripMenuItem";
			this->clearToolStripMenuItem->Size = System::Drawing::Size(143, 22);
			this->clearToolStripMenuItem->Text = L"Clear";
			// 
			// groupBox1
			// 
			this->groupBox1->Controls->Add(this->checkbox_hidepw);
			this->groupBox1->Controls->Add(this->tableLayoutPanel1);
			this->groupBox1->Location = System::Drawing::Point(12, 52);
			this->groupBox1->Name = L"groupBox1";
			this->groupBox1->Size = System::Drawing::Size(483, 110);
			this->groupBox1->TabIndex = 1;
			this->groupBox1->TabStop = false;
			this->groupBox1->Text = L"Encrypt Private Key with Passphrase";
			// 
			// checkbox_hidepw
			// 
			this->checkbox_hidepw->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
			this->checkbox_hidepw->AutoSize = true;
			this->checkbox_hidepw->Checked = true;
			this->checkbox_hidepw->CheckState = System::Windows::Forms::CheckState::Checked;
			this->checkbox_hidepw->Location = System::Drawing::Point(403, 85);
			this->checkbox_hidepw->Name = L"checkbox_hidepw";
			this->checkbox_hidepw->Size = System::Drawing::Size(48, 17);
			this->checkbox_hidepw->TabIndex = 1;
			this->checkbox_hidepw->Text = L"Hide";
			this->checkbox_hidepw->UseVisualStyleBackColor = true;
			this->checkbox_hidepw->CheckStateChanged += gcnew System::EventHandler(this, &MakeKeystore::checkbox_hidepw_CheckStateChanged);
			// 
			// tableLayoutPanel1
			// 
			this->tableLayoutPanel1->ColumnCount = 3;
			this->tableLayoutPanel1->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute,
				29)));
			this->tableLayoutPanel1->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
				17.05607F)));
			this->tableLayoutPanel1->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
				82.94392F)));
			this->tableLayoutPanel1->Controls->Add(this->textbox_pk, 2, 0);
			this->tableLayoutPanel1->Controls->Add(this->lbl_passkey, 1, 1);
			this->tableLayoutPanel1->Controls->Add(this->pictureBox1, 0, 0);
			this->tableLayoutPanel1->Controls->Add(this->pictureBox2, 0, 1);
			this->tableLayoutPanel1->Controls->Add(this->label_key, 1, 0);
			this->tableLayoutPanel1->Controls->Add(this->textbox_pass, 2, 1);
			this->tableLayoutPanel1->Location = System::Drawing::Point(15, 24);
			this->tableLayoutPanel1->Name = L"tableLayoutPanel1";
			this->tableLayoutPanel1->RowCount = 2;
			this->tableLayoutPanel1->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent, 50)));
			this->tableLayoutPanel1->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent, 50)));
			this->tableLayoutPanel1->Size = System::Drawing::Size(445, 55);
			this->tableLayoutPanel1->TabIndex = 0;
			// 
			// lbl_passkey
			// 
			this->lbl_passkey->Anchor = System::Windows::Forms::AnchorStyles::Left;
			this->lbl_passkey->AutoSize = true;
			this->lbl_passkey->Location = System::Drawing::Point(32, 34);
			this->lbl_passkey->Name = L"lbl_passkey";
			this->lbl_passkey->Size = System::Drawing::Size(56, 13);
			this->lbl_passkey->TabIndex = 6;
			this->lbl_passkey->Text = L"Password:";
			this->lbl_passkey->Click += gcnew System::EventHandler(this, &MakeKeystore::lbl_passkey_Click);
			// 
			// label_key
			// 
			this->label_key->Anchor = System::Windows::Forms::AnchorStyles::Left;
			this->label_key->AutoSize = true;
			this->label_key->Location = System::Drawing::Point(32, 7);
			this->label_key->Name = L"label_key";
			this->label_key->Size = System::Drawing::Size(64, 13);
			this->label_key->TabIndex = 6;
			this->label_key->Text = L"Private Key:";
			// 
			// textbox_pass
			// 
			this->textbox_pass->Anchor = System::Windows::Forms::AnchorStyles::Left;
			this->textbox_pass->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->textbox_pass->ContextMenuStrip = this->keystoreContextMenu;
			this->textbox_pass->Location = System::Drawing::Point(102, 31);
			this->textbox_pass->MaxLength = 100;
			this->textbox_pass->Name = L"textbox_pass";
			this->textbox_pass->PasswordChar = '*';
			this->textbox_pass->ShortcutsEnabled = false;
			this->textbox_pass->Size = System::Drawing::Size(340, 20);
			this->textbox_pass->TabIndex = 2;
			this->textbox_pass->WordWrap = false;
			this->textbox_pass->TextChanged += gcnew System::EventHandler(this, &MakeKeystore::textbox_pass_TextChanged);
			// 
			// button1
			// 
			this->button1->DialogResult = System::Windows::Forms::DialogResult::OK;
			this->button1->Location = System::Drawing::Point(355, 173);
			this->button1->Name = L"button1";
			this->button1->Size = System::Drawing::Size(108, 26);
			this->button1->TabIndex = 1;
			this->button1->Text = L"Save & Use";
			this->button1->UseMnemonic = false;
			this->button1->UseVisualStyleBackColor = true;
			this->button1->Click += gcnew System::EventHandler(this, &MakeKeystore::button1_Click);
			// 
			// button2
			// 
			this->button2->DialogResult = System::Windows::Forms::DialogResult::Cancel;
			this->button2->Location = System::Drawing::Point(12, 173);
			this->button2->Name = L"button2";
			this->button2->Size = System::Drawing::Size(75, 26);
			this->button2->TabIndex = 3;
			this->button2->Text = L"Cancel";
			this->button2->UseVisualStyleBackColor = true;
			this->button2->Click += gcnew System::EventHandler(this, &MakeKeystore::button2_Click);
			// 
			// textBox3
			// 
			this->textBox3->BorderStyle = System::Windows::Forms::BorderStyle::None;
			this->textBox3->Cursor = System::Windows::Forms::Cursors::Arrow;
			this->textBox3->Location = System::Drawing::Point(14, 14);
			this->textBox3->MaxLength = 512;
			this->textBox3->Multiline = true;
			this->textBox3->Name = L"textBox3";
			this->textBox3->ReadOnly = true;
			this->textBox3->Size = System::Drawing::Size(569, 35);
			this->textBox3->TabIndex = 0;
			this->textBox3->TabStop = false;
			this->textBox3->Text = L"Encrypts an account\'s Private Key with a user-supplied password.\r\nClick \"Save & U"
				L"se\" to use this Keystore for Solo Mining. \"Save Keystore...\" to save it for late"
				L"r use.\r\n";
			// 
			// button4
			// 
			this->button4->Enabled = false;
			this->button4->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"button4.Image")));
			this->button4->Location = System::Drawing::Point(227, 173);
			this->button4->Name = L"button4";
			this->button4->Size = System::Drawing::Size(118, 26);
			this->button4->TabIndex = 2;
			this->button4->Text = L"Save Keystore...";
			this->button4->TextImageRelation = System::Windows::Forms::TextImageRelation::ImageBeforeText;
			this->button4->UseVisualStyleBackColor = true;
			// 
			// toolTip1
			// 
			this->toolTip1->Popup += gcnew System::Windows::Forms::PopupEventHandler(this, &MakeKeystore::toolTip1_Popup);
			// 
			// MakeKeystore
			// 
			this->AcceptButton = this->button1;
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->CancelButton = this->button2;
			this->ClientSize = System::Drawing::Size(506, 208);
			this->ControlBox = false;
			this->Controls->Add(this->button4);
			this->Controls->Add(this->textBox3);
			this->Controls->Add(this->button2);
			this->Controls->Add(this->button1);
			this->Controls->Add(this->groupBox1);
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::FixedDialog;
			this->Icon = (cli::safe_cast<System::Drawing::Icon^>(resources->GetObject(L"$this.Icon")));
			this->MaximizeBox = false;
			this->MinimizeBox = false;
			this->Name = L"MakeKeystore";
			this->ShowIcon = false;
			this->ShowInTaskbar = false;
			this->SizeGripStyle = System::Windows::Forms::SizeGripStyle::Hide;
			this->StartPosition = System::Windows::Forms::FormStartPosition::CenterParent;
			this->Text = L"COSMiC - Import Keystore";
			this->Load += gcnew System::EventHandler(this, &MakeKeystore::MakeKeystore_Load);
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox2))->EndInit();
			this->keystoreContextMenu->ResumeLayout(false);
			this->groupBox1->ResumeLayout(false);
			this->groupBox1->PerformLayout();
			this->tableLayoutPanel1->ResumeLayout(false);
			this->tableLayoutPanel1->PerformLayout();
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion

	private: System::Void button1_Click(System::Object^ sender, System::EventArgs^ e)
	{ // Save button pressed

		// === check password length ===
		if (textbox_pass->Text->Length < 1 || textbox_pass->Text == "Type a good password here." || textbox_pass->Text->Length > MAX_PASSWORD_LENGTH) {
			MessageBox::Show("Please enter a password.", "COSMiC", MessageBoxButtons::OK, MessageBoxIcon::Information);
			this->DialogResult = System::Windows::Forms::DialogResult::None;  // don't close the form
			textbox_pass->Focus();  // focus the textbox control whose input text needs fixing.
			return;
		}

		// === check private key length and format === */
		// alternately:  if (temp_sk256 == Uint256::ZERO) { ...
		if (*temp_sk256 == Uint256::ZERO || (textbox_pk->Text->Length != 64 && textbox_pk->Text->Length != 66)) {
			MessageBox::Show("Please enter a valid private key. This is usually \n"
				"64 hex characters (0-9, A-F), often starting with '0x'.", "COSMiC", MessageBoxButtons::OK, MessageBoxIcon::Information);
			this->DialogResult = System::Windows::Forms::DialogResult::None;  // don't close the form
			//textbox_pk->Focus();  // focus the textbox control whose input text needs fixing.
		}

		// === do we have the needed parameters?  (private key and password to encrypt it with) ===
		msclr::interop::marshal_context mctx;	//marshal_context^ m_mctx = gcnew marshal_context();
		if (gVerbosity==V_DEBUG) { /* Debug mode only [REMOVE] */
			unsigned char DEBUG_bytes[32]{};					//
			temp_sk256->getBigEndianBytes( DEBUG_bytes );		//
			print_bytes(DEBUG_bytes, 32, "Debug: SKey bytes");	}// <- [REMOVE]
		//
		// instead.. just get the stored uint256 from text changed handler on the textbox
		//if (!CheckInput_PrivateKeyField()) { ...

		// was here:  check for non-hex characters (no `0x` specifier).
		// max password input length? <-- [fixme/todo]
		Console::WriteLine("\nEncrypting Keystore...");
		this->Text = "COSMiC - Encrypting Keystore...";  // Form title
		button1->Text = "Encrypting...";

		this->Refresh();  // so we see the update
		tableLayoutPanel1->Enabled = false;

		const size_t passLength = textbox_pass->Text->Length;
		//if ((pass_byteslen_variable % 2) != 0)  { /* redundant */ }  // <- for characters, not bytes

		unsigned char skey_uchar[PRIVATE_KEY_BYTESLEN]{};		// private key bytes
//		unsigned char pass_uchar[PASSWORD_ARRAY_LENGTH]{};		// password to uchar array... format idea: bytes(ascii values)? or..   (CHECK LENGTH, MAX, etc.)
		
//unsigned char* pass_uchar = (unsigned char*)sodium_malloc(pass_byteslen_variable);  // ^ MAKE THIS VARIABLE LENGTH!! <--- pass along the length as needed. overwrite when done.    <--- OVERWRITE AND FREE `pass_uchar` <-
		//memset(pass_uchar, 0, pass_byteslen_variable);

		memcpy(&gSk256, temp_sk256, 32);	//gSk256 = *temp_sk256;


		//cstring_to_uchar_array(mctx.marshal_as<const char*>(textbox_pk->Text), PRIVATE_KEY_BYTESLEN, skey_uchar);	// PK field's Text (64 chars. hex) to byte-array

		//cstring_to_uchar_array(mctx.marshal_as<const char*>(textbox_pk->Text), PRIVATE_KEY_BYTESLEN, skey_uchar);	// PK field's Text (64 chars. hex) to byte-array
		temp_sk256->getBigEndianBytes( static_cast<unsigned char[32]>(skey_uchar) );  //<-- is reinterpret_cast preferable here?

		//cstring_to_uchar_array(mctx.marshal_as<const char*>(textbox_pass->Text), textbox_pass->Text->Length, pass_uchar);	//<--- CHECK LENGTH [FIXME].  [Note to self]: converting directly to
		//pass_uchar = mctx.marshal_as<unsigned char*>(textbox_pass->Text); //, pass_byteslen_variable, pass_uchar);  //new
	   //pass_uchar = mctx.marshal_as<unsigned char*>(textbox_pass->Text); //, pass_byteslen_variable, pass_uchar);  //new
		
		//Console::WriteLine("DEBUG:  strlen( pass_uchar ):  {0}", strlen(pass_uchar));  // <- Debug, Remove
// ^ [TESTME] get the password as ascii values, unicode, bytes ...? be consistent on save/load. Appears this string should be invalid once the marshal context goes out of scope. Overwrite it regardless <--

//		Console::WriteLine("passphrase (in textbox_pass, type System::String:  {0}, length {1}", textbox_pass->Text, textbox_pass->Text->Length);
		//print_bytes((uint8_t*)pass_uchar, pass_byteslen_variable, "passphrase as unsigned char array (NOT converted!)");


		// [TODO] make passphrase (arg to StoreKey()) an unsigned char array & overwrite it after calling?
		// === save out the key to Configuration ===

		//const unsigned short storekey_result = StoreKey(skey_uchar, pass_uchar, pass_byteslen_variable);
		if (gVerbosity == V_DEBUG) {
			print_bytes_uchar((const unsigned char*)(mctx.marshal_as<const char*>(textbox_pass->Text)), passLength, "passphrase as char array (MARSHALED, No 'manual' conversion. Check Value!)");
			print_bytes_uchar(skey_uchar, 32, "skey as unsigned char array (NOT converted!)");  //<-
		} // ^^ [TESTME]
		const unsigned short storekey_result = StoreKey(skey_uchar, mctx.marshal_as<const char*>(textbox_pass->Text), passLength);
		
	//DEBUG PRINTS: Remove these!
		print_bytes((uint8_t*)mctx.marshal_as<const char*>(textbox_pass->Text), static_cast<uint8_t>(passLength), "print_bytes() w/ marshal'd char array from System::String (password)");		// <- REMOVE
		printf("--- %s --- \n", mctx.marshal_as<const char*>(textbox_pass->Text));						// <- DBG
		//randombytes_buf( pkey_pass, textbox_pass->Text->Length );		// <- overwrite skey bytes array
		randombytes_buf(skey_uchar, PRIVATE_KEY_BYTESLEN);				// <- overwrite skey bytes array
//		randombytes_buf(pass_uchar, pass_byteslen_variable);		// <- overwrite password bytes array								<-------- !
//		sodium_free(pass_uchar);									// free pointer to `pass_byteslen_variable` bytes of secure memory  <--- !
		randombytes_buf(temp_sk256, 32);								// <- overwrite temp uint256 on form
		Force_GC();														// <- force .NET garbage collection			<-- after form closes instead?
		//
		if (storekey_result != 0) {
			MessageBox::Show("An error occurred saving the account to Configuration. \n" "Is the disk full? Is COSMiC's folder read-only?", 
				"COSMiC", MessageBoxButtons::OK, MessageBoxIcon::Error);
			//return;
		} // ... anything else?
		
	//	textbox_pass->Clear();		// clear fields.  ensure ->Text MStrings are garbage-collected <-- [WIP]: more secure password entry.
		textbox_pk->Clear();		// <- field should only contain password character (*) anyway
	//	delete m_mctx;
	// [TODO / WIP]: any further error checks in the Save/Load account process!
		return;
	}

	private: System::Void textbox_pass_TextChanged(System::Object^ sender, System::EventArgs^ e)
	{
		if (textbox_pass->Text == "Type a good password here.")
			button1->Enabled = false;
		else  button1->Enabled = true;
	}

	private: System::Void checkbox_hidepw_CheckStateChanged(System::Object^ sender, System::EventArgs^ e)
	{
		// set visibility of private key/password entry characters based on this checkbox's checked state
		if (gVerbosity == V_DEBUG) { Console::WriteLine("MakeKeystore:  Changing visibility of input fields"); }
			
		//textbox_pk->PasswordChar = '-';		// or: '\0'
		//textbox_pass->PasswordChar = '-';		//
		textbox_pk->UseSystemPasswordChar = checkbox_hidepw->Checked;
		textbox_pass->UseSystemPasswordChar = checkbox_hidepw->Checked;
	}

	private: System::Void button2_Click(System::Object^ sender, System::EventArgs^ e){
	}

	private: System::Void MakeKeystore_Load(System::Object^ sender, System::EventArgs^ e)
	{
		//checkbox_hidepw->Checked = true;
		//textbox_pk->PasswordChar = '-';
		//textbox_pass->PasswordChar = '-';
	}

	private: System::Void textbox_pk_TextChanged_1(System::Object^ sender, System::EventArgs^ e)
	{//Either textbox has changed text contents:
		if (gVerbosity == V_DEBUG) { Console::WriteLine("textbox_pk textchanged event"); }
		//
		bool bConditions{ false };
		if (textbox_pk->TextLength == 66 && textbox_pass->TextLength != 0 && textbox_pass->Text != "Type a good password here.")
			bConditions = true;									  // check input text lengths, password entry has changed from default
		button1->Enabled = bConditions;					  // toggle based on above conditions
	}



	// Takes a managed string (and its specified length) input for proper length/format
	private:
	__forceinline System::Boolean CheckInput_PrivateKeyField(System::String^ InputMString, System::UInt16 Start_Index, System::UInt16 MString_Length)		/* <- experimenting, get rid of 2nd param. */
	{
		int LastCharacter = InputMString->Length - 1;  // Last character in the System::String's index (from 0).
		int i{ Start_Index };
		while (i <= LastCharacter)
		{
			if (Uri::IsHexDigit(InputMString[i]))
			{
				if (gVerbosity == V_DEBUG) { Console::WriteLine("Character OK.  Index: {0}.  Character: {1}.", i, InputMString[i]); }  //<-- remove
				if (i >= LastCharacter) {	/* is this the last character in the string? */
					return true;			/* yes */
					break;
				}
				++i;						/* no */
			}
			else { /* the character at index `i` is not a valid hex digit */
				if (gVerbosity == V_DEBUG) { Console::WriteLine("Bad character at index {0}:  {1}", i, InputMString[i]); }  //<- remove
				return false;
			}
		}
		return false;  // default: should only be hittable if i > LastCharacter.
	}


	private:
	System::Boolean PasteToPasswordField ( void )
	{
		System::String^ paste_text = gcnew String("");  // ensure this is cleared/GC'd!  [WIP] <--
		msclr::interop::marshal_context mctx;			// <- use mng'd type & delete after use?

		System::String^ PasteText = Clipboard::GetText(TextDataFormat::Text);
		if (!PasteText || PasteText->Length < 1 || PasteText->Length >= textbox_pass->MaxLength) {
			LOG_F( WARNING, "PasteToPasswordField():  Clipboard text is not usable length for password" );
			Console::WriteLine( "No usable text in Clipboard. Maximum length 100, minimum length 1" );
			PasteText = mstr_over;
			return false;
		}

	//	=== Check the clipboard text, expecting all hex and this length 64 or 66 === 
		if (PasteText->Length < 1 || PasteText->Length > textbox_pass->MaxLength) { /* 100 */
			LOG_F(ERROR, "Bad input length in PasteToPasswordField().");
			PasteText = mstr_over;
			return false;
		}

	// write asterisks to the textbox instead of the pasted text (password):
		StringBuilder^ sb = gcnew System::Text::StringBuilder( PRIVATE_KEY_LENGTH, PRIVATE_KEY_LENGTH_WITH_0X /* ? */);  // <--- use bHasHexSpecifier ? x : y; ?
		for (int i = 0; i < PasteText->Length; ++i)
			sb->Append("*");//^ here too: PRIVATE_KEY_LENGTH_WITH_0X.

		textbox_pk->Text = sb->ToString();  // write asterisks to priv.key box.
		return true;  // OK
	}


	private:
	bool PasteToSKeyField(void)
	{
		/*bool bValidInput{ false }, bParseSuccess{ false }, */ bool bHasHexSpecifier{ false };
		System::String^ PasteText = gcnew String(" ");  // ensure this is cleared/GC'd!  [WIP] <--
		msclr::interop::marshal_context mctx;			// <- use mng'd type & delete after use?

		if (!Clipboard::ContainsText(TextDataFormat::Text)) {
			Console::WriteLine("Clipboard does not contain text");
			return false; }

	//	=== Check the clipboard text, expecting all hex and this length 64 or 66 === 
		try {
			PasteText = Clipboard::GetText(TextDataFormat::Text);
		} catch (Exception^ e) {
			const std::string s_exceptionMessage{ mctx.marshal_as<std::string>(e->Message) };
			const std::string s_exceptionStackTrace{ gVerbosity == V_DEBUG ? mctx.marshal_as<std::string>(e->StackTrace) : "" };  // stack trace if V_DEBUG only.

			LOG_F( ERROR, "Caught exception pasting s.key:  %s ", s_exceptionMessage.c_str() );
			LOG_IF_F( ERROR, gVerbosity>V_NORM, "Exception call stack: \n%s \n", s_exceptionStackTrace.c_str() );

			PasteText = mstr_over;  // <- probably accomplishes nothing due to how System::String works. The idea is to
									//		invalidate the pasted text String before the Garbage Collection.
			return false;
		}
		
		//
		if (gVerbosity==V_DEBUG) {
			Console::WriteLine("Got text:  {0},   {1}", PasteText, Clipboard::GetText(TextDataFormat::Text));  }// <--- remove!
		//
		if (!PasteText || (PasteText->Length != 64 && PasteText->Length != 66)) {
			LOG_F(WARNING, "Clipboard text is not acceptable length for private key: %d", PasteText->Length);
			Console::WriteLine("No usable text in Clipboard.  Expecting 64 or 66-character private key, with/without `0x`. Got {0}.", PasteText->Length);
			PasteText = mstr_over;
			return false; }

		if (PasteText->Length==66 && PasteText->Substring(0,2)=="0x")
			bHasHexSpecifier = true;

		// check for hex characters only, from specified character index (2: to skip over "0x" specifier).
		if (!CheckInput_PrivateKeyField(PasteText, bHasHexSpecifier ? 2 : 0, bHasHexSpecifier ? PRIVATE_KEY_LENGTH_WITH_0X : PRIVATE_KEY_LENGTH)) {   // <- redundant 2nd arg.?   <---- made extraneous by Approach "B" <--
			LOG_F(ERROR, "Non-hex input in PasteToSKeyField().");
			PasteText = mstr_over;
			return false; }

		*temp_sk256 = Uint256::ZERO;
		*temp_sk256 = Uint256( mctx.marshal_as<const char*>(bHasHexSpecifier ? PasteText->Substring(2) : PasteText) );  //<- WIP. Obviate need for substr. [fixme/todo]
		if (*temp_sk256 == Uint256::ZERO) {
			Console::WriteLine("Parse input to uint256 failed in pasteToolStripMenuItem_Click()! Should never happen [FIXME]!");
			LOG_F(ERROR, "Parse input to uint256 failed in pasteToolStripMenuItem_Click(). Please report this bug");   // error! <--- wip/todo
			PasteText = mstr_over;
			return false;
		}

		// write asterisks to the textbox instead of the pasted text (key):
		StringBuilder^ sb = gcnew System::Text::StringBuilder(PRIVATE_KEY_LENGTH_WITH_0X, PRIVATE_KEY_LENGTH_WITH_0X /* ? */);  // <--- use bHasHexSpecifier ? x : y; ?
		for (USHORT i2 = 0; i2 < (bHasHexSpecifier ? PRIVATE_KEY_LENGTH_WITH_0X : PRIVATE_KEY_LENGTH); ++i2)
			sb->Append( "*" );

		textbox_pk->Text = sb->ToString();  // write asterisks to priv.key box.
		textbox_pk->Enabled = false;        // <- new. "lock in" key if pasted :)
		return true;  // OK
	}






// handle a valid key pressed with the password or secretkey field selected (not backspace/delete)
private: System::Void KeyUp_HandleValidKey(System::Windows::Forms::KeyEventArgs^ e, System::Object^ sender)
{
	Console::WriteLine("Handling other keyUp with valid key:  {0}  pressed in control {1}: ", e->KeyCode, sender->ToString());
	System::IFormatProvider^ iFormat = gcnew Globalization::CultureInfo("en-US");
	unsigned int caret_pos{ };
	//
	//secureString_pass->AppendChar(e->KeyCode.ToChar(iFormat));
	//textbox_pass->Text->Insert(caret_pos, )

	//ResetDisplayCharacters();
	//...

}


	   // backspace pressed in the password or secretkey field.  [TODO]: support 'Delete', anything else?
private: System::Void KeyUp_HandleBackspace(System::Windows::Forms::KeyEventArgs^ e)
{
	Console::WriteLine("textbox_pass:  Backspace. (new length {0})", textbox_pass->TextLength);
	if (textbox_pass->SelectionLength)
	{
		if (textbox_pass->SelectionLength >= MAX_PASSWORD_LENGTH || textbox_pass->SelectionLength > MAX_PASSWORD_LENGTH) {
			Console::WriteLine("MakeKeystore form: KeyUp_HandleBackspace: impossible happened.");
			return;
		}

		if (textbox_pass->SelectionLength >= textbox_pass->TextLength) {
			Console::WriteLine("clearing textbox_pass and securestring ");
			secureString_pass->Clear();
			textbox_pass->Clear();
			return;
		}

		Console::WriteLine("{0} characters selected from positions {1} ", textbox_pass->SelectionLength, textbox_pass->SelectionStart);
		//
		if (textbox_pass->SelectionLength == textbox_pass->TextLength || textbox_pass->TextLength == 1)
		{ // the whole entry text is selected, or this backspace will make the field empty:
			textbox_pass->Clear();
			secureString_pass->Clear();
			Console::WriteLine("Cleared textbox_pass:  now contains {0} with length {1} ", textbox_pass->Text, textbox_pass->TextLength);  //<- DBG
		}
		else {
			textbox_pass->Text->Remove(textbox_pass->SelectionStart, textbox_pass->SelectionLength);
			//		   const unsigned short selected_characters = textbox_pass->SelectionLength;
			//		   if (selected_characters > 0 && selected_characters < MAX_PASSWORD_LENGTH)
			//		   {
			for (unsigned short i = 0; i < textbox_pass->SelectionLength; ++i)
				secureString_pass->RemoveAt(textbox_pass->SelectionStart);
			Console::WriteLine("secureString now contains:  {0},  System::String version:  {1},  length {2}.", secureString_pass, secureString_pass->ToString(), secureString_pass->Length);
			//		   }
		}

		return;  // remove
	}

	if (textbox_pass->TextLength > 0 || textbox_pass->SelectionLength > 0)
	{
		textbox_pass->Text = textbox_pass->Text->Remove(textbox_pass->TextLength - 1, 1);  // remove rightmost character
		Console::WriteLine("textbox_pass:  Backspace ignored, empty  (length {0})", textbox_pass->TextLength);  // <---- DBG
		//secureString_pass->RemoveAt()

	}
}

private: System::Void toolTip1_Popup(System::Object^ sender, System::Windows::Forms::PopupEventArgs^ e)
{
}

private: System::Void textbox_pk_KeyUp(System::Object^ sender, System::Windows::Forms::KeyEventArgs^ e)
{

}

private: System::Void textbox_pk_TextChanged(System::Object^ sender, System::EventArgs^ e)
{
	if (*temp_sk256 != Uint256::ZERO)
	{ // a private key was already entered when this event was raised
		if (gVerbosity == V_DEBUG)  { Console::WriteLine("An skey is already stored"); }


		// any other cleanup?
		//Force_GC();  <-- on form closure, in CosmicWind calling func?  [todo / wip]
		return;
	}

	// otherwise:
	if (textbox_pk->Text->Length % 2 != 0)
		return;
	if (textbox_pk->Text->Length != PRIVATE_KEY_LENGTH && textbox_pk->Text->Length != PRIVATE_KEY_LENGTH_WITH_0X)
		return;
	const bool bHasHexSpecifier = (textbox_pk->Text->Length == 66 && textbox_pk->Text->Substring(0, 2) == "0x");
	msclr::interop::marshal_context marshalctx;  // = gcnew marshal_context();
	//

	*temp_sk256 = Uint256::ZERO;  // clear stored pk in the form instance
	
	// check for hex characters only:
	if (!CheckInput_PrivateKeyField(textbox_pk->Text, bHasHexSpecifier ? 2 : 0, bHasHexSpecifier ? PRIVATE_KEY_LENGTH_WITH_0X : PRIVATE_KEY_LENGTH)) {
		if (gVerbosity > V_NORM) { Console::WriteLine("PK is appropriate length but not hex format"); }
		return; }

	*temp_sk256 = Uint256(marshalctx.marshal_as<const char*>(bHasHexSpecifier ? textbox_pk->Text->Substring(2) : textbox_pk->Text));
	//delete marshalctx;
	MakeKeystore::Force_GC();			// 
	if (*temp_sk256 != Uint256::ZERO)	/* parsed successfully, stored in form instance member */
	{
		// ...
		unsigned char debug_bytes[32]{};												//						 <-- remove (DBG)
		temp_sk256->getBigEndianBytes(debug_bytes);										// out to debug array :)
		print_bytes_uchar(debug_bytes, 32, "DEBUG: uint256 bytes of entered skey");		// neat ordered bytes	 <-- remove (DBG)
		if (gVerbosity > V_NORM) { Console::WriteLine("Valid secret key set."); }

		textbox_pk->Enabled = false;	// dim the entry field ("locked in" PK)
		StringBuilder^ sb = gcnew StringBuilder();
		for (unsigned short i = 0; i < textbox_pk->Text->Length; ++i)
			sb->Append("*");
		textbox_pk->Text = sb->ToString();	// replace entered text with asterisks
	}
	else
	{ // parse error:
		Console::WriteLine("Error parsing Skey from textbox");
		//LOG_F(WARNING, gVerbosity==V_DEBUG, Console::WriteLine("Error parsing Skey from textbox");
		// ...
	}
}

private: System::Void lbl_passkey_Click(System::Object^ sender, System::EventArgs^ e) {
	if (gVerbosity != V_DEBUG)
		return;

	Console::WriteLine("Password field contains  {0}", textbox_pass->Text);
}



};	//class MakeKeystore

} //namespace Cosmic

#endif