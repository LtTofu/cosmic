// ConfigSoloMining.h for COSMiC V4
// 2020-2021 LtTofu
#pragma once
#ifndef CONFIGSOLOMINING
#pragma message("Including " __FILE__ ", last modified " __TIMESTAMP__ ".")
#define CONFIGSOLOMINING
//		A WinForm that loads keys relevant to Solo Mode and allows the user to edit them.
//		Serves as a hub for Solo mode config:	loading keystores, creating them from user input
//		(via MakeKeystore form), displaying the Ethereum public address (derived from the keystore).
//
//		`Save Changes` does just that: saves settings out to the Configuration, 
//		from this Form's controls' values.

//		Discard button: doesn't save to the Config.

//	  [TODO]: saving out independent keystore files (in encrypted form) and loading them.
//	  SEE: MakeKeystore.h re: keystore salting/encryption specifics.
#include "msclr/marshal_cppstd.h"
#include <msclr/marshal.h>
#include <bitcoin-cryptography-library/cpp/Uint256.hpp>
//#include <libsodium/sodium.h>
#include "loguru/loguru.hpp"	//
#include "defs.hpp"		// #include "defs.h"

#include <cinttypes>  // was: #include <stdint.h>
#include <stdio.h>
#include <iostream>
#define __STDC_WANT_LIB_EXT1__ 1
#include <string>
#include <sstream>
#include <iomanip>
#include <cinttypes>  // was: #include <stdint.h>. for format specifiers for 64-bit unsigned int's

#include "Forms/MakeKeystore.h"	// ConfigSoloMining will instantiate form `MakeKeystore`, which prompts
							// the user for a key & password and encrypts it for the Configuration.
#include "Keystore.h"	//<----
//#include "net_solo.h"
// [TODO / WIP]: condense and revise the functions in this file. this will be fun, it already works nicely. <--

extern int gLibSodiumStatus;
extern int gSolo_ChainID;
extern Uint256 gSk256;  // net_solo.h
//extern std::string gStr_SoloEthAddress;
//extern std::string gStr_ContractAddress;
//extern uint64_t gU64_GasPrice;
//extern uint64_t gU64_GasLimit;

namespace Cosmic {

	using namespace System;
	using namespace System::Diagnostics;
	using namespace System::Configuration;
	using namespace System::Windows::Forms;
	using namespace System::Security;
	using namespace System::Security::Cryptography;
	using namespace System::IO;
	
	using namespace msclr::interop;
	using namespace loguru;

// TODO: consolidate (see: comms_solo.cpp, defs.h)
#define SAVE_PKEY_TOCONFIG	1
#define SAVE_PKEY_TOKEYFILE 0
#define SALT_LENGTH 32
//constexpr auto PUBKEY_LENGTH = 32;				// length, in bytes, of a key used to decrypt the keystore (from a user-supplied password)  <-- [MOVEME] ?
//constexpr auto CIPHERTEXT_BYTESLENGTH = 48;		// 80;		// [MOVEME]
////constexpr auto CIPHERTEXT_STRINGLENGTH = 160;	// for reference only!

//
// consolidate. already including net_solo.h <-
//#include "net_solo.h"
//
const std::string gStr_SoloNodeAddressDefault = "https://mainnet.infura.io/v3/ea35aa7ec382411e88d3b698e7a3d197";  // constexpr?

//uint8_t* hexstr2uint8(const char* cstring);				//or #include "util.hpp"
//bool Cosmic::CosmicWind::LoadSKeyFromConfig(const std::string str_password);	//in CosmicWind.cpp
//unsigned short SaveEncPrivateKey(const unsigned short saveWhere, const std::string& encPKey, const std::string& nonce, 
//		const unsigned int encPKeyLen, const int kdfAlgo, const std::string& salt);

	using namespace System;
	using namespace System::Windows::Forms;
//	using namespace System::ComponentModel;
//	using namespace System::Collections;
//	//using namespace System::Data;
//	using namespace System::Drawing;

	/// <summary>
	/// Summary for ConfigSoloMining
	/// </summary>
	public ref class ConfigSoloMining : public System::Windows::Forms::Form
	{
	public:
		ConfigSoloMining(void)
		{
			InitializeComponent();
			// constructor
		}

	protected:
		/// <summary>
		/// destructor
		/// </summary>
		~ConfigSoloMining()
		{
			// destructor
			if (components)
			{
				delete components;
			}
		}

//	public:
	//void OpenKeyFile(void)
	//{ // (wip: load a keystore from a file saved by COSMiC. [todo]: load some keystore file formats generated elsewhere?)
	//	/* [REF]
	//		Filename = "Select a text file",
	//		Filter = "Text files (*.txt)|*.txt",
	//		Title = "Open text file"  */
	//	//IO::Stream^ newStream;
	//	OpenFileDialog^ getKeyFile = gcnew OpenFileDialog();
	//	//getKeyFile->InitialDirectory = @"";

	//	System::Windows::Forms::DialogResult rslt = getKeyFile->ShowDialog();
	//	if (rslt == System::Windows::Forms::DialogResult::OK)
	//	{
	//		String^ filePath = getKeyFile->FileName;
	//		StreamReader^ sr = gcnew StreamReader(filePath);
	//		Console::WriteLine("file: " + getKeyFile->FileName);
	//		Console::WriteLine("contents: " + sr->ReadLine());

	//		std::string str_pass = "??";
	//		uint8_t bytes_pwhash[32] = { 0 };
	//		//print_bytes(bytes_pwhash, 32, "password hash");

	//		// get keccak256 hash of password:
	//		printf("strlen(pw):  %zu \n", (int)(strlen(pass)));
	//		uint16_t theLength = (uint16_t)strlen(pass);
	//		//
	//		// ... Unfinished  [TODO]
	//	}
	//}

	private: System::Windows::Forms::Button^  button2;
	private: System::Windows::Forms::Button^ button3;
	private: System::Windows::Forms::PictureBox^  pictureBox1;
	private: System::Windows::Forms::GroupBox^ groupbox_settings;
	private: System::Windows::Forms::Label^  label6;
	private: System::Windows::Forms::CheckBox^  checkbox_useapi;
	private: System::Windows::Forms::Label^  label1;
	private: System::Windows::Forms::Label^  label3;
	private: System::Windows::Forms::Label^ label4;
	private: System::Windows::Forms::TextBox^  textbox_out;
	private: System::Windows::Forms::GroupBox^ groupbox_ethaccount;
	private: System::Windows::Forms::TextBox^  textBox3;
	private: System::Windows::Forms::Label^  label2;

	private: System::Windows::Forms::CheckBox^  checkbox_savepass;
	private: System::Windows::Forms::Button^  button8;
	private: System::Windows::Forms::TextBox^  textbox_passphrase;
	private: System::Windows::Forms::Button^  button_readkey;
	private: System::Windows::Forms::HelpProvider^  helpProvider;
	private: System::Windows::Forms::Button^  button4;
	private: System::Windows::Forms::TableLayoutPanel^ tableLayoutPanel1;
	private: System::Windows::Forms::TextBox^ textbox_nodeaddr;
	private: System::Windows::Forms::NumericUpDown^ nud_chainid;
	private: System::Windows::Forms::NumericUpDown^ nud_gaslimit;
	private: System::Windows::Forms::Label^ label5;
	private: System::Windows::Forms::TextBox^ textbox_contractaddress;
	private: System::Windows::Forms::NumericUpDown^ nud_gasprice;
	private:
		/// <summary>
		/// Required designer variable.
		/// </summary>
		System::ComponentModel::Container ^components;

#pragma region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		void InitializeComponent(void)
		{
			System::ComponentModel::ComponentResourceManager^ resources = (gcnew System::ComponentModel::ComponentResourceManager(ConfigSoloMining::typeid));
			this->button2 = (gcnew System::Windows::Forms::Button());
			this->button3 = (gcnew System::Windows::Forms::Button());
			this->label4 = (gcnew System::Windows::Forms::Label());
			this->label6 = (gcnew System::Windows::Forms::Label());
			this->pictureBox1 = (gcnew System::Windows::Forms::PictureBox());
			this->groupbox_settings = (gcnew System::Windows::Forms::GroupBox());
			this->tableLayoutPanel1 = (gcnew System::Windows::Forms::TableLayoutPanel());
			this->textbox_nodeaddr = (gcnew System::Windows::Forms::TextBox());
			this->nud_chainid = (gcnew System::Windows::Forms::NumericUpDown());
			this->label5 = (gcnew System::Windows::Forms::Label());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->label3 = (gcnew System::Windows::Forms::Label());
			this->textbox_contractaddress = (gcnew System::Windows::Forms::TextBox());
			this->nud_gasprice = (gcnew System::Windows::Forms::NumericUpDown());
			this->nud_gaslimit = (gcnew System::Windows::Forms::NumericUpDown());
			this->checkbox_useapi = (gcnew System::Windows::Forms::CheckBox());
			this->textbox_out = (gcnew System::Windows::Forms::TextBox());
			this->groupbox_ethaccount = (gcnew System::Windows::Forms::GroupBox());
			this->button4 = (gcnew System::Windows::Forms::Button());
			this->button8 = (gcnew System::Windows::Forms::Button());
			this->textBox3 = (gcnew System::Windows::Forms::TextBox());
			this->checkbox_savepass = (gcnew System::Windows::Forms::CheckBox());
			this->textbox_passphrase = (gcnew System::Windows::Forms::TextBox());
			this->button_readkey = (gcnew System::Windows::Forms::Button());
			this->label2 = (gcnew System::Windows::Forms::Label());
			this->helpProvider = (gcnew System::Windows::Forms::HelpProvider());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->BeginInit();
			this->groupbox_settings->SuspendLayout();
			this->tableLayoutPanel1->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->nud_chainid))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->nud_gasprice))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->nud_gaslimit))->BeginInit();
			this->groupbox_ethaccount->SuspendLayout();
			this->SuspendLayout();
			// 
			// button2
			// 
			this->button2->DialogResult = System::Windows::Forms::DialogResult::OK;
			this->button2->FlatAppearance->BorderColor = System::Drawing::Color::Blue;
			this->button2->FlatAppearance->BorderSize = 2;
			this->button2->Location = System::Drawing::Point(440, 460);
			this->button2->Name = L"button2";
			this->button2->Size = System::Drawing::Size(112, 23);
			this->button2->TabIndex = 2;
			this->button2->Text = L"Save Settings";
			this->button2->UseVisualStyleBackColor = true;
			this->button2->Click += gcnew System::EventHandler(this, &ConfigSoloMining::button2_Click);
			// 
			// button3
			// 
			this->button3->DialogResult = System::Windows::Forms::DialogResult::Cancel;
			this->button3->Location = System::Drawing::Point(313, 460);
			this->button3->Name = L"button3";
			this->button3->Size = System::Drawing::Size(112, 23);
			this->button3->TabIndex = 4;
			this->button3->Text = L"Discard Changes";
			this->button3->UseVisualStyleBackColor = true;
			// 
			// label4
			// 
			this->label4->Anchor = System::Windows::Forms::AnchorStyles::Left;
			this->label4->AutoSize = true;
			this->label4->Location = System::Drawing::Point(3, 32);
			this->label4->Name = L"label4";
			this->label4->Size = System::Drawing::Size(91, 13);
			this->label4->TabIndex = 2;
			this->label4->Text = L"Contract Address:";
			// 
			// label6
			// 
			this->label6->Anchor = System::Windows::Forms::AnchorStyles::Left;
			this->label6->AutoSize = true;
			this->label6->Location = System::Drawing::Point(3, 110);
			this->label6->Name = L"label6";
			this->label6->Size = System::Drawing::Size(53, 13);
			this->label6->TabIndex = 8;
			this->label6->Text = L"Gas Limit:";
			// 
			// pictureBox1
			// 
			this->pictureBox1->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"pictureBox1.Image")));
			this->pictureBox1->Location = System::Drawing::Point(12, 460);
			this->pictureBox1->Name = L"pictureBox1";
			this->pictureBox1->Size = System::Drawing::Size(18, 25);
			this->pictureBox1->TabIndex = 4;
			this->pictureBox1->TabStop = false;
			this->pictureBox1->Visible = false;
			// 
			// groupbox_settings
			// 
			this->groupbox_settings->Controls->Add(this->tableLayoutPanel1);
			this->groupbox_settings->Location = System::Drawing::Point(12, 16);
			this->groupbox_settings->Name = L"groupbox_settings";
			this->groupbox_settings->Size = System::Drawing::Size(562, 192);
			this->groupbox_settings->TabIndex = 0;
			this->groupbox_settings->TabStop = false;
			this->groupbox_settings->Text = L"Ethereum Node / Solo Mining Settings";
			// 
			// tableLayoutPanel1
			// 
			this->tableLayoutPanel1->ColumnCount = 2;
			this->tableLayoutPanel1->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle()));
			this->tableLayoutPanel1->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle()));
			this->tableLayoutPanel1->Controls->Add(this->textbox_nodeaddr, 1, 0);
			this->tableLayoutPanel1->Controls->Add(this->nud_chainid, 1, 2);
			this->tableLayoutPanel1->Controls->Add(this->label6, 0, 4);
			this->tableLayoutPanel1->Controls->Add(this->label5, 0, 2);
			this->tableLayoutPanel1->Controls->Add(this->label1, 0, 0);
			this->tableLayoutPanel1->Controls->Add(this->label4, 0, 1);
			this->tableLayoutPanel1->Controls->Add(this->label3, 0, 3);
			this->tableLayoutPanel1->Controls->Add(this->textbox_contractaddress, 1, 1);
			this->tableLayoutPanel1->Controls->Add(this->nud_gasprice, 1, 3);
			this->tableLayoutPanel1->Controls->Add(this->nud_gaslimit, 1, 4);
			this->tableLayoutPanel1->Controls->Add(this->checkbox_useapi, 1, 5);
			this->tableLayoutPanel1->Location = System::Drawing::Point(10, 23);
			this->tableLayoutPanel1->Name = L"tableLayoutPanel1";
			this->tableLayoutPanel1->RowCount = 6;
			this->tableLayoutPanel1->RowStyles->Add((gcnew System::Windows::Forms::RowStyle()));
			this->tableLayoutPanel1->RowStyles->Add((gcnew System::Windows::Forms::RowStyle()));
			this->tableLayoutPanel1->RowStyles->Add((gcnew System::Windows::Forms::RowStyle()));
			this->tableLayoutPanel1->RowStyles->Add((gcnew System::Windows::Forms::RowStyle()));
			this->tableLayoutPanel1->RowStyles->Add((gcnew System::Windows::Forms::RowStyle()));
			this->tableLayoutPanel1->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute, 20)));
			this->tableLayoutPanel1->Size = System::Drawing::Size(490, 154);
			this->tableLayoutPanel1->TabIndex = 0;
			// 
			// textbox_nodeaddr
			// 
			this->textbox_nodeaddr->Dock = System::Windows::Forms::DockStyle::Fill;
			this->textbox_nodeaddr->Location = System::Drawing::Point(100, 3);
			this->textbox_nodeaddr->MaxLength = 127;
			this->textbox_nodeaddr->Name = L"textbox_nodeaddr";
			this->textbox_nodeaddr->Size = System::Drawing::Size(387, 20);
			this->textbox_nodeaddr->TabIndex = 0;
			this->textbox_nodeaddr->Text = L"https://mainnet.infura.io/v3/YOURAPIKEY";
			// 
			// nud_chainid
			// 
			this->nud_chainid->AutoSize = true;
			this->nud_chainid->Location = System::Drawing::Point(100, 55);
			this->nud_chainid->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 9999, 0, 0, 0 });
			this->nud_chainid->Name = L"nud_chainid";
			this->nud_chainid->Size = System::Drawing::Size(47, 20);
			this->nud_chainid->TabIndex = 2;
			// 
			// label5
			// 
			this->label5->Anchor = System::Windows::Forms::AnchorStyles::Left;
			this->label5->AutoSize = true;
			this->label5->Location = System::Drawing::Point(3, 58);
			this->label5->Name = L"label5";
			this->label5->Size = System::Drawing::Size(48, 13);
			this->label5->TabIndex = 4;
			this->label5->Text = L"ChainID:";
			// 
			// label1
			// 
			this->label1->Anchor = System::Windows::Forms::AnchorStyles::Left;
			this->label1->AutoSize = true;
			this->label1->Location = System::Drawing::Point(3, 6);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(77, 13);
			this->label1->TabIndex = 0;
			this->label1->Text = L"Node Address:";
			// 
			// label3
			// 
			this->label3->Anchor = System::Windows::Forms::AnchorStyles::Left;
			this->label3->AutoSize = true;
			this->helpProvider->SetHelpKeyword(this->label3, L"gasprice");
			this->helpProvider->SetHelpString(this->label3, L"Set the price per unit gas paid when submitting solutions to the Network (in GWei"
				L", or 1 billion Wei).");
			this->label3->Location = System::Drawing::Point(3, 84);
			this->label3->Name = L"label3";
			this->helpProvider->SetShowHelp(this->label3, true);
			this->label3->Size = System::Drawing::Size(89, 13);
			this->label3->TabIndex = 6;
			this->label3->Text = L"Gas Price (Gwei):";
			// 
			// textbox_contractaddress
			// 
			this->textbox_contractaddress->Dock = System::Windows::Forms::DockStyle::Fill;
			this->textbox_contractaddress->Location = System::Drawing::Point(100, 29);
			this->textbox_contractaddress->Name = L"textbox_contractaddress";
			this->textbox_contractaddress->Size = System::Drawing::Size(387, 20);
			this->textbox_contractaddress->TabIndex = 1;
			this->textbox_contractaddress->Text = L"0x2BF91c18Cd4AE9C2f2858ef9FE518180F7B5096D";
			// 
			// nud_gasprice
			// 
			this->nud_gasprice->AutoSize = true;
			this->nud_gasprice->DecimalPlaces = 1;
			this->helpProvider->SetHelpKeyword(this->nud_gasprice, L"gasprice");
			this->helpProvider->SetHelpString(this->nud_gasprice, L"Set the price per unit gas paid when submitting solutions to the Network (in GWei"
				L", or 1 billion Wei).");
			this->nud_gasprice->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 65536 });
			this->nud_gasprice->Location = System::Drawing::Point(100, 81);
			this->nud_gasprice->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 9999, 0, 0, 0 });
			this->nud_gasprice->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 65536 });
			this->nud_gasprice->Name = L"nud_gasprice";
			this->helpProvider->SetShowHelp(this->nud_gasprice, true);
			this->nud_gasprice->Size = System::Drawing::Size(56, 20);
			this->nud_gasprice->TabIndex = 3;
			this->nud_gasprice->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 5, 0, 0, 0 });
			// 
			// nud_gaslimit
			// 
			this->nud_gaslimit->AutoSize = true;
			this->nud_gaslimit->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 500, 0, 0, 0 });
			this->nud_gaslimit->Location = System::Drawing::Point(100, 107);
			this->nud_gaslimit->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 9999999, 0, 0, 0 });
			this->nud_gaslimit->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
			this->nud_gaslimit->Name = L"nud_gaslimit";
			this->nud_gaslimit->Size = System::Drawing::Size(71, 20);
			this->nud_gaslimit->TabIndex = 4;
			this->nud_gaslimit->ThousandsSeparator = true;
			this->nud_gaslimit->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 200000, 0, 0, 0 });
			// 
			// checkbox_useapi
			// 
			this->checkbox_useapi->AutoSize = true;
			this->checkbox_useapi->Enabled = false;
			this->checkbox_useapi->Location = System::Drawing::Point(100, 133);
			this->checkbox_useapi->Name = L"checkbox_useapi";
			this->checkbox_useapi->Size = System::Drawing::Size(133, 17);
			this->checkbox_useapi->TabIndex = 5;
			this->checkbox_useapi->Text = L"Use Network Gasprice";
			this->checkbox_useapi->UseVisualStyleBackColor = true;
			// 
			// textbox_out
			// 
			this->textbox_out->AcceptsReturn = true;
			this->textbox_out->BackColor = System::Drawing::SystemColors::ControlLight;
			this->textbox_out->Enabled = false;
			this->textbox_out->ForeColor = System::Drawing::SystemColors::WindowText;
			this->textbox_out->Location = System::Drawing::Point(18, 81);
			this->textbox_out->MaxLength = 512;
			this->textbox_out->Multiline = true;
			this->textbox_out->Name = L"textbox_out";
			this->textbox_out->ReadOnly = true;
			this->textbox_out->Size = System::Drawing::Size(520, 73);
			this->textbox_out->TabIndex = 1;
			this->textbox_out->Text = L"\r\n-- No Account Loaded  --\r\n \r\nPlease \"Import...\" an Ethereum account to get star"
				L"ted.";
			this->textbox_out->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			// 
			// groupbox_ethaccount
			// 
			this->groupbox_ethaccount->Controls->Add(this->button4);
			this->groupbox_ethaccount->Controls->Add(this->button8);
			this->groupbox_ethaccount->Controls->Add(this->textBox3);
			this->groupbox_ethaccount->Controls->Add(this->checkbox_savepass);
			this->groupbox_ethaccount->Controls->Add(this->textbox_out);
			this->groupbox_ethaccount->Controls->Add(this->textbox_passphrase);
			this->groupbox_ethaccount->Controls->Add(this->button_readkey);
			this->groupbox_ethaccount->Location = System::Drawing::Point(12, 214);
			this->groupbox_ethaccount->Name = L"groupbox_ethaccount";
			this->groupbox_ethaccount->Size = System::Drawing::Size(562, 229);
			this->groupbox_ethaccount->TabIndex = 1;
			this->groupbox_ethaccount->TabStop = false;
			this->groupbox_ethaccount->Text = L"Ethereum Account";
			// 
			// button4
			// 
			this->button4->Enabled = false;
			this->button4->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"button4.Image")));
			this->button4->Location = System::Drawing::Point(373, 192);
			this->button4->Name = L"button4";
			this->button4->Size = System::Drawing::Size(165, 23);
			this->button4->TabIndex = 6;
			this->button4->Text = L"  Import from Keystore File...";
			this->button4->TextImageRelation = System::Windows::Forms::TextImageRelation::ImageBeforeText;
			this->button4->UseVisualStyleBackColor = true;
			// 
			// button8
			// 
			this->button8->FlatAppearance->BorderSize = 3;
			this->button8->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"button8.Image")));
			this->button8->Location = System::Drawing::Point(373, 163);
			this->button8->Name = L"button8";
			this->button8->Size = System::Drawing::Size(165, 23);
			this->button8->TabIndex = 5;
			this->button8->Text = L"  Import from Private Key...";
			this->button8->TextImageRelation = System::Windows::Forms::TextImageRelation::ImageBeforeText;
			this->button8->UseVisualStyleBackColor = true;
			this->button8->Click += gcnew System::EventHandler(this, &ConfigSoloMining::importButton_Click);
			// 
			// textBox3
			// 
			this->textBox3->BorderStyle = System::Windows::Forms::BorderStyle::None;
			this->textBox3->Cursor = System::Windows::Forms::Cursors::Arrow;
			this->textBox3->Location = System::Drawing::Point(19, 27);
			this->textBox3->MaxLength = 512;
			this->textBox3->Multiline = true;
			this->textBox3->Name = L"textBox3";
			this->textBox3->ReadOnly = true;
			this->textBox3->Size = System::Drawing::Size(529, 48);
			this->textBox3->TabIndex = 0;
			this->textBox3->TabStop = false;
			this->textBox3->Text = resources->GetString(L"textBox3.Text");
			// 
			// checkbox_savepass
			// 
			this->checkbox_savepass->AutoSize = true;
			this->checkbox_savepass->Enabled = false;
			this->checkbox_savepass->Location = System::Drawing::Point(18, 196);
			this->checkbox_savepass->Name = L"checkbox_savepass";
			this->checkbox_savepass->Size = System::Drawing::Size(162, 17);
			this->checkbox_savepass->TabIndex = 3;
			this->checkbox_savepass->Text = L"Save Password (less secure)";
			this->checkbox_savepass->UseVisualStyleBackColor = true;
			// 
			// textbox_passphrase
			// 
			this->helpProvider->SetHelpString(this->textbox_passphrase, L"For the best security, a longer password with character variety is recommended. S"
				L"uggestions include using upper and lowercase letters and at least one number and"
				L" special character.");
			this->textbox_passphrase->Location = System::Drawing::Point(18, 165);
			this->textbox_passphrase->MaxLength = 250;
			this->textbox_passphrase->Name = L"textbox_passphrase";
			this->helpProvider->SetShowHelp(this->textbox_passphrase, true);
			this->textbox_passphrase->Size = System::Drawing::Size(339, 20);
			this->textbox_passphrase->TabIndex = 2;
			this->textbox_passphrase->Text = L"Enter Password...";
			this->textbox_passphrase->Click += gcnew System::EventHandler(this, &ConfigSoloMining::textbox_passphrase_Click);
			this->textbox_passphrase->TextChanged += gcnew System::EventHandler(this, &ConfigSoloMining::textbox_passphrase_TextChanged);
			this->textbox_passphrase->Enter += gcnew System::EventHandler(this, &ConfigSoloMining::textbox_passphrase_Enter);
			// 
			// button_readkey
			// 
			this->helpProvider->SetHelpString(this->button_readkey, L"Loads the Ethereum account loaded into the miner, using its matching password.\\n\\"
				L"nBefore Loading, \"Import\" an account using one of the buttons to the right.");
			this->button_readkey->Location = System::Drawing::Point(260, 192);
			this->button_readkey->Name = L"button_readkey";
			this->helpProvider->SetShowHelp(this->button_readkey, true);
			this->button_readkey->Size = System::Drawing::Size(98, 22);
			this->button_readkey->TabIndex = 4;
			this->button_readkey->Text = L"Load";
			this->button_readkey->UseVisualStyleBackColor = true;
			this->button_readkey->Click += gcnew System::EventHandler(this, &ConfigSoloMining::button_readkey_Click);
			// 
			// label2
			// 
			this->label2->AutoSize = true;
			this->label2->Location = System::Drawing::Point(36, 465);
			this->label2->Name = L"label2";
			this->label2->Size = System::Drawing::Size(97, 13);
			this->label2->TabIndex = 6;
			this->label2->Text = L"a tip would go here";
			this->label2->Visible = false;
			// 
			// ConfigSoloMining
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(586, 497);
			this->Controls->Add(this->groupbox_ethaccount);
			this->Controls->Add(this->groupbox_settings);
			this->Controls->Add(this->button3);
			this->Controls->Add(this->pictureBox1);
			this->Controls->Add(this->button2);
			this->Controls->Add(this->label2);
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::FixedDialog;
			this->HelpButton = true;
			this->Icon = (cli::safe_cast<System::Drawing::Icon^>(resources->GetObject(L"$this.Icon")));
			this->MaximizeBox = false;
			this->MinimizeBox = false;
			this->Name = L"ConfigSoloMining";
			this->ShowIcon = false;
			this->ShowInTaskbar = false;
			this->SizeGripStyle = System::Windows::Forms::SizeGripStyle::Hide;
			this->StartPosition = System::Windows::Forms::FormStartPosition::CenterParent;
			this->Text = L"COSMiC - Configure Solo Mining";
			this->Load += gcnew System::EventHandler(this, &ConfigSoloMining::ConfigSoloMining_Load);
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->EndInit();
			this->groupbox_settings->ResumeLayout(false);
			this->tableLayoutPanel1->ResumeLayout(false);
			this->tableLayoutPanel1->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->nud_chainid))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->nud_gasprice))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->nud_gaslimit))->EndInit();
			this->groupbox_ethaccount->ResumeLayout(false);
			this->groupbox_ethaccount->PerformLayout();
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion

	// Save Settings button clicked
	private: System::Void button2_Click(System::Object^  sender, System::EventArgs^  e)
	{
		System::Configuration::Configuration ^configHandle = ConfigurationManager::OpenExeConfiguration(ConfigurationUserLevel::None);
		msclr::interop::marshal_context marshalctx;

		// [todo]:  consider using TryParse here instead of Convert::
		if (Convert::ToDouble(nud_gasprice->Value) >= 0)  // TODO: and max check? Already enforced by control value min/max
		{
			gU64_GasPrice = Convert::ToUInt64(nud_gasprice->Value*1000000000);  // set internally
//			String^ MStr = gU64_GasPrice.ToString("X2");
//			gStr_GasPrice_hex = marshalctx.marshal_as<std::string>(MStr);
			std::stringstream sst;
			sst << std::setfill('0') << std::setw(2) << std::hex << gU64_GasPrice;
			gStr_GasPrice_hex = sst.str();
			if ((gStr_GasPrice_hex.length() % 2) != 0)  // uneven length: pad with 0
				gStr_GasPrice_hex = "0" + gStr_GasPrice_hex;

			if (gVerbosity > V_NORM) {
				std::cout << "set User GasPrice (uint64t):  " << std::hex << gU64_GasPrice << ", as string:  " << gStr_GasPrice_hex << std::endl;
				std::cout << "String Form:  " << gStr_GasPrice_hex << std::endl; }

			// save to config file
			if (ConfigurationManager::AppSettings["userGasPrice"])
				configHandle->AppSettings->Settings->Remove("userGasPrice");
			configHandle->AppSettings->Settings->Add("userGasPrice", Convert::ToString(nud_gasprice->Value));
		}
		else
		{
			MessageBox::Show("Invalid gas price. Please check the setting or 'Discard Changes'. ", 
				"COSMiC", MessageBoxButtons::OK, MessageBoxIcon::Information);
			this->DialogResult = System::Windows::Forms::DialogResult::None;  // don't close the form
			nud_gasprice->Focus();  // focus the numericUpDown control that needs fixing.
			return;
		}

		// store user gas limit
		// Fixme: revise this section
		if (Convert::ToDouble(nud_gaslimit->Value) >= 0) {
			gU64_GasLimit = Convert::ToUInt64(nud_gaslimit->Value);

			// get and store hex repres'n that node/network expects (RLP encoding done at submit-time w/ rest of payload.)
			// NOTE: if you improve this conversion code (likely), please update its counterpart in CosmicWind_Load() handler
			std::stringstream sst;
			sst << std::setfill('0') << std::setw(2) << std::hex << gU64_GasLimit;
			gStr_GasLimit_hex = sst.str();
			if ((gStr_GasLimit_hex.length() % 2) != 0)  // uneven length: pad with 0 (FIXME: better approach)
				gStr_GasLimit_hex = "0" + gStr_GasLimit_hex;

			if (gVerbosity > V_NORM) {
				printf("set User GasLimit:  DEC: %" PRIu64 "  HEX: %" PRIx64 " \n", gU64_GasLimit, gU64_GasLimit);
				if (gVerbosity == V_DEBUG)  printf("string form:  %s \n", gStr_GasLimit_hex.c_str() );  }

			// save to config file
			if (ConfigurationManager::AppSettings["userGasLimit"])
				configHandle->AppSettings->Settings->Remove("userGasLimit");
			configHandle->AppSettings->Settings->Add("userGasLimit", Convert::ToString(nud_gaslimit->Value));
		}
		else {
			MessageBox::Show("An invalid Gas Limit has been entered. Please check the setting or click 'Discard Changes'. ", "COSMiC", MessageBoxButtons::OK, MessageBoxIcon::Information);
			this->DialogResult = System::Windows::Forms::DialogResult::None;  // don't close the form
			nud_gaslimit->Focus();  // focus the numericUpDown control that needs fixing.
			return;
		}

		// there is also gStr_Solo_GasPrice_FromNode < --- (TODO)

		// i'm getting to it GEEZ
		// New sketch
		gSolo_ChainID = (int)nud_chainid->Value;  // save it globally

		if (ConfigurationManager::AppSettings["chainID"]) // <-- better?				...... check for key before remove (don't exception.)
			configHandle->AppSettings->Settings->Remove("chainID");							// remove key (if present)
		configHandle->AppSettings->Settings->Add("chainID", nud_chainid->Value.ToString() );  // write to config
		//


		// save the ethereum network node address/port internally
		// FIXME: do error checking of input node addr/port!
		if (textbox_nodeaddr->Text == "") { /* currently just checks if empty */
			printf("Invalid node address. \n");
			MessageBox::Show("The Ethereum node address entered appears invalid. Please check the setting or click 'Discard Changes'. ", "COSMiC", MessageBoxButtons::OK, MessageBoxIcon::Information);
			this->DialogResult = System::Windows::Forms::DialogResult::None;
			textbox_nodeaddr->Focus();  // focus the field that needs fixing
			return;
		}
		gStr_SoloNodeAddress = marshalctx.marshal_as<std::string>(textbox_nodeaddr->Text); // set internally

		// save it to config
		if (ConfigurationManager::AppSettings["soloNetworkNode"])
			configHandle->AppSettings->Settings->Remove("soloNetworkNode");
		configHandle->AppSettings->Settings->Add("soloNetworkNode", Convert::ToString(textbox_nodeaddr->Text));



		const std::string s1 = marshalctx.marshal_as<std::string>(textbox_contractaddress->Text);
		if (s1.length() == 40 && s1.find_first_not_of(DEF_HEXCHARS) == std::string::npos)
			gStr_ContractAddress = "0x" + s1;  // prepend with 0x
		else if (s1.substr(0, 2) == "0x" && s1.length() == 42 && s1.substr(2).find_first_not_of(DEF_HEXCHARS) == std::string::npos)
			gStr_ContractAddress = s1;		   // set it globally.
		else {
			MessageBox::Show("The Contract Address provided does not appear to be valid. Expected 0x followed by 40 hex characters (0-9, A-F).", "COSMiC",
				MessageBoxButtons::OK, MessageBoxIcon::Information);
			this->DialogResult = System::Windows::Forms::DialogResult::None;
			textbox_contractaddress->Focus();
			return;
		}

		// address looks valid and was set globally
		String^ ScratchMStr = gcnew String(gStr_ContractAddress.c_str());
		if (ConfigurationManager::AppSettings["soloContractAddress"])
			configHandle->AppSettings->Settings->Remove("soloContractAddress");
		configHandle->AppSettings->Settings->Add("soloContractAddress", ScratchMStr );
		
		// save out the configuration
		configHandle->Save(ConfigurationSaveMode::Modified);
		ConfigurationManager::RefreshSection("appSettings");
		Console::WriteLine("Updated config.");

		// update solo mining address (EDIT: now updated prior to form close)
		//if (scratch_soloaddr != "0x00")
		//	gStr_SoloEthAddress = scratch_soloaddr;
	}

	private: System::Void ConfigSoloMining_Load(System::Object^  sender, System::EventArgs^  e)
	{
		// Form Loaded
		msclr::interop::marshal_context marshalctx;  // for marshalling types
		unsigned short i{ 0 };
		
		LOG_IF_F(INFO, NORMALVERBOSITY, "Reading Contract Address from Config.");
		System::Configuration::Configuration ^configHndl = ConfigurationManager::OpenExeConfiguration(ConfigurationUserLevel::None);
		textbox_contractaddress->Text = ConfigurationManager::AppSettings["soloContractAddress"];

		LOG_IF_F(INFO, NORMALVERBOSITY, "Reading node address/port from Config.");
		if (ConfigurationManager::AppSettings["soloNetworkNode"])
			textbox_nodeaddr->Text = ConfigurationManager::AppSettings["soloNetworkNode"];

		LOG_IF_F(INFO, NORMALVERBOSITY, "Reading User Gas Price from Config.");
		if (ConfigurationManager::AppSettings["userGasPrice"])
			nud_gasprice->Value = Convert::ToDecimal(ConfigurationManager::AppSettings["userGasPrice"]);

		LOG_IF_F(INFO, NORMALVERBOSITY, "Reading User Gas Limit from Config.");
		if (ConfigurationManager::AppSettings["userGasLimit"])
			nud_gaslimit->Value = Convert::ToDecimal(ConfigurationManager::AppSettings["userGasLimit"]);

		LOG_IF_F(INFO, NORMALVERBOSITY, "Reading ChainID from Config.");
		if (ConfigurationManager::AppSettings["chainID"])
			nud_chainid->Value = Convert::ToDecimal(ConfigurationManager::AppSettings["chainID"]);
		else {
			LOG_IF_F(INFO, NORMALVERBOSITY, "chainID not found in Config: using default %u (Ethereum MainNet). ", CHAINID_ETHEREUM);
			nud_chainid->Value = Convert::ToDecimal(CHAINID_ETHEREUM);
		}

		LOG_IF_F(INFO, HIGHVERBOSITY, "Checking for encrypted Ethereum Account in Config.");
		if (ConfigurationManager::AppSettings["encryptedPKbytes"] &&
			ConfigurationManager::AppSettings["encryptedPKnonce"])  // TODO: further checking? keys must exist
		{
			//std::string encryptedPKbytes_str = marshalctx.marshal_as<std::string>(ConfigurationManager::AppSettings["encryptedPKbytes"]);
			//std::string encryptedPKnonce_str = marshalctx.marshal_as<std::string>(ConfigurationManager::AppSettings["encryptedPKnonce"]);
			textbox_out->BackColor = textbox_out->DefaultBackColor;
			textbox_out->Text = "-- Encrypted Ethereum Account -- \r\n\n";
			textbox_out->Text += "To use, please enter matching passphrase below and click `Load`. ";
			textbox_out->Enabled = true;
		}
		else
			LOG_IF_F(INFO, NORMALVERBOSITY, "encryptedPKbytes or encryptedPKnonce not present in Config.");
	}

// Read Keystore button
private: System::Void button_readkey_Click(System::Object^  sender, System::EventArgs^  e)
{
	// Read in the stored Ethereum Account for Solo Mining from the Configuration.
	msclr::interop::marshal_context marshalctx;  // for marshalling types
	std::string password = marshalctx.marshal_as<std::string>(textbox_passphrase->Text);

	textbox_out->Enabled = true;
	String^ oldTxt = textbox_out->Text;
	textbox_out->Text = "\r\n-- Please Wait... --";
	textbox_out->Refresh();						 // so we see the update
	button_readkey->Text = "Decrypting...";
	button_readkey->Refresh();					 // ditto
	if (Keystore::LoadSKeyFromConfig(password) == true)  // with phrase to generate unlock key
	{
		textbox_out->BackColor = textbox_out->DefaultBackColor;
		textbox_out->Enabled = true;
		textbox_out->BackColor = System::Drawing::Color::FromArgb(	static_cast<System::Int32>(static_cast<System::Byte>(225)), 
																	static_cast<System::Int32>(static_cast<System::Byte>(255)), 
																	static_cast<System::Int32>(static_cast<System::Byte>(226)) );	//light green indicates "ready"
		textbox_out->Text = "\r\n-- Encrypted Ethereum Account " + gcnew String(gStr_SoloEthAddress.c_str()) + " --";
		textbox_passphrase->Enabled = false;	 //if acct loaded successfully, disable pw input field
		button_readkey->Enabled = false;		 //also disable the associated account load button
	}
	else
		textbox_out->Text = oldTxt;

	button_readkey->Text = "Load";
}

private: System::Void importButton_Click(System::Object^  sender, System::EventArgs^  e)
{
	// Import Keystore button (...)
	Cosmic::MakeKeystore^ MakeKeystoreFormInst = gcnew Cosmic::MakeKeystore();
	this->Enabled = false;  // dim form
	MakeKeystoreFormInst->ShowDialog();
	this->Enabled = true;	// undim form

	String^ scratchMStr = "";
	if (MakeKeystoreFormInst->DialogResult == System::Windows::Forms::DialogResult::OK)
	{
		scratchMStr = MakeKeystoreFormInst->GetPassword();
		if (scratchMStr != "" /* check max length? enforced by form's textbox */)
		{
			textbox_passphrase->Text = scratchMStr;	 // populate textbox <--

			textbox_passphrase->Enabled = true;
			checkbox_savepass->Enabled = true;		 // enabled, not checked
			button_readkey->Enabled = true;			 // "Load" button

			button_readkey->PerformClick();			 // load account so user can start mining!
		}
	}
}

private: System::Void textbox_passphrase_TextChanged(System::Object^  sender, System::EventArgs^  e)
{
	textbox_passphrase->Text == "Enter Password..." ? textbox_passphrase->PasswordChar = NULL : textbox_passphrase->PasswordChar = '-';
}

private: System::Void textbox_passphrase_Enter(System::Object^  sender, System::EventArgs^  e)
{
	textbox_passphrase->SelectAll();
}
private: System::Void textbox_passphrase_Click(System::Object^  sender, System::EventArgs^  e)
{
	textbox_passphrase->SelectAll();
}

};	//class ConfigSoloMining


} //namespace

#else
#pragma message("Not re-including CONFIGSOLOMINING !")
#endif	//CONFIGSOLOMINING