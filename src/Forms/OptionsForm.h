#pragma once
// OptionsForm : General application and Pool Mode options
// 2020 LtTofu

#include <iostream>
#include <stdio.h>
#include <string>
#include <msclr\marshal_cppstd.h>		// for String^ to std::string conversion
#include "util.hpp"

using namespace System;
using namespace System::Windows;
using namespace System::Windows::Forms;

extern bool gCudaSolving;
extern unsigned short gVerbosity;
extern std::string gStr_PoolHTTPAddress;

//#include "net_pool.h"
//#include "net_solo.h" -or- "network.hpp"  [idea/wip]
extern std::string gStr_MinerEthAddress;

extern unsigned int gNetInterval;
extern double gAutoDonationPercent;
extern int gDiffUpdateFrequency;
// ^ consolidating...


bool checkString ( const std::string& theStr, const size_t expectedLength,
	const bool requireHexSpecifier, const bool expectHex );  // misc.cpp

#include "defs.hpp"

#include "network.hpp"

namespace Cosmic {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	//using namespace System::Data;
	using namespace System::Drawing;
	using namespace System::Configuration;

	/// <summary>
	/// General and Pool Mining Settings Form
	/// </summary>
	public ref class OptionsForm : public System::Windows::Forms::Form
	{
	public:
		OptionsForm(void)
		{
			//constructor
			InitializeComponent();
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~OptionsForm()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::TextBox^ etext_poolurl;
	private: System::Windows::Forms::Button^ but_discard;
	private: System::Windows::Forms::Button^ but_savesettings;
	private: System::Windows::Forms::TextBox^ etext_mineraddress;
	private: System::Windows::Forms::ToolTip^ toolTip1;
	private: System::Windows::Forms::Label^ lbl_optionpoolurl;
	private: System::Windows::Forms::Label^ lbl_ethaddress;

	private: System::Windows::Forms::Label^ lbl_donation;
	private: System::Windows::Forms::NumericUpDown^ nud_donatepct;
	private: System::Windows::Forms::GroupBox^ groupBox1;
	private: System::Windows::Forms::PictureBox^ pictureBox1;
	private: System::Windows::Forms::GroupBox^ groupBox3;

	private: System::Windows::Forms::TrackBar^ slider_netinterval;

	private: System::Windows::Forms::GroupBox^ groupBox4;
	private: System::Windows::Forms::CheckBox^ ckbox_autostart;
	private: System::Windows::Forms::TrackBar^ slider_diffupdate;
	private: System::Windows::Forms::Label^ label10;
	private: System::Windows::Forms::Label^ labeldiffupdate;


	private: System::Windows::Forms::ComboBox^ combobox_verbosity;
	private: System::Windows::Forms::Label^ label3;
	private: System::Windows::Forms::TableLayoutPanel^ tableLayoutPanel2;
	private: System::Windows::Forms::GroupBox^ groupBox2;
	private: System::Windows::Forms::Button^ button1;

	private: System::Windows::Forms::Label^ lbl_netinterval_indicator;


	private: System::Windows::Forms::Label^ label1;
	private: System::Windows::Forms::TrackBar^ slider_netinterval_solo;
	private: System::Windows::Forms::Label^ lbl_diffupdate_indicator;
	private: System::Windows::Forms::Label^ lbl_netinterval_solo_indicator;

	private: System::ComponentModel::IContainer^ components;
		   //protected:
		   //private:
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
			   this->etext_poolurl = (gcnew System::Windows::Forms::TextBox());
			   this->but_discard = (gcnew System::Windows::Forms::Button());
			   this->but_savesettings = (gcnew System::Windows::Forms::Button());
			   this->etext_mineraddress = (gcnew System::Windows::Forms::TextBox());
			   this->toolTip1 = (gcnew System::Windows::Forms::ToolTip(this->components));
			   this->lbl_ethaddress = (gcnew System::Windows::Forms::Label());
			   this->slider_netinterval = (gcnew System::Windows::Forms::TrackBar());
			   this->lbl_optionpoolurl = (gcnew System::Windows::Forms::Label());
			   this->lbl_donation = (gcnew System::Windows::Forms::Label());
			   this->nud_donatepct = (gcnew System::Windows::Forms::NumericUpDown());
			   this->label3 = (gcnew System::Windows::Forms::Label());
			   this->ckbox_autostart = (gcnew System::Windows::Forms::CheckBox());
			   this->combobox_verbosity = (gcnew System::Windows::Forms::ComboBox());
			   this->label10 = (gcnew System::Windows::Forms::Label());
			   this->slider_diffupdate = (gcnew System::Windows::Forms::TrackBar());
			   this->labeldiffupdate = (gcnew System::Windows::Forms::Label());
			   this->lbl_netinterval_indicator = (gcnew System::Windows::Forms::Label());
			   this->label1 = (gcnew System::Windows::Forms::Label());
			   this->lbl_netinterval_solo_indicator = (gcnew System::Windows::Forms::Label());
			   this->lbl_diffupdate_indicator = (gcnew System::Windows::Forms::Label());
			   this->groupBox3 = (gcnew System::Windows::Forms::GroupBox());
			   this->pictureBox1 = (gcnew System::Windows::Forms::PictureBox());
			   this->groupBox4 = (gcnew System::Windows::Forms::GroupBox());
			   this->groupBox1 = (gcnew System::Windows::Forms::GroupBox());
			   this->tableLayoutPanel2 = (gcnew System::Windows::Forms::TableLayoutPanel());
			   this->slider_netinterval_solo = (gcnew System::Windows::Forms::TrackBar());
			   this->groupBox2 = (gcnew System::Windows::Forms::GroupBox());
			   this->button1 = (gcnew System::Windows::Forms::Button());
			   (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->slider_netinterval))->BeginInit();
			   (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->nud_donatepct))->BeginInit();
			   (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->slider_diffupdate))->BeginInit();
			   this->groupBox3->SuspendLayout();
			   (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->BeginInit();
			   this->groupBox4->SuspendLayout();
			   this->groupBox1->SuspendLayout();
			   this->tableLayoutPanel2->SuspendLayout();
			   (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->slider_netinterval_solo))->BeginInit();
			   this->groupBox2->SuspendLayout();
			   this->SuspendLayout();
			   // 
			   // etext_poolurl
			   // 
			   this->etext_poolurl->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			   this->etext_poolurl->Cursor = System::Windows::Forms::Cursors::IBeam;
			   this->etext_poolurl->Location = System::Drawing::Point(18, 40);
			   this->etext_poolurl->Name = L"etext_poolurl";
			   this->etext_poolurl->Size = System::Drawing::Size(358, 20);
			   this->etext_poolurl->TabIndex = 0;
			   this->toolTip1->SetToolTip(this->etext_poolurl, L"Pool that you would like to mine to and the Port. Usually 8586. 8080 for Pools th"
				   L"at use CloudFlare.\r\nVisit your Pool\'s address, without the Port, in a Web Browse"
				   L"r to view your Shares/Payout info.");
			   // 
			   // but_discard
			   // 
			   this->but_discard->Anchor = System::Windows::Forms::AnchorStyles::Right;
			   this->but_discard->DialogResult = System::Windows::Forms::DialogResult::Cancel;
			   this->but_discard->Location = System::Drawing::Point(177, 466);
			   this->but_discard->Name = L"but_discard";
			   this->but_discard->Size = System::Drawing::Size(108, 28);
			   this->but_discard->TabIndex = 6;
			   this->but_discard->Text = L"Discard Changes";
			   this->toolTip1->SetToolTip(this->but_discard, L"No changes made in this dialog will be saved.");
			   this->but_discard->UseVisualStyleBackColor = true;
			   this->but_discard->Click += gcnew System::EventHandler(this, &OptionsForm::button1_Click);
			   // 
			   // but_savesettings
			   // 
			   this->but_savesettings->Anchor = System::Windows::Forms::AnchorStyles::Right;
			   this->but_savesettings->DialogResult = System::Windows::Forms::DialogResult::OK;
			   this->but_savesettings->FlatAppearance->BorderSize = 2;
			   this->but_savesettings->Location = System::Drawing::Point(297, 466);
			   this->but_savesettings->Name = L"but_savesettings";
			   this->but_savesettings->Size = System::Drawing::Size(108, 28);
			   this->but_savesettings->TabIndex = 5;
			   this->but_savesettings->Text = L"Save Changes";
			   this->toolTip1->SetToolTip(this->but_savesettings, L"Settings will be saved to the Configuration File.");
			   this->but_savesettings->UseVisualStyleBackColor = true;
			   this->but_savesettings->Click += gcnew System::EventHandler(this, &OptionsForm::button2_Click);
			   // 
			   // etext_mineraddress
			   // 
			   this->etext_mineraddress->Location = System::Drawing::Point(20, 86);
			   this->etext_mineraddress->Name = L"etext_mineraddress";
			   this->etext_mineraddress->Size = System::Drawing::Size(356, 20);
			   this->etext_mineraddress->TabIndex = 1;
			   this->etext_mineraddress->Text = L"0x####################";
			   // 
			   // toolTip1
			   // 
			   this->toolTip1->AutomaticDelay = 400;
			   this->toolTip1->AutoPopDelay = 10000;
			   this->toolTip1->BackColor = System::Drawing::SystemColors::ActiveCaption;
			   this->toolTip1->InitialDelay = 400;
			   this->toolTip1->ReshowDelay = 80;
			   this->toolTip1->ToolTipIcon = System::Windows::Forms::ToolTipIcon::Info;
			   this->toolTip1->ToolTipTitle = L"Info";
			   this->toolTip1->UseAnimation = false;
			   this->toolTip1->UseFading = false;
			   // 
			   // lbl_ethaddress
			   // 
			   this->lbl_ethaddress->AutoSize = true;
			   this->lbl_ethaddress->Location = System::Drawing::Point(17, 69);
			   this->lbl_ethaddress->Name = L"lbl_ethaddress";
			   this->lbl_ethaddress->Size = System::Drawing::Size(209, 13);
			   this->lbl_ethaddress->TabIndex = 2;
			   this->lbl_ethaddress->Text = L"Ethereum Address  ( for Pool mining mode )";
			   // 
			   // slider_netinterval
			   // 
			   this->slider_netinterval->BackColor = System::Drawing::SystemColors::ButtonFace;
			   this->slider_netinterval->Cursor = System::Windows::Forms::Cursors::Hand;
			   this->slider_netinterval->Dock = System::Windows::Forms::DockStyle::Fill;
			   this->slider_netinterval->LargeChange = 25;
			   this->slider_netinterval->Location = System::Drawing::Point(122, 3);
			   this->slider_netinterval->Maximum = 2000;
			   this->slider_netinterval->Minimum = 300;
			   this->slider_netinterval->Name = L"slider_netinterval";
			   this->slider_netinterval->RightToLeft = System::Windows::Forms::RightToLeft::No;
			   this->slider_netinterval->Size = System::Drawing::Size(152, 53);
			   this->slider_netinterval->TabIndex = 0;
			   this->toolTip1->SetToolTip(this->slider_netinterval, L"How often to access the Pool/Network, in Milliseconds (ms).");
			   this->slider_netinterval->Value = 575;
			   this->slider_netinterval->ValueChanged += gcnew System::EventHandler(this, &OptionsForm::Slider_netinterval_ValueChanged);
			   // 
			   // lbl_optionpoolurl
			   // 
			   this->lbl_optionpoolurl->AutoSize = true;
			   this->lbl_optionpoolurl->Location = System::Drawing::Point(17, 24);
			   this->lbl_optionpoolurl->Name = L"lbl_optionpoolurl";
			   this->lbl_optionpoolurl->Size = System::Drawing::Size(249, 13);
			   this->lbl_optionpoolurl->TabIndex = 9;
			   this->lbl_optionpoolurl->Text = L"Pool Address/Port (ex. http://mike.rs:8586 or 8080)";
			   this->toolTip1->SetToolTip(this->lbl_optionpoolurl, L"Pool that you would like to mine to and the Port. Usually 8586. 8080 for Pools th"
				   L"at use CloudFlare.\r\nVisit your Pool\'s address, without the Port, in a Web Browse"
				   L"r to view your Shares/Payout info.");
			   // 
			   // lbl_donation
			   // 
			   this->lbl_donation->AutoSize = true;
			   this->lbl_donation->Location = System::Drawing::Point(17, 24);
			   this->lbl_donation->Name = L"lbl_donation";
			   this->lbl_donation->Size = System::Drawing::Size(207, 13);
			   this->lbl_donation->TabIndex = 1;
			   this->lbl_donation->Text = L"Auto-Donation a.k.a. Devfee (in Shares %)";
			   // 
			   // nud_donatepct
			   // 
			   this->nud_donatepct->DecimalPlaces = 1;
			   this->nud_donatepct->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 5, 0, 0, 65536 });
			   this->nud_donatepct->Location = System::Drawing::Point(270, 22);
			   this->nud_donatepct->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 10, 0, 0, 0 });
			   this->nud_donatepct->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 15, 0, 0, 65536 });
			   this->nud_donatepct->Name = L"nud_donatepct";
			   this->nud_donatepct->Size = System::Drawing::Size(73, 20);
			   this->nud_donatepct->TabIndex = 0;
			   this->nud_donatepct->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 15, 0, 0, 65536 });
			   // 
			   // label3
			   // 
			   this->label3->AutoSize = true;
			   this->label3->Location = System::Drawing::Point(213, 24);
			   this->label3->Name = L"label3";
			   this->label3->Size = System::Drawing::Size(53, 13);
			   this->label3->TabIndex = 3;
			   this->label3->Text = L"Verbosity:";
			   this->toolTip1->SetToolTip(this->label3, L"How much information to display.");
			   // 
			   // ckbox_autostart
			   // 
			   this->ckbox_autostart->AutoSize = true;
			   this->ckbox_autostart->Location = System::Drawing::Point(20, 23);
			   this->ckbox_autostart->Name = L"ckbox_autostart";
			   this->ckbox_autostart->Size = System::Drawing::Size(161, 17);
			   this->ckbox_autostart->TabIndex = 0;
			   this->ckbox_autostart->Text = L"Auto-Start Mining on Launch";
			   this->toolTip1->SetToolTip(this->ckbox_autostart, L"Intended for use when starting multiple copies\r\nof COSMiC from a batch file. Note"
				   L" that you will\r\nnot be able to change Configuration until after\r\nclicking \'Stop "
				   L"Mining\' when using this option.");
			   this->ckbox_autostart->UseVisualStyleBackColor = true;
			   // 
			   // combobox_verbosity
			   // 
			   this->combobox_verbosity->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			   this->combobox_verbosity->FormattingEnabled = true;
			   this->combobox_verbosity->Items->AddRange(gcnew cli::array< System::Object^  >(3) { L"Less", L"Normal", L"Verbose" });
			   this->combobox_verbosity->Location = System::Drawing::Point(272, 19);
			   this->combobox_verbosity->MaxDropDownItems = 4;
			   this->combobox_verbosity->Name = L"combobox_verbosity";
			   this->combobox_verbosity->Size = System::Drawing::Size(104, 21);
			   this->combobox_verbosity->TabIndex = 1;
			   this->toolTip1->SetToolTip(this->combobox_verbosity, L"How much information to display.");
			   // 
			   // label10
			   // 
			   this->label10->AutoSize = true;
			   this->label10->Dock = System::Windows::Forms::DockStyle::Fill;
			   this->label10->ImageAlign = System::Drawing::ContentAlignment::MiddleLeft;
			   this->label10->Location = System::Drawing::Point(3, 59);
			   this->label10->Name = L"label10";
			   this->label10->Size = System::Drawing::Size(113, 59);
			   this->label10->TabIndex = 6;
			   this->label10->Text = L"Difficulty Update (every x intervals)";
			   this->label10->TextAlign = System::Drawing::ContentAlignment::MiddleLeft;
			   // 
			   // slider_diffupdate
			   // 
			   this->slider_diffupdate->Dock = System::Windows::Forms::DockStyle::Fill;
			   this->slider_diffupdate->Location = System::Drawing::Point(122, 62);
			   this->slider_diffupdate->Maximum = 1500;
			   this->slider_diffupdate->Minimum = 1;
			   this->slider_diffupdate->Name = L"slider_diffupdate";
			   this->slider_diffupdate->RightToLeft = System::Windows::Forms::RightToLeft::No;
			   this->slider_diffupdate->Size = System::Drawing::Size(152, 53);
			   this->slider_diffupdate->TabIndex = 1;
			   this->slider_diffupdate->Value = 50;
			   this->slider_diffupdate->ValueChanged += gcnew System::EventHandler(this, &OptionsForm::Slider_diffupdate_ValueChanged);
			   // 
			   // labeldiffupdate
			   // 
			   this->labeldiffupdate->AutoSize = true;
			   this->labeldiffupdate->Dock = System::Windows::Forms::DockStyle::Fill;
			   this->labeldiffupdate->ImageAlign = System::Drawing::ContentAlignment::MiddleLeft;
			   this->labeldiffupdate->Location = System::Drawing::Point(3, 0);
			   this->labeldiffupdate->Name = L"labeldiffupdate";
			   this->labeldiffupdate->Size = System::Drawing::Size(113, 59);
			   this->labeldiffupdate->TabIndex = 4;
			   this->labeldiffupdate->Text = L"Pool Access Interval";
			   this->labeldiffupdate->TextAlign = System::Drawing::ContentAlignment::MiddleLeft;
			   this->toolTip1->SetToolTip(this->labeldiffupdate, L"How often to access the Pool/Network, in Milliseconds (ms).");
			   // 
			   // lbl_netinterval_indicator
			   // 
			   this->lbl_netinterval_indicator->AutoEllipsis = true;
			   this->lbl_netinterval_indicator->BorderStyle = System::Windows::Forms::BorderStyle::Fixed3D;
			   this->lbl_netinterval_indicator->Dock = System::Windows::Forms::DockStyle::Fill;
			   this->lbl_netinterval_indicator->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			   this->lbl_netinterval_indicator->Location = System::Drawing::Point(280, 0);
			   this->lbl_netinterval_indicator->Name = L"lbl_netinterval_indicator";
			   this->lbl_netinterval_indicator->Padding = System::Windows::Forms::Padding(2);
			   this->lbl_netinterval_indicator->Size = System::Drawing::Size(76, 59);
			   this->lbl_netinterval_indicator->TabIndex = 19;
			   this->lbl_netinterval_indicator->Text = L"-1";
			   this->lbl_netinterval_indicator->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			   this->toolTip1->SetToolTip(this->lbl_netinterval_indicator, L"How often to get updated Mining Parameters from the Pool/Network, in Milliseconds"
				   L" (ms).");
			   this->lbl_netinterval_indicator->UseCompatibleTextRendering = true;
			   // 
			   // label1
			   // 
			   this->label1->AutoSize = true;
			   this->label1->Location = System::Drawing::Point(3, 118);
			   this->label1->Name = L"label1";
			   this->label1->Size = System::Drawing::Size(112, 26);
			   this->label1->TabIndex = 26;
			   this->label1->Text = L"Node Access Interval (Solo Mode)";
			   // 
			   // lbl_netinterval_solo_indicator
			   // 
			   this->lbl_netinterval_solo_indicator->AutoEllipsis = true;
			   this->lbl_netinterval_solo_indicator->BorderStyle = System::Windows::Forms::BorderStyle::Fixed3D;
			   this->lbl_netinterval_solo_indicator->Dock = System::Windows::Forms::DockStyle::Fill;
			   this->lbl_netinterval_solo_indicator->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			   this->lbl_netinterval_solo_indicator->Location = System::Drawing::Point(280, 59);
			   this->lbl_netinterval_solo_indicator->Name = L"lbl_netinterval_solo_indicator";
			   this->lbl_netinterval_solo_indicator->Padding = System::Windows::Forms::Padding(2);
			   this->lbl_netinterval_solo_indicator->Size = System::Drawing::Size(76, 59);
			   this->lbl_netinterval_solo_indicator->TabIndex = 29;
			   this->lbl_netinterval_solo_indicator->Text = L"-1";
			   this->lbl_netinterval_solo_indicator->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			   this->lbl_netinterval_solo_indicator->UseCompatibleTextRendering = true;
			   // 
			   // lbl_diffupdate_indicator
			   // 
			   this->lbl_diffupdate_indicator->AutoEllipsis = true;
			   this->lbl_diffupdate_indicator->BorderStyle = System::Windows::Forms::BorderStyle::Fixed3D;
			   this->lbl_diffupdate_indicator->Dock = System::Windows::Forms::DockStyle::Fill;
			   this->lbl_diffupdate_indicator->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			   this->lbl_diffupdate_indicator->Location = System::Drawing::Point(280, 118);
			   this->lbl_diffupdate_indicator->Name = L"lbl_diffupdate_indicator";
			   this->lbl_diffupdate_indicator->Padding = System::Windows::Forms::Padding(2);
			   this->lbl_diffupdate_indicator->Size = System::Drawing::Size(76, 59);
			   this->lbl_diffupdate_indicator->TabIndex = 30;
			   this->lbl_diffupdate_indicator->Text = L"-1";
			   this->lbl_diffupdate_indicator->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			   this->lbl_diffupdate_indicator->UseCompatibleTextRendering = true;
			   // 
			   // groupBox3
			   // 
			   this->groupBox3->Controls->Add(this->nud_donatepct);
			   this->groupBox3->Controls->Add(this->pictureBox1);
			   this->groupBox3->Controls->Add(this->lbl_donation);
			   this->groupBox3->Location = System::Drawing::Point(12, 342);
			   this->groupBox3->Name = L"groupBox3";
			   this->groupBox3->Size = System::Drawing::Size(392, 53);
			   this->groupBox3->TabIndex = 3;
			   this->groupBox3->TabStop = false;
			   this->groupBox3->Text = L"Support Development";
			   // 
			   // pictureBox1
			   // 
			   this->pictureBox1->BackgroundImageLayout = System::Windows::Forms::ImageLayout::Stretch;
			   this->pictureBox1->Location = System::Drawing::Point(350, 22);
			   this->pictureBox1->Name = L"pictureBox1";
			   this->pictureBox1->Size = System::Drawing::Size(24, 20);
			   this->pictureBox1->TabIndex = 12;
			   this->pictureBox1->TabStop = false;
			   this->pictureBox1->Click += gcnew System::EventHandler(this, &OptionsForm::pictureBox1_Click_1);
			   // 
			   // groupBox4
			   // 
			   this->groupBox4->Controls->Add(this->label3);
			   this->groupBox4->Controls->Add(this->ckbox_autostart);
			   this->groupBox4->Controls->Add(this->combobox_verbosity);
			   this->groupBox4->Location = System::Drawing::Point(12, 404);
			   this->groupBox4->Name = L"groupBox4";
			   this->groupBox4->Size = System::Drawing::Size(392, 55);
			   this->groupBox4->TabIndex = 4;
			   this->groupBox4->TabStop = false;
			   this->groupBox4->Text = L"Miscellaneous";
			   // 
			   // groupBox1
			   // 
			   this->groupBox1->Controls->Add(this->etext_mineraddress);
			   this->groupBox1->Controls->Add(this->etext_poolurl);
			   this->groupBox1->Controls->Add(this->lbl_ethaddress);
			   this->groupBox1->Controls->Add(this->lbl_optionpoolurl);
			   this->groupBox1->Location = System::Drawing::Point(12, 12);
			   this->groupBox1->Name = L"groupBox1";
			   this->groupBox1->Size = System::Drawing::Size(392, 119);
			   this->groupBox1->TabIndex = 0;
			   this->groupBox1->TabStop = false;
			   this->groupBox1->Text = L"Pool and Payment Settings";
			   // 
			   // tableLayoutPanel2
			   // 
			   this->tableLayoutPanel2->ColumnCount = 3;
			   this->tableLayoutPanel2->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
				   33.24324F)));
			   this->tableLayoutPanel2->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
				   44.05405F)));
			   this->tableLayoutPanel2->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
				   22.70271F)));
			   this->tableLayoutPanel2->Controls->Add(this->lbl_diffupdate_indicator, 2, 2);
			   this->tableLayoutPanel2->Controls->Add(this->lbl_netinterval_solo_indicator, 2, 1);
			   this->tableLayoutPanel2->Controls->Add(this->labeldiffupdate, 0, 0);
			   this->tableLayoutPanel2->Controls->Add(this->slider_diffupdate, 1, 1);
			   this->tableLayoutPanel2->Controls->Add(this->lbl_netinterval_indicator, 2, 0);
			   this->tableLayoutPanel2->Controls->Add(this->slider_netinterval, 1, 0);
			   this->tableLayoutPanel2->Controls->Add(this->label10, 0, 1);
			   this->tableLayoutPanel2->Controls->Add(this->label1, 0, 2);
			   this->tableLayoutPanel2->Controls->Add(this->slider_netinterval_solo, 1, 2);
			   this->tableLayoutPanel2->GrowStyle = System::Windows::Forms::TableLayoutPanelGrowStyle::FixedSize;
			   this->tableLayoutPanel2->Location = System::Drawing::Point(17, 19);
			   this->tableLayoutPanel2->Name = L"tableLayoutPanel2";
			   this->tableLayoutPanel2->RowCount = 5;
			   this->tableLayoutPanel2->RowStyles->Add((gcnew System::Windows::Forms::RowStyle()));
			   this->tableLayoutPanel2->RowStyles->Add((gcnew System::Windows::Forms::RowStyle()));
			   this->tableLayoutPanel2->RowStyles->Add((gcnew System::Windows::Forms::RowStyle()));
			   this->tableLayoutPanel2->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute, 25)));
			   this->tableLayoutPanel2->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute, 25)));
			   this->tableLayoutPanel2->Size = System::Drawing::Size(359, 171);
			   this->tableLayoutPanel2->TabIndex = 26;
			   // 
			   // slider_netinterval_solo
			   // 
			   this->slider_netinterval_solo->Location = System::Drawing::Point(122, 121);
			   this->slider_netinterval_solo->Maximum = 2000;
			   this->slider_netinterval_solo->Minimum = 100;
			   this->slider_netinterval_solo->Name = L"slider_netinterval_solo";
			   this->slider_netinterval_solo->Size = System::Drawing::Size(152, 53);
			   this->slider_netinterval_solo->TabIndex = 2;
			   this->slider_netinterval_solo->Value = 399;
			   this->slider_netinterval_solo->ValueChanged += gcnew System::EventHandler(this, &OptionsForm::slider_netinterval_solo_ValueChanged);
			   // 
			   // groupBox2
			   // 
			   this->groupBox2->Controls->Add(this->tableLayoutPanel2);
			   this->groupBox2->Location = System::Drawing::Point(12, 140);
			   this->groupBox2->Name = L"groupBox2";
			   this->groupBox2->Size = System::Drawing::Size(392, 196);
			   this->groupBox2->TabIndex = 1;
			   this->groupBox2->TabStop = false;
			   this->groupBox2->Text = L"Network Timings";
			   // 
			   // button1
			   // 
			   this->button1->Enabled = false;
			   this->button1->Location = System::Drawing::Point(12, 474);
			   this->button1->Name = L"button1";
			   this->button1->Size = System::Drawing::Size(75, 20);
			   this->button1->TabIndex = 7;
			   this->button1->Text = L"defaults";
			   this->button1->UseVisualStyleBackColor = true;
			   this->button1->Visible = false;
			   // 
			   // OptionsForm
			   // 
			   this->AcceptButton = this->but_savesettings;
			   this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			   this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			   this->CancelButton = this->but_discard;
			   this->ClientSize = System::Drawing::Size(414, 507);
			   this->Controls->Add(this->button1);
			   this->Controls->Add(this->but_savesettings);
			   this->Controls->Add(this->but_discard);
			   this->Controls->Add(this->groupBox2);
			   this->Controls->Add(this->groupBox4);
			   this->Controls->Add(this->groupBox1);
			   this->Controls->Add(this->groupBox3);
			   this->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				   static_cast<System::Byte>(0)));
			   this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::FixedDialog;
			   this->MaximizeBox = false;
			   this->MinimizeBox = false;
			   this->Name = L"OptionsForm";
			   this->ShowIcon = false;
			   this->ShowInTaskbar = false;
			   this->SizeGripStyle = System::Windows::Forms::SizeGripStyle::Hide;
			   this->StartPosition = System::Windows::Forms::FormStartPosition::CenterScreen;
			   this->Text = L"COSMiC - General Options";
			   this->Load += gcnew System::EventHandler(this, &OptionsForm::OptionsForm_Load);
			   (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->slider_netinterval))->EndInit();
			   (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->nud_donatepct))->EndInit();
			   (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->slider_diffupdate))->EndInit();
			   this->groupBox3->ResumeLayout(false);
			   this->groupBox3->PerformLayout();
			   (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->EndInit();
			   this->groupBox4->ResumeLayout(false);
			   this->groupBox4->PerformLayout();
			   this->groupBox1->ResumeLayout(false);
			   this->groupBox1->PerformLayout();
			   this->tableLayoutPanel2->ResumeLayout(false);
			   this->tableLayoutPanel2->PerformLayout();
			   (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->slider_netinterval_solo))->EndInit();
			   this->groupBox2->ResumeLayout(false);
			   this->ResumeLayout(false);

		   }
#pragma endregion
	private: System::Void OptionsForm_Load(System::Object^ sender, System::EventArgs^ e)
	{ // When the options form loads ..
		UInt32 scratch_UInt32 = DEFAULT_NETINTERVAL_SOLO;   // separate network access timing for solo mode
															// (to mind # of node requests sent)
		if (gVerbosity == V_DEBUG)  combobox_verbosity->Items->Add("Verybose");  // new dropdown item if debug
		if (gCudaSolving) { /* can't change these while mining: */
			groupBox1->Enabled = false;
			this->toolTip1->SetToolTip(groupBox1, "Please stop mining to change these settings.");
		}
		// get the pool URL from the .config file, put in textbox
		String^ poolurl = ConfigurationManager::AppSettings["PoolURL"];
		etext_poolurl->Text = poolurl;

		// get the miner address from the .config file, put in textbox
		String^ mineraddr = ConfigurationManager::AppSettings["MinerEthAddress"];
		etext_mineraddress->Text = mineraddr;

		// Possible TODO: reject/handle (all) invalid settings from the config file

		// get network access interval from the .config file, put value in slider + text in label adjacent
		slider_netinterval->Value = Convert::ToInt32(ConfigurationManager::AppSettings["NetworkInterval"]);
		lbl_netinterval_indicator->Text = System::Convert::ToString(slider_netinterval->Value) + " ms";

		// get difficulty update frequency from the .config file, put value in slider + text in label adjacent
		slider_diffupdate->Value = Convert::ToInt32(ConfigurationManager::AppSettings["DiffUpdateFreq"]);
		lbl_diffupdate_indicator->Text = System::Convert::ToString(slider_diffupdate->Value);

		// solo-mode network access interval value from Configuration:
		slider_netinterval_solo->Value = DEFAULT_NETINTERVAL_SOLO; // redundant
		if (ConfigurationManager::AppSettings["soloNetInterval"])
		{
			if (UInt32::TryParse(ConfigurationManager::AppSettings["soloNetInterval"], scratch_UInt32))
				slider_netinterval_solo->Value = (int)scratch_UInt32;  // OK- set slider's value to Config-specified value
			else
			{ // parse err:
				printf("error parsing Solo Network Interval setting from Config, using default: %d ms. \n", DEFAULT_NETINTERVAL_SOLO);
				slider_netinterval_solo->Value = DEFAULT_NETINTERVAL_SOLO; // <-- redundant
			}
		}
		else {  // if the key "soloNetInterval" is not found in the Configuration:
			slider_netinterval_solo->Value = DEFAULT_NETINTERVAL_SOLO;  // set to default
			if (gVerbosity > V_NORM)  printf("key not found (solo net interval)- setting to default: %d ms \n", DEFAULT_NETINTERVAL_SOLO);
		}
		// TODO: if key not found, write default to Configuration here?
		lbl_netinterval_solo_indicator->Text = slider_netinterval_solo->Value.ToString() + " ms";  // update field next to slider

		// get verbosity setting from internal, not from .config, consider making this the standard behavior to avoid spinning
		// up a (potential) HDD to read Configuration? (test this)
		combobox_verbosity->SelectedIndex = (int)gVerbosity;

		// get network access interval from the .config file, put value in slider + text in label adjacent
		double scratch_donatePct = 0;

		String^ scratch_donatePctString = ConfigurationManager::AppSettings["AutoDonatePercent"];
		// if Double::TryParse call was successful, process the donation % from the config file (US English Windows)
		if (Double::TryParse(scratch_donatePctString, scratch_donatePct))
		{
			if (scratch_donatePct < 1.5 || scratch_donatePct > 10)
				scratch_donatePct = 1.5;	// reject invalid settings, reset to 1 (default)
			nud_donatepct->Value = (System::Decimal)scratch_donatePct;
		}
		else      // Double::TryParse conversion failed probably due to region (using comma for decimal point character)
		{
			scratch_donatePctString = scratch_donatePctString->Replace(".", ",");		// replace . with , for Windows configurations with , for decimal
			if (Double::TryParse(scratch_donatePctString, scratch_donatePct))
			{
				// conversion successful
				if (scratch_donatePct < 1.5 || scratch_donatePct > 10)  // reject invalid settings, reset
					scratch_donatePct = 1.5;
				nud_donatepct->Value = (System::Decimal)scratch_donatePct;
			}
			else
			{
				// conversion still unsuccessful
				MessageBox::Show("Invalid Donation Percent setting in Config file. Please check this file in a text editor. If your OS version uses commas for decimal points, change it to the '.' character in Control Panel or change AutoDonatePct in COSMiC's .Config file to use a comma instead.");
				scratch_donatePct = 1.5;
				nud_donatepct->Value = (System::Decimal)scratch_donatePct;
			}
			// the text was not a number, show user an error if appropiate
		}

		// if auto-start is enabled in the Configuration
		if (ConfigurationManager::AppSettings["AutoStart"] == "true")
			ckbox_autostart->Checked = true;
		else
			ckbox_autostart->Checked = false;

		// AUTOSTART (FIXME): work in progress
		//
	}

private: System::Void button1_Click(System::Object^  sender, System::EventArgs^  e)
{
	// Discard button was clicked: hide the form, don't change anything
	this->Hide();
}
private: System::Void button2_Click(System::Object^  sender, System::EventArgs^  e)
{ // [WIP / FIXME]
	// Save Settings button was pressed

	// TODO: More checks for bad input. Consider checksumming ethereum address :)
	if (etext_mineraddress->Text->Length != 42)  /* [redundant?]  Check the max input length of the field! Set appropriately. Same for Pool URL, Node addr, etc. [WIP/TODO] <-- */
	{
		/* [TODO]: error dialog, user needs to fix or click `Discard Changes` */
		etext_mineraddress->Focus();
		MessageBox::Show("Invalid Ethereum address. Expecting '0x' followed by 40 hex digits. \n"
			"Please correct the address or click 'Discard Changes'.",
			"COSMiC", MessageBoxButtons::OK, MessageBoxIcon::Error);
		this->DialogResult = System::Windows::Forms::DialogResult::None;  // don't close form
		return;  // don't close form
	}
	/*   ^ [wip] ^  */
	
	// [todo]: support 40-character input without `0x` specifier and silently add it?  already doing this in ConfigureSoloMining form.
	const std::string new_mining_address = msclr::interop::marshal_as<std::string>(etext_mineraddress->Text);
	if (!checkString(new_mining_address, 42, true, true))
	{
		/* [TODO]: error dialog, user needs to fix or click `Discard Changes` */
		etext_mineraddress->Focus();
		MessageBox::Show("Invalid Ethereum address. Expecting '0x' followed by 40 hex digits. \n"
			"Please correct the address or click 'Discard Changes'.",
			"COSMiC", MessageBoxButtons::OK, MessageBoxIcon::Error);
		this->DialogResult = System::Windows::Forms::DialogResult::None;  // don't close form
		return;  // don't close form
	}
	// new ^

//	=== UPDATE THE CONFIGURATION: ===
// - get handle to the configuration file
	System::Configuration::Configuration ^theConfig = ConfigurationManager::OpenExeConfiguration(ConfigurationUserLevel::None);
	String^ mstr_scratch;

// - save the pool url and miner's ethereum address
	theConfig->AppSettings->Settings->Remove("PoolURL");
	theConfig->AppSettings->Settings->Add("PoolURL", etext_poolurl->Text);
	theConfig->AppSettings->Settings->Remove("MinerEthAddress");
	theConfig->AppSettings->Settings->Add("MinerEthAddress", etext_mineraddress->Text);

// - save the rate at which mining params are retrieved
	theConfig->AppSettings->Settings->Remove("NetworkInterval");
	mstr_scratch = slider_netinterval->Value.ToString();
	theConfig->AppSettings->Settings->Add("NetworkInterval", mstr_scratch);

// - save the diffupdate frequency
	theConfig->AppSettings->Settings->Remove("DiffUpdateFreq");
	mstr_scratch = slider_diffupdate->Value.ToString();
	theConfig->AppSettings->Settings->Add("DiffUpdateFreq", mstr_scratch);

// - save the auto-donation %
	theConfig->AppSettings->Settings->Remove("AutoDonatePercent");
	mstr_scratch = nud_donatepct->Value.ToString();
	theConfig->AppSettings->Settings->Add("AutoDonatePercent", mstr_scratch);

// - save the selected device (old, single-gpu stuff)
//	theConfig->AppSettings->Settings->Remove("CudaDeviceID");
//	scratchString = nud_devicenum->Value.ToString();
//	theConfig->AppSettings->Settings->Add("CudaDeviceID", scratchString);

// - save auto-start setting
	theConfig->AppSettings->Settings->Remove("AutoStart");
	ckbox_autostart->Checked ? mstr_scratch = "true" : "false";  // fix checkbox's name <-
	theConfig->AppSettings->Settings->Add("AutoStart", mstr_scratch);

// - save that the miner has been configured from its default 'shipping' state
	theConfig->AppSettings->Settings->Remove("DialogConfigured");
	theConfig->AppSettings->Settings->Add("DialogConfigured", "true");

	const unsigned short new_verbosity = (unsigned int)combobox_verbosity->SelectedIndex;
	if (new_verbosity <= 3) // if new intensity does not exceed valid bounds...
	{
		gVerbosity = (unsigned short)combobox_verbosity->SelectedIndex;
		if (gVerbosity > V_NORM)  // hehe
			printf("\nSetting Verbosity Level to %d. Saving to configuration... \n", gVerbosity);
		theConfig->AppSettings->Settings->Remove("Verbosity");
		theConfig->AppSettings->Settings->Add("Verbosity", Convert::ToString(gVerbosity));
	}
	else
	{ // exceeded bounds of valid VERBOSITY_ settings
		printf("Error: Invalid Verbosity setting %d from combobox_verbosity in OptionsForm. Defaulting to Normal.\n", new_verbosity);
		MessageBox::Show("Invalid Verbosity setting " + Convert::ToString(new_verbosity) + " from combobox_verbosity in OptionsForm. " +
			"Defaulting to Normal Verbosity. \nPlease report this to the developer.", "COSMiC Miner - Error", MessageBoxButtons::OK, MessageBoxIcon::Warning);
		gVerbosity = V_NORM;
	}

	// new (20191204): solo network access interval:
	if ( ConfigurationManager::AppSettings["soloNetInterval"] )  // only remove if present
		theConfig->AppSettings->Settings->Remove("soloNetInterval");
	theConfig->AppSettings->Settings->Add("soloNetInterval", slider_netinterval_solo->Value.ToString() );

	// save all the settings from the dialog to the configuration
	try {
		theConfig->Save(ConfigurationSaveMode::Modified);
		ConfigurationManager::RefreshSection("appSettings");  }
	catch (...)
	{ // ... (todo: remove?)
		MessageBox::Show("Exception caught while saving Configuration.", "COSMiC - Exception");
		domesg_verb("Exception caught while saving Config. Is the disk full?", true, V_NORM);
	}

	domesg_verb("Updated configuration file.", true, V_MORE);
	gStr_PoolHTTPAddress = msclr::interop::marshal_as<std::string>(etext_poolurl->Text);  // [WIP]: update pool URL field's contents (OptionsForm)

	LOG_IF_F(INFO, gVerbosity >= V_MORE, "Updating internal parameters.");
	gNetInterval = slider_netinterval->Value;
	gAutoDonationPercent = (double)nud_donatepct->Value;
	gDiffUpdateFrequency = (int)slider_diffupdate->Value;
//	domesg_verb("Updated settings.", true, V_MORE);



	if (new_mining_address != gStr_MinerEthAddress)
	{ // otherwise (not mining, pool mode):
		//	if(!gSoloMiningMode && !gCudaSolving) { ... }  // pool mode and not mining
		// [note]:	pool url and mining address fields are disabled/grayed out while mining, so they can't be changed

		//	gMiningParameters.mineraddress_str = msclr::interop::marshal_as<std::string>(etext_mineraddress->Text);  // elsewhere.
		gStr_MinerEthAddress = new_mining_address;
		LOG_IF_F(INFO, gVerbosity >= V_MORE, "Set pool-mode mining address to %s.", gStr_MinerEthAddress.c_str());
	}
}

private: System::Void pictureBox1_Click_1(System::Object^  sender, System::EventArgs^  e)
{ // clicked the heart
	Console::WriteLine("Thanks for using COSMiC!");
}

private: System::Void Slider_netinterval_ValueChanged(System::Object^ sender, System::EventArgs^ e)
{ // on net interval slider's value changed
	lbl_netinterval_indicator->Text = System::Convert::ToString(slider_netinterval->Value) + " ms";
}

private: System::Void Slider_diffupdate_ValueChanged(System::Object^ sender, System::EventArgs^ e)
{ // on difficulty interval slider's value changed
	lbl_diffupdate_indicator->Text = Convert::ToString(slider_diffupdate->Value)/* + " ms"*/;
}

private: System::Void slider_netinterval_solo_ValueChanged(System::Object^ sender, System::EventArgs^ e)
{ // solo mode network interval slider value changed
	lbl_netinterval_solo_indicator->Text = slider_netinterval_solo->Value.ToString() + " ms";
}

private: System::Void pictureBox2_Click(System::Object^ sender, System::EventArgs^ e)
{
	etext_poolurl->Text = "localhost:8080";
}

};	//class OptionsForm

} //namespace

