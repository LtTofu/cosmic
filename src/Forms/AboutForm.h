#pragma once
// AboutForm.h : About COSMiC... and Thanks
// 2020 LtTofu


#include "defs.hpp"  // #include "defs.h"

namespace Cosmic {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
//	//using namespace System::Data;
	using namespace System::Drawing;
	using namespace System::Numerics;	//<-- for BigInteger

	/// <summary>
	/// Summary for AboutForm
	/// </summary>
	public ref class AboutForm : public System::Windows::Forms::Form
	{
	public:
		AboutForm( System::Numerics::BigInteger total_hashes )
		{ // write the # of total hashes computed by the miner since install [TODO].
			InitializeComponent();
			lbl_totalhashes->Text = (total_hashes < BigInteger::One) ? "--" : total_hashes.ToString();	// [WIP] / [TODO].
			// constructor code here
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~AboutForm()
		{
			if (components)
			{
				delete components;
			}
		}

	private: System::Windows::Forms::Label^  label1;
	private: System::Windows::Forms::TextBox^  textBox1;
	private: System::Windows::Forms::Label^  label2;
	private: System::Windows::Forms::Label^  label3;
	private: System::Windows::Forms::Label^  label4;
	private: System::Windows::Forms::Label^  label5;
	private: System::Windows::Forms::Button^  button1;
	private: System::Windows::Forms::Label^  label6;
	private: System::Windows::Forms::TextBox^  textBox2;

	private: System::Windows::Forms::Label^ lbl_builddate;
	private: System::Windows::Forms::Label^ lbl_totalhashes;



	protected:

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
			System::ComponentModel::ComponentResourceManager^ resources = (gcnew System::ComponentModel::ComponentResourceManager(AboutForm::typeid));
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->textBox1 = (gcnew System::Windows::Forms::TextBox());
			this->label2 = (gcnew System::Windows::Forms::Label());
			this->label3 = (gcnew System::Windows::Forms::Label());
			this->label4 = (gcnew System::Windows::Forms::Label());
			this->label5 = (gcnew System::Windows::Forms::Label());
			this->button1 = (gcnew System::Windows::Forms::Button());
			this->label6 = (gcnew System::Windows::Forms::Label());
			this->textBox2 = (gcnew System::Windows::Forms::TextBox());
			this->lbl_builddate = (gcnew System::Windows::Forms::Label());
			this->lbl_totalhashes = (gcnew System::Windows::Forms::Label());
			this->SuspendLayout();
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Font = (gcnew System::Drawing::Font(L"Segoe UI", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->label1->Location = System::Drawing::Point(17, 30);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(108, 17);
			this->label1->TabIndex = 22;
			this->label1->Text = L"2018-2020 LtTofu";
			// 
			// textBox1
			// 
			this->textBox1->BackColor = System::Drawing::SystemColors::ControlLightLight;
			this->textBox1->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->textBox1->Cursor = System::Windows::Forms::Cursors::Arrow;
			this->textBox1->Location = System::Drawing::Point(20, 60);
			this->textBox1->MaxLength = 16384;
			this->textBox1->Multiline = true;
			this->textBox1->Name = L"textBox1";
			this->textBox1->ReadOnly = true;
			this->textBox1->ScrollBars = System::Windows::Forms::ScrollBars::Vertical;
			this->textBox1->Size = System::Drawing::Size(731, 221);
			this->textBox1->TabIndex = 23;
			this->textBox1->Text = resources->GetString(L"textBox1.Text");
			this->textBox1->TextChanged += gcnew System::EventHandler(this, &AboutForm::textBox1_TextChanged);
			// 
			// label2
			// 
			this->label2->AutoSize = true;
			this->label2->Font = (gcnew System::Drawing::Font(L"Segoe UI Semibold", 12, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->label2->Location = System::Drawing::Point(16, 9);
			this->label2->Name = L"label2";
			this->label2->Size = System::Drawing::Size(197, 21);
			this->label2->TabIndex = 24;
			this->label2->Text = L"COSMiC V4.1.5 (preview8)";
			// 
			// label3
			// 
			this->label3->AutoSize = true;
			this->label3->Font = (gcnew System::Drawing::Font(L"Segoe UI", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->label3->Location = System::Drawing::Point(237, 12);
			this->label3->Name = L"label3";
			this->label3->Size = System::Drawing::Size(318, 17);
			this->label3->TabIndex = 25;
			this->label3->Text = L"Development Snapshot - See README for more info!";
			// 
			// label4
			// 
			this->label4->AutoSize = true;
			this->label4->Font = (gcnew System::Drawing::Font(L"Segoe UI Semibold", 9, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->label4->Location = System::Drawing::Point(278, 298);
			this->label4->Name = L"label4";
			this->label4->Size = System::Drawing::Size(245, 15);
			this->label4->TabIndex = 26;
			this->label4->Text = L"The author would like to thank the following:";
			// 
			// label5
			// 
			this->label5->AutoSize = true;
			this->label5->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->label5->Location = System::Drawing::Point(105, 384);
			this->label5->Name = L"label5";
			this->label5->Size = System::Drawing::Size(0, 16);
			this->label5->TabIndex = 27;
			// 
			// button1
			// 
			this->button1->DialogResult = System::Windows::Forms::DialogResult::OK;
			this->button1->Location = System::Drawing::Point(655, 380);
			this->button1->Name = L"button1";
			this->button1->Size = System::Drawing::Size(95, 27);
			this->button1->TabIndex = 1;
			this->button1->Text = L"OK";
			this->button1->UseVisualStyleBackColor = true;
			// 
			// label6
			// 
			this->label6->AutoSize = true;
			this->label6->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->label6->Location = System::Drawing::Point(188, 341);
			this->label6->Name = L"label6";
			this->label6->Size = System::Drawing::Size(416, 15);
			this->label6->TabIndex = 29;
			this->label6->Text = L"Contact: LtTofu#9565 on the 0xBitcoin Discord - https://discord.gg/EyuFrB6p";
			// 
			// textBox2
			// 
			this->textBox2->BackColor = System::Drawing::SystemColors::ControlLightLight;
			this->textBox2->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->textBox2->Cursor = System::Windows::Forms::Cursors::Arrow;
			this->textBox2->Font = (gcnew System::Drawing::Font(L"Segoe UI", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->textBox2->Location = System::Drawing::Point(19, 316);
			this->textBox2->MaxLength = 1000;
			this->textBox2->Name = L"textBox2";
			this->textBox2->ReadOnly = true;
			this->textBox2->Size = System::Drawing::Size(731, 22);
			this->textBox2->TabIndex = 30;
			this->textBox2->Text = L"Infernal Toast, Mikers, Zegordo, V0x, Azlehria, Ray Valeri, 0x1d00ffff, Diordna, "
				L"GeoffedUP, Tosti, Libcurl Team, Nayuki and You, the Miner!";
			this->textBox2->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			// 
			// lbl_builddate
			// 
			this->lbl_builddate->AutoSize = true;
			this->lbl_builddate->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->lbl_builddate->Location = System::Drawing::Point(336, 385);
			this->lbl_builddate->Name = L"lbl_builddate";
			this->lbl_builddate->Size = System::Drawing::Size(67, 15);
			this->lbl_builddate->TabIndex = 32;
			this->lbl_builddate->Text = L"Build Date:";
			// 
			// lbl_totalhashes
			// 
			this->lbl_totalhashes->AutoSize = true;
			this->lbl_totalhashes->Enabled = false;
			this->lbl_totalhashes->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Underline, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->lbl_totalhashes->Location = System::Drawing::Point(23, 385);
			this->lbl_totalhashes->Name = L"lbl_totalhashes";
			this->lbl_totalhashes->Size = System::Drawing::Size(82, 15);
			this->lbl_totalhashes->TabIndex = 33;
			this->lbl_totalhashes->Text = L"Total Hashes:";
			this->lbl_totalhashes->Visible = false;
			// 
			// AboutForm
			// 
			this->AcceptButton = this->button1;
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(769, 417);
			this->Controls->Add(this->lbl_totalhashes);
			this->Controls->Add(this->lbl_builddate);
			this->Controls->Add(this->textBox2);
			this->Controls->Add(this->label6);
			this->Controls->Add(this->button1);
			this->Controls->Add(this->label5);
			this->Controls->Add(this->label4);
			this->Controls->Add(this->label3);
			this->Controls->Add(this->label2);
			this->Controls->Add(this->textBox1);
			this->Controls->Add(this->label1);
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::FixedDialog;
			this->MaximizeBox = false;
			this->MinimizeBox = false;
			this->Name = L"AboutForm";
			this->ShowIcon = false;
			this->ShowInTaskbar = false;
			this->SizeGripStyle = System::Windows::Forms::SizeGripStyle::Hide;
			this->StartPosition = System::Windows::Forms::FormStartPosition::CenterParent;
			this->Text = L"About COSMiC!";
			this->Load += gcnew System::EventHandler(this, &AboutForm::AboutForm_Load);
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
	private: System::Void pictureBox1_Click(System::Object^  sender, System::EventArgs^  e) {
	}
	private: System::Void textBox1_TextChanged(System::Object^  sender, System::EventArgs^  e) {
	}
	private: System::Void AboutForm_Load(System::Object^ sender, System::EventArgs^ e)
	{
		// About Box loaded
		lbl_builddate->Text += __DATE__ + "  " + __TIME__;  // append build date/time to label
		//lbl_totalhashes->Text = "Total Hashes (est.):  " + gHashesEver;
	}

};

} //namespace
