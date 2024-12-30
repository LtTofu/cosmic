#pragma once

#include <msclr/marshal.h>
#include <msclr/marshal_cppstd.h>
#include <string>

namespace Cosmic {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	//using namespace System::Data;
	using namespace System::Drawing;

	/// <summary>
	/// Summary for TxReceiptForm
	/// </summary>
	public ref class TxReceiptForm : public System::Windows::Forms::Form
	{
	public:
		String^ receiptJson = "";
		int txView_solutionNo = -1;

		TxReceiptForm(const int txView_itemNo, String^ txHashStr, String^ receiptStr)
		{ // txView_itemNo will probably be useful later
			InitializeComponent();
			
			/* input already checked in	function that instantiates this Form */
			receiptJson = receiptStr;
			txView_solutionNo = txView_itemNo;
			lbl_txhash->Text = txHashStr;
			textBox1->AppendText(receiptStr);
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~TxReceiptForm()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::TextBox^ textBox1;
	protected:
	private: System::Windows::Forms::Button^ button1;
	private: System::Windows::Forms::Label^ lbl_txhash;
	private: System::Windows::Forms::Label^ lbl_txhash_pref;
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
			this->textBox1 = (gcnew System::Windows::Forms::TextBox());
			this->button1 = (gcnew System::Windows::Forms::Button());
			this->lbl_txhash = (gcnew System::Windows::Forms::Label());
			this->lbl_txhash_pref = (gcnew System::Windows::Forms::Label());
			this->SuspendLayout();
			// 
			// textBox1
			// 
			this->textBox1->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
				| System::Windows::Forms::AnchorStyles::Left)
				| System::Windows::Forms::AnchorStyles::Right));
			this->textBox1->Location = System::Drawing::Point(12, 12);
			this->textBox1->MaxLength = 300;
			this->textBox1->Multiline = true;
			this->textBox1->Name = L"textBox1";
			this->textBox1->ReadOnly = true;
			this->textBox1->ScrollBars = System::Windows::Forms::ScrollBars::Vertical;
			this->textBox1->Size = System::Drawing::Size(553, 230);
			this->textBox1->TabIndex = 0;
			this->textBox1->TabStop = false;
			// 
			// button1
			// 
			this->button1->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
			this->button1->DialogResult = System::Windows::Forms::DialogResult::OK;
			this->button1->Location = System::Drawing::Point(491, 248);
			this->button1->Name = L"button1";
			this->button1->Size = System::Drawing::Size(75, 23);
			this->button1->TabIndex = 2;
			this->button1->Text = L"Close";
			this->button1->UseVisualStyleBackColor = true;
			this->button1->Click += gcnew System::EventHandler(this, &TxReceiptForm::Button1_Click);
			// 
			// lbl_txhash
			// 
			this->lbl_txhash->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
			this->lbl_txhash->AutoSize = true;
			this->lbl_txhash->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Underline, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->lbl_txhash->ForeColor = System::Drawing::Color::Blue;
			this->lbl_txhash->Location = System::Drawing::Point(64, 253);
			this->lbl_txhash->Name = L"lbl_txhash";
			this->lbl_txhash->Size = System::Drawing::Size(24, 13);
			this->lbl_txhash->TabIndex = 1;
			this->lbl_txhash->Text = L"0x0";
			this->lbl_txhash->Click += gcnew System::EventHandler(this, &TxReceiptForm::Lbl_txhash_Click);
			// 
			// lbl_txhash_pref
			// 
			this->lbl_txhash_pref->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
			this->lbl_txhash_pref->AutoSize = true;
			this->lbl_txhash_pref->Location = System::Drawing::Point(12, 253);
			this->lbl_txhash_pref->Name = L"lbl_txhash_pref";
			this->lbl_txhash_pref->Size = System::Drawing::Size(52, 13);
			this->lbl_txhash_pref->TabIndex = 3;
			this->lbl_txhash_pref->Text = L"TX Hash:";
			// 
			// TxReceiptForm
			// 
			this->AcceptButton = this->button1;
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->AutoSizeMode = System::Windows::Forms::AutoSizeMode::GrowAndShrink;
			this->ClientSize = System::Drawing::Size(577, 279);
			this->Controls->Add(this->lbl_txhash_pref);
			this->Controls->Add(this->lbl_txhash);
			this->Controls->Add(this->button1);
			this->Controls->Add(this->textBox1);
			this->MaximizeBox = false;
			this->MinimizeBox = false;
			this->Name = L"TxReceiptForm";
			this->ShowInTaskbar = false;
			this->StartPosition = System::Windows::Forms::FormStartPosition::CenterParent;
			this->Text = L"Transaction Receipt";
			this->Load += gcnew System::EventHandler(this, &TxReceiptForm::TxReceiptForm_Load);
			this->Shown += gcnew System::EventHandler(this, &TxReceiptForm::TxReceiptForm_Shown);
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
	private: System::Void TxReceiptForm_Shown(System::Object^ sender, System::EventArgs^ e)
	{
		textBox1->Text = receiptJson;
		//
	}
	private: System::Void Button1_Click(System::Object^ sender, System::EventArgs^ e)
	{
		this->Close();  // close this form (X close button also works)
	}
	private: System::Void TxReceiptForm_Load(System::Object^ sender, System::EventArgs^ e)
	{
		//
	}
private: System::Void Lbl_txhash_Click(System::Object^ sender, System::EventArgs^ e)
{ // Tx Hash was clicked

	msclr::interop::marshal_context marshalctx;  // for marshalling to native types from managed ones
	std::string txhash_str = marshalctx.marshal_as<std::string>(lbl_txhash->Text);  // from txhash column

	// must be appropriate length, prepended by `0x`, hexadecimal characters only. 32 bytes = 64 hex digits + 0x = 66.
	// [TODO] just in case? sanity checked in the calling function that spawned this txreceiptform. <-
	//if (txhash_str.length() != 66 || txhash_str.substr(0, 2) != "0x" || txhash_str.substr(2).find_first_not_of(DEF_HEXCHARS) != std::string::npos)
	if (!checkString(txhash_str, 66, true, true)) {
		Console::WriteLine("Bad TxHash! Not opening in browser.");
		return;
	}

	// open chosen block explorer in the system's selected browser w/ the TxHash (TODO: hardcoded to Etherscan for now).
	LOG_IF_F(INFO, HIGHVERBOSITY, "Opening Tx on Etherscan in default web browser ");
	try {
		System::String^ MStr = "https://www.etherscan.io/tx/" + gcnew String(txhash_str.c_str());
		System::Diagnostics::Process::Start(MStr);  // TODO: make user-selectable (ETC blockchain etc.)
	}
	catch (System::ComponentModel::Win32Exception^ other) {
		MessageBox::Show("Exception caught: " + other->Message, "COSMiC", MessageBoxButtons::OK, MessageBoxIcon::Error);
		LOG_F(WARNING, "TxReceiptForm:	Exception caught trying to launch browser. Aborting! ");
		return;
	}

}

};
}
