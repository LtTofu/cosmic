#pragma once
//#include "ConfigSoloMining.h"  // Load_SKey_From_Config()

namespace Cosmic {
#include "Keystore.h"	//<-----

	//#include <msclr/marshal.h>
	//#include <msclr/marshal_cppstd.h>
	//#include <msclr/marshal_windows.h>>
	//void domesg_verb(const std::string& to_print, const bool make_event, const unsigned short req_verbosity);

	//using namespace System;
	//using namespace System::ComponentModel;
	//using namespace System::Collections;
	//using namespace System::Windows::Forms;
	////using namespace System::Data;
	//using namespace System::Drawing;
	/// <summary>
	/// Simple password entry dialog (implemented as Windows Form).
	/// Works in conjunction with ConfigSoloMining form.
	/// </summary>
	public ref class EnterPasswordForm : public System::Windows::Forms::Form
	{
	public:
		EnterPasswordForm(void)
		{ //constructor
			InitializeComponent();
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~EnterPasswordForm()
		{
			if (components)
			{ //destructor
				delete components;
			}
		}
	private: System::Windows::Forms::TextBox^  textBox1;
	private: System::Windows::Forms::TextBox^  textbox_pw;
//	protected:
	private: System::Windows::Forms::Button^  button1;
	private: System::Windows::Forms::Button^  button2;
	private: System::Windows::Forms::Button^ button3;
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
			System::ComponentModel::ComponentResourceManager^ resources = (gcnew System::ComponentModel::ComponentResourceManager(Cosmic::EnterPasswordForm::typeid));
			this->textBox1 = (gcnew System::Windows::Forms::TextBox());
			this->textbox_pw = (gcnew System::Windows::Forms::TextBox());
			this->button1 = (gcnew System::Windows::Forms::Button());
			this->button2 = (gcnew System::Windows::Forms::Button());
			this->button3 = (gcnew System::Windows::Forms::Button());
			this->SuspendLayout();
			// 
			// textBox1
			// 
			this->textBox1->BorderStyle = System::Windows::Forms::BorderStyle::None;
			this->textBox1->Location = System::Drawing::Point(12, 15);
			this->textBox1->Name = L"textBox1";
			this->textBox1->ReadOnly = true;
			this->textBox1->Size = System::Drawing::Size(210, 13);
			this->textBox1->TabIndex = 2;
			this->textBox1->TabStop = false;
			this->textBox1->Text = L"Please enter password to load Keystore.";
			// 
			// textbox_pw
			// 
			this->textbox_pw->Location = System::Drawing::Point(12, 43);
			this->textbox_pw->Name = L"textbox_pw";
			this->textbox_pw->PasswordChar = '-';
			this->textbox_pw->Size = System::Drawing::Size(408, 20);
			this->textbox_pw->TabIndex = 0;
			this->textbox_pw->TextChanged += gcnew System::EventHandler(this, &EnterPasswordForm::Textbox_pw_TextChanged);
			// 
			// button1
			// 
			this->button1->DialogResult = System::Windows::Forms::DialogResult::OK;
			this->button1->Location = System::Drawing::Point(327, 79);
			this->button1->Name = L"button1";
			this->button1->Size = System::Drawing::Size(93, 23);
			this->button1->TabIndex = 1;
			this->button1->Text = L"OK";
			this->button1->UseVisualStyleBackColor = true;
			this->button1->Click += gcnew System::EventHandler(this, &EnterPasswordForm::button1_Click);
			// 
			// button2
			// 
			this->button2->DialogResult = System::Windows::Forms::DialogResult::Cancel;
			this->button2->Location = System::Drawing::Point(247, 79);
			this->button2->Name = L"button2";
			this->button2->Size = System::Drawing::Size(74, 23);
			this->button2->TabIndex = 2;
			this->button2->Text = L"Cancel";
			this->button2->UseVisualStyleBackColor = true;
			// 
			// button3
			// 
			this->button3->Location = System::Drawing::Point(393, 10);
			this->button3->Name = L"button3";
			this->button3->Size = System::Drawing::Size(27, 23);
			this->button3->TabIndex = 3;
			this->button3->Text = L"button3";
			this->button3->UseVisualStyleBackColor = true;
			this->button3->Visible = false;
			this->button3->Click += gcnew System::EventHandler(this, &EnterPasswordForm::Button3_Click);
			// 
			// EnterPasswordForm
			// 
			this->AcceptButton = this->button1;
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(435, 114);
			this->ControlBox = false;
			this->Controls->Add(this->button3);
			this->Controls->Add(this->button2);
			this->Controls->Add(this->button1);
			this->Controls->Add(this->textbox_pw);
			this->Controls->Add(this->textBox1);
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::FixedDialog;
			this->Icon = (cli::safe_cast<System::Drawing::Icon^>(resources->GetObject(L"$this.Icon")));
			this->MaximizeBox = false;
			this->MinimizeBox = false;
			this->Name = L"EnterPasswordForm";
			this->ShowIcon = false;
			this->ShowInTaskbar = false;
			this->SizeGripStyle = System::Windows::Forms::SizeGripStyle::Hide;
			this->StartPosition = System::Windows::Forms::FormStartPosition::CenterParent;
			this->Text = L"COSMiC";
			this->Load += gcnew System::EventHandler(this, &EnterPasswordForm::EnterPasswordForm_Load);
			this->ResumeLayout(false);
			this->PerformLayout();

		}

private: System::Void button1_Click(System::Object^  sender, System::EventArgs^  e)
{
	msclr::interop::marshal_context marshalctx;  // for marshalling managed->native types

	// password provided- try the encrypted acct
	if (Keystore::LoadSKeyFromConfig(marshalctx.marshal_as<std::string>(textbox_pw->Text))) {	// with pass. to generate unlock key.
		domesg_verb("Got account: " + gStr_SoloEthAddress, false, V_MORE);
		return;  // dialog will close
	}
	
	return;  // dialog will close. Default: no need to report error, handled by calling function.
}

private: System::Void Button3_Click(System::Object^ sender, System::EventArgs^ e)
{
	textbox_pw->Text = "Enter Password...";  // <---- REMOVE  (FIXME)
}

private: System::Void EnterPasswordForm_Load(System::Object^ sender, System::EventArgs^ e)
{
	if (gVerbosity >= V_DEBUG)
	  button3->Visible = true;
	else  button3->Visible = false;
}

// if the text in the password entry field has changed:
private: System::Void Textbox_pw_TextChanged(System::Object^ sender, System::EventArgs^ e)
{
	// enable the OK button if a PW is entered
	if (textbox_pw->Text->Length > 0)  button1->Enabled = true;
	 else  button1->Enabled = false;
}

}; //class EnterPasswordForm


} //namespace
