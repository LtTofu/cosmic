// ConfigIntensityForm : 
// 2020 LtTofu
#pragma once

#ifndef CONFIGINTENSITYFORM
#define CONFIGINTENSITYFORM

#include "defs.hpp"  // #include "defs.h"

#include "cuda_device.hpp"
//void Cuda_UpdateDeviceIntensity(const unsigned short deviceID);			// hashburner.cu
//extern unsigned int gCudaDeviceIntensities[CUDA_MAX_DEVICES];			// CosmicWind.cpp
//#include "generic_solver.hpp"											// <--- ? 
#define MULTIPLE_DEVICES_SELECTED UINT8_MAX

namespace Cosmic {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	//using namespace System::Data;
	using namespace System::Drawing;
	using namespace System::Configuration;

//extern unsigned short theCudaDevices[CUDA_MAX_DEVICES];

	public ref class ConfigIntensityForm : public System::Windows::Forms::Form
	{

//	public:  unsigned short *theCudaDevices;  // (if multiple) devices we're configuring
	public:  cli::array<unsigned short>^ theCudaDevices;
	public:  unsigned short singleDeviceNo = UINT8_MAX;		//UINT8_MAX indicates multiple devices selected (`theCudaDevices[]`).

	ConfigIntensityForm( unsigned short devID, unsigned short multipleDevices[] )
	{ // get the devices user selected in the DevicesView to set intensity for.
		InitializeComponent();
		theCudaDevices = gcnew cli::array<unsigned short>(32);
		for (unsigned short i = 0; i < CUDA_MAX_DEVICES; ++i)  theCudaDevices[i] = UINT8_MAX;  // initialize array to all "no device"

		if (devID != MULTIPLE_DEVICES_SELECTED)
			singleDeviceNo = devID;  // save argument to pub var, single device index#.
		 else
		 {
			// todo/fixme: this could be simplified.
			if (gVerbosity == V_DEBUG)  printf("Multiple devices selected, ConfigIntensityForm() with CUDA device #'s: ");
			for (unsigned short i = 0; i < CUDA_MAX_DEVICES; ++i)
			{ // skip empty elements (0xFF):
				if (multipleDevices[i] == UINT8_MAX)  // FIXME: in the calling func that instantiates too (CosmicWind.h) !
					continue;
				if (gVerbosity == V_DEBUG)  printf(" %u ", multipleDevices[i]);
				theCudaDevices[i] = multipleDevices[i];
			}
			singleDeviceNo = MULTIPLE_DEVICES_SELECTED;  //see multipleDevices[].
		 }	
	}

protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~ConfigIntensityForm()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::Label^  label2;
	protected:
	private: System::Windows::Forms::TextBox^  textbox_intensitynum;
	private: System::Windows::Forms::Label^  label3;
	private: System::Windows::Forms::Label^  label4;
	private: System::Windows::Forms::Label^  label5;
	private: System::Windows::Forms::TrackBar^  trackbar_intensity;
	private: System::Windows::Forms::Label^  label10;
	private: System::Windows::Forms::Button^  button1;
	private: System::Windows::Forms::Label^  label1;
	private: System::Windows::Forms::Button^  button2;
	private: System::Windows::Forms::GroupBox^ groupBox1;
	private: System::Windows::Forms::GroupBox^ groupBox2;
	private: System::Windows::Forms::Panel^ panel1;
	private: System::Windows::Forms::NumericUpDown^ nud_tpb;
	private: System::Windows::Forms::Label^ lbl_cacheconfig;
	private: System::Windows::Forms::ComboBox^ comboBox1;
	private: System::Windows::Forms::CheckBox^ checkBox1;
	private: System::Windows::Forms::NumericUpDown^ nud_grid;
	private: System::Windows::Forms::Label^ lbl_blocks;
	private: System::Windows::Forms::Label^ lbl_threads;
	private: System::Windows::Forms::NumericUpDown^ nud_threads;
	private: System::Windows::Forms::Label^ label7;
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
			System::ComponentModel::ComponentResourceManager^ resources = (gcnew System::ComponentModel::ComponentResourceManager(ConfigIntensityForm::typeid));
			this->label2 = (gcnew System::Windows::Forms::Label());
			this->textbox_intensitynum = (gcnew System::Windows::Forms::TextBox());
			this->label3 = (gcnew System::Windows::Forms::Label());
			this->label4 = (gcnew System::Windows::Forms::Label());
			this->label5 = (gcnew System::Windows::Forms::Label());
			this->trackbar_intensity = (gcnew System::Windows::Forms::TrackBar());
			this->label10 = (gcnew System::Windows::Forms::Label());
			this->button1 = (gcnew System::Windows::Forms::Button());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->button2 = (gcnew System::Windows::Forms::Button());
			this->groupBox1 = (gcnew System::Windows::Forms::GroupBox());
			this->groupBox2 = (gcnew System::Windows::Forms::GroupBox());
			this->panel1 = (gcnew System::Windows::Forms::Panel());
			this->nud_tpb = (gcnew System::Windows::Forms::NumericUpDown());
			this->lbl_cacheconfig = (gcnew System::Windows::Forms::Label());
			this->comboBox1 = (gcnew System::Windows::Forms::ComboBox());
			this->checkBox1 = (gcnew System::Windows::Forms::CheckBox());
			this->nud_grid = (gcnew System::Windows::Forms::NumericUpDown());
			this->lbl_blocks = (gcnew System::Windows::Forms::Label());
			this->lbl_threads = (gcnew System::Windows::Forms::Label());
			this->nud_threads = (gcnew System::Windows::Forms::NumericUpDown());
			this->label7 = (gcnew System::Windows::Forms::Label());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->trackbar_intensity))->BeginInit();
			this->groupBox1->SuspendLayout();
			this->groupBox2->SuspendLayout();
			this->panel1->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->nud_tpb))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->nud_grid))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->nud_threads))->BeginInit();
			this->SuspendLayout();
			// 
			// label2
			// 
			this->label2->AutoSize = true;
			this->label2->Location = System::Drawing::Point(29, 76);
			this->label2->Name = L"label2";
			this->label2->Size = System::Drawing::Size(13, 13);
			this->label2->TabIndex = 5;
			this->label2->Text = L"8";
			// 
			// textbox_intensitynum
			// 
			this->textbox_intensitynum->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->textbox_intensitynum->Location = System::Drawing::Point(479, 30);
			this->textbox_intensitynum->Name = L"textbox_intensitynum";
			this->textbox_intensitynum->ReadOnly = true;
			this->textbox_intensitynum->Size = System::Drawing::Size(39, 20);
			this->textbox_intensitynum->TabIndex = 2;
			this->textbox_intensitynum->TabStop = false;
			this->textbox_intensitynum->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			// 
			// label3
			// 
			this->label3->AutoSize = true;
			this->label3->Location = System::Drawing::Point(454, 76);
			this->label3->Name = L"label3";
			this->label3->Size = System::Drawing::Size(19, 13);
			this->label3->TabIndex = 8;
			this->label3->Text = L"32";
			// 
			// label4
			// 
			this->label4->AutoSize = true;
			this->label4->Location = System::Drawing::Point(165, 76);
			this->label4->Name = L"label4";
			this->label4->Size = System::Drawing::Size(19, 13);
			this->label4->TabIndex = 6;
			this->label4->Text = L"16";
			// 
			// label5
			// 
			this->label5->AutoSize = true;
			this->label5->Location = System::Drawing::Point(307, 76);
			this->label5->Name = L"label5";
			this->label5->Size = System::Drawing::Size(19, 13);
			this->label5->TabIndex = 7;
			this->label5->Text = L"24";
			// 
			// trackbar_intensity
			// 
			this->trackbar_intensity->LargeChange = 1;
			this->trackbar_intensity->Location = System::Drawing::Point(23, 23);
			this->trackbar_intensity->Maximum = 32;
			this->trackbar_intensity->Minimum = 8;
			this->trackbar_intensity->Name = L"trackbar_intensity";
			this->trackbar_intensity->Size = System::Drawing::Size(450, 53);
			this->trackbar_intensity->TabIndex = 1;
			this->trackbar_intensity->Value = 24;
			this->trackbar_intensity->ValueChanged += gcnew System::EventHandler(this, &ConfigIntensityForm::trackbar_intensity_ValueChanged);
			// 
			// label10
			// 
			this->label10->AutoSize = true;
			this->label10->Location = System::Drawing::Point(20, 113);
			this->label10->Name = L"label10";
			this->label10->Size = System::Drawing::Size(368, 13);
			this->label10->TabIndex = 9;
			this->label10->Text = L"Ideas: 24 (GTX970), 25 (GTX1060-3GB), 27-28 (1070ti/1080), 29-31 (1080ti)\r\n";
			// 
			// button1
			// 
			this->button1->DialogResult = System::Windows::Forms::DialogResult::Yes;
			this->button1->Location = System::Drawing::Point(404, 444);
			this->button1->Name = L"button1";
			this->button1->Size = System::Drawing::Size(124, 23);
			this->button1->TabIndex = 3;
			this->button1->Text = L"Save Setting";
			this->button1->UseVisualStyleBackColor = true;
			this->button1->Click += gcnew System::EventHandler(this, &ConfigIntensityForm::button1_Click);
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Location = System::Drawing::Point(15, 20);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(523, 91);
			this->label1->TabIndex = 0;
			this->label1->Text = resources->GetString(L"label1.Text");
			// 
			// button2
			// 
			this->button2->DialogResult = System::Windows::Forms::DialogResult::Cancel;
			this->button2->Location = System::Drawing::Point(250, 444);
			this->button2->Name = L"button2";
			this->button2->Size = System::Drawing::Size(124, 23);
			this->button2->TabIndex = 4;
			this->button2->Text = L"Discard Changes";
			this->button2->UseVisualStyleBackColor = true;
			// 
			// groupBox1
			// 
			this->groupBox1->Controls->Add(this->textbox_intensitynum);
			this->groupBox1->Controls->Add(this->trackbar_intensity);
			this->groupBox1->Controls->Add(this->label3);
			this->groupBox1->Controls->Add(this->label4);
			this->groupBox1->Controls->Add(this->label10);
			this->groupBox1->Controls->Add(this->label2);
			this->groupBox1->Controls->Add(this->label5);
			this->groupBox1->Location = System::Drawing::Point(10, 132);
			this->groupBox1->Name = L"groupBox1";
			this->groupBox1->Size = System::Drawing::Size(543, 140);
			this->groupBox1->TabIndex = 10;
			this->groupBox1->TabStop = false;
			this->groupBox1->Text = L"Intensity";
			// 
			// groupBox2
			// 
			this->groupBox2->AutoSize = true;
			this->groupBox2->Controls->Add(this->panel1);
			this->groupBox2->Location = System::Drawing::Point(10, 278);
			this->groupBox2->Name = L"groupBox2";
			this->groupBox2->Size = System::Drawing::Size(543, 148);
			this->groupBox2->TabIndex = 12;
			this->groupBox2->TabStop = false;
			this->groupBox2->Text = L"Advanced";
			// 
			// panel1
			// 
			this->panel1->Controls->Add(this->nud_tpb);
			this->panel1->Controls->Add(this->lbl_cacheconfig);
			this->panel1->Controls->Add(this->comboBox1);
			this->panel1->Controls->Add(this->checkBox1);
			this->panel1->Controls->Add(this->nud_grid);
			this->panel1->Controls->Add(this->lbl_blocks);
			this->panel1->Controls->Add(this->lbl_threads);
			this->panel1->Controls->Add(this->nud_threads);
			this->panel1->Controls->Add(this->label7);
			this->panel1->Dock = System::Windows::Forms::DockStyle::Fill;
			this->panel1->Enabled = false;
			this->panel1->Location = System::Drawing::Point(3, 16);
			this->panel1->Name = L"panel1";
			this->panel1->Size = System::Drawing::Size(537, 129);
			this->panel1->TabIndex = 10;
			// 
			// nud_tpb
			// 
			this->nud_tpb->Enabled = false;
			this->nud_tpb->Location = System::Drawing::Point(61, 62);
			this->nud_tpb->Name = L"nud_tpb";
			this->nud_tpb->Size = System::Drawing::Size(120, 20);
			this->nud_tpb->TabIndex = 5;
			// 
			// lbl_cacheconfig
			// 
			this->lbl_cacheconfig->AutoSize = true;
			this->lbl_cacheconfig->Enabled = false;
			this->lbl_cacheconfig->Location = System::Drawing::Point(221, 12);
			this->lbl_cacheconfig->Name = L"lbl_cacheconfig";
			this->lbl_cacheconfig->Size = System::Drawing::Size(71, 13);
			this->lbl_cacheconfig->TabIndex = 6;
			this->lbl_cacheconfig->Text = L"Cache Config";
			// 
			// comboBox1
			// 
			this->comboBox1->Enabled = false;
			this->comboBox1->FormattingEnabled = true;
			this->comboBox1->Location = System::Drawing::Point(296, 9);
			this->comboBox1->Name = L"comboBox1";
			this->comboBox1->Size = System::Drawing::Size(121, 21);
			this->comboBox1->TabIndex = 7;
			// 
			// checkBox1
			// 
			this->checkBox1->AutoSize = true;
			this->checkBox1->Enabled = false;
			this->checkBox1->Location = System::Drawing::Point(9, 88);
			this->checkBox1->Name = L"checkBox1";
			this->checkBox1->Size = System::Drawing::Size(87, 17);
			this->checkBox1->TabIndex = 8;
			this->checkBox1->Text = L"Lazy Launch";
			this->checkBox1->UseVisualStyleBackColor = true;
			// 
			// nud_grid
			// 
			this->nud_grid->Enabled = false;
			this->nud_grid->Location = System::Drawing::Point(61, 10);
			this->nud_grid->Name = L"nud_grid";
			this->nud_grid->Size = System::Drawing::Size(120, 20);
			this->nud_grid->TabIndex = 0;
			// 
			// lbl_blocks
			// 
			this->lbl_blocks->AutoSize = true;
			this->lbl_blocks->Enabled = false;
			this->lbl_blocks->Location = System::Drawing::Point(6, 12);
			this->lbl_blocks->Name = L"lbl_blocks";
			this->lbl_blocks->Size = System::Drawing::Size(39, 13);
			this->lbl_blocks->TabIndex = 1;
			this->lbl_blocks->Text = L"Blocks";
			// 
			// lbl_threads
			// 
			this->lbl_threads->AutoSize = true;
			this->lbl_threads->Enabled = false;
			this->lbl_threads->Location = System::Drawing::Point(6, 38);
			this->lbl_threads->Name = L"lbl_threads";
			this->lbl_threads->Size = System::Drawing::Size(46, 13);
			this->lbl_threads->TabIndex = 2;
			this->lbl_threads->Text = L"Threads";
			// 
			// nud_threads
			// 
			this->nud_threads->Enabled = false;
			this->nud_threads->Location = System::Drawing::Point(61, 36);
			this->nud_threads->Name = L"nud_threads";
			this->nud_threads->Size = System::Drawing::Size(120, 20);
			this->nud_threads->TabIndex = 3;
			// 
			// label7
			// 
			this->label7->AutoSize = true;
			this->label7->Enabled = false;
			this->label7->Location = System::Drawing::Point(6, 64);
			this->label7->Name = L"label7";
			this->label7->Size = System::Drawing::Size(28, 13);
			this->label7->TabIndex = 4;
			this->label7->Text = L"TPB";
			// 
			// ConfigIntensityForm
			// 
			this->AcceptButton = this->button1;
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->CancelButton = this->button2;
			this->ClientSize = System::Drawing::Size(562, 479);
			this->ControlBox = false;
			this->Controls->Add(this->groupBox2);
			this->Controls->Add(this->groupBox1);
			this->Controls->Add(this->button2);
			this->Controls->Add(this->label1);
			this->Controls->Add(this->button1);
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::FixedDialog;
			this->HelpButton = true;
			this->MaximizeBox = false;
			this->MinimizeBox = false;
			this->Name = L"ConfigIntensityForm";
			this->ShowIcon = false;
			this->ShowInTaskbar = false;
			this->SizeGripStyle = System::Windows::Forms::SizeGripStyle::Hide;
			this->StartPosition = System::Windows::Forms::FormStartPosition::CenterParent;
			this->Text = L"COSMiC - Device Intensity";
			this->Load += gcnew System::EventHandler(this, &ConfigIntensityForm::ConfigIntensityForm_Load);
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->trackbar_intensity))->EndInit();
			this->groupBox1->ResumeLayout(false);
			this->groupBox1->PerformLayout();
			this->groupBox2->ResumeLayout(false);
			this->panel1->ResumeLayout(false);
			this->panel1->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->nud_tpb))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->nud_grid))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->nud_threads))->EndInit();
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion

	private: System::Void button1_Click(System::Object^ sender, System::EventArgs^ e)
	{ // Save button clicked

		// get handle to the Configuration:
		System::Configuration::Configuration^ configHndl = ConfigurationManager::OpenExeConfiguration(ConfigurationUserLevel::None);
		String^ configKey = "";

		if (singleDeviceNo != UINT8_MAX)  // single device being set up
		{
			// Managed String for the device intensity's key in the Config file
			String^ configKey = "cudaDevice" + Convert::ToString(singleDeviceNo) + "Intensity";
			unsigned short scratchDevNum = UINT8_MAX;

			// TODO: prevent exception by only removing key if it exists already!
			if (ConfigurationManager::AppSettings[configKey])
				configHndl->AppSettings->Settings->Remove(configKey);
			configHndl->AppSettings->Settings->Add(configKey, textbox_intensitynum->Text); // Slider position -> String^

			Solvers[singleDeviceNo]->intensity = static_cast<unsigned int>(trackbar_intensity->Value);
			Solvers[singleDeviceNo]->SetIntensity();	//<--
		}
		else
		{ //multiple devices:

			for (unsigned short i = 0; i < CUDA_MAX_DEVICES; ++i)
			{
				if (theCudaDevices[i] == UINT8_MAX)
					continue;											// skip empty elements

				configKey = "cudaDevice" + i.ToString() + "Intensity";
				if ( ConfigurationManager::AppSettings[configKey] )		 // check if key exists first
					configHndl->AppSettings->Settings->Remove(configKey);  // remove if present (prevent exc.).
				//
				configHndl->AppSettings->Settings->Add(configKey, textbox_intensitynum->Text); // Slider position in String^ format here
				//
				Solvers[i]->intensity = static_cast<unsigned int>(trackbar_intensity->Value);  // set intensity internally
				Solvers[i]->SetIntensity();	//update # of threads	[WIP]: changing intensity while mining.

			}
			// ...

		}
		
		// save out the configuration
		configHndl->Save(ConfigurationSaveMode::Modified);
		ConfigurationManager::RefreshSection("appSettings");
		Console::WriteLine("Updated config.");
	}
private: System::Void trackbar_intensity_ValueChanged(System::Object^  sender, System::EventArgs^  e)
{
	textbox_intensitynum->Text = Convert::ToString(trackbar_intensity->Value);
}

private: System::Void ConfigIntensityForm_Load(System::Object^  sender, System::EventArgs^  e)
{
	if (singleDeviceNo == MULTIPLE_DEVICES_SELECTED) { // setting intensity on >1 device (use array of device#s instead)
		LOG_IF_F(INFO, DEBUGMODE, "ConfigIntensityForm: configuring multiple devices ");
		trackbar_intensity->Value = DEFAULT_CUDA_INTENSITY;
		textbox_intensitynum->Text = trackbar_intensity->Value.ToString();
	} else { //single device
		if (singleDeviceNo >= CUDA_MAX_DEVICES) { //REMOVE? <---
			MessageBox::Show("unexpected device no.: " + singleDeviceNo.ToString(), "Error");
			return;
		}

		LOG_IF_F(INFO, DEBUGMODE, "OK- configuring intensity of single device# %u \n", singleDeviceNo);	// debug only <--
		trackbar_intensity->Value = static_cast<int>(Solvers[singleDeviceNo]->intensity);		// just keep these 
		textbox_intensitynum->Text = trackbar_intensity->Value.ToString();	// "
	}
}

};


} //namespace

#else
#pragma message("Not re-including " __FILE__ ", modified " __TIMESTAMP__)
#endif	//CONFIGINTENSITYFORM
