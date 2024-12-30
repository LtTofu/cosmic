#pragma once
//void domesg_verb(const std::string/*&*/ to_print, const bool make_event, const unsigned short req_verbosity)
#ifndef CONFIGHWMON_FORM
#pragma message("Including " __FILE__ ", Last Modified: " __TIMESTAMP__ ".")
#define CONFIGHWMON_FORM

#include "coredefs.hpp"
#include "hwmon.h"

// move?
constexpr auto DEFAULT_MINRPM = 800;
constexpr auto DEFAULT_MAXGPUTEMP = 80;

// [old]:
#define CUDA_MAX_DEVICES 19

extern std::string gpuName[CUDA_MAX_DEVICES];  // Cosmic.cpp
// [todo]:	populate gpuName[] during detection regardless of device(s) dis/enabled.

namespace Cosmic {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
//	using namespace System::Data;
	using namespace System::Drawing;
	using namespace System::Configuration;

	/// <summary>
	/// Summary for ConfigHwMon
	/// </summary>
	public ref class ConfigHwMon : public System::Windows::Forms::Form
	{
	private: System::Windows::Forms::TextBox^ textbox_header;
	public:
		String^ ConfigKeys_MaxGPUTempEnable = "null";
		String^ ConfigKeys_MinRPMEnable = "null";
		String^ ConfigKeys_HwMonEnable = "null";
		String^ ConfigKeys_MaxGPUTemp = "null";
		String^ ConfigKeys_MinRPM = "null";
	private: System::Windows::Forms::TextBox^ textBox1;
	public:

		unsigned short hwMonFormDevID = 0;
		
	//public:
		ConfigHwMon(unsigned short devID)
		{ // constructor
			InitializeComponent();
			
			hwMonFormDevID = devID;	// copy passed parameter (CUDA device index) to public var

			// Managed String for the device intensity's key in the Config file
			ConfigKeys_MinRPM = "cudaDevice" + Convert::ToString(hwMonFormDevID) + "MinRPM";
			ConfigKeys_MaxGPUTemp = "cudaDevice" + Convert::ToString(hwMonFormDevID) + "MaxGPUTemp";
			ConfigKeys_HwMonEnable = "cudaDevice" + Convert::ToString(hwMonFormDevID) + "HWmon";
			ConfigKeys_MaxGPUTempEnable = "cudaDevice" + Convert::ToString(hwMonFormDevID) + "MaxGPUTempEnable";
			ConfigKeys_MinRPMEnable = "cudaDevice" + Convert::ToString(hwMonFormDevID) + "MinRPMEnable";
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~ConfigHwMon()
		{
			// destructor
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::GroupBox^  groupBox1;
	protected:
	private: System::Windows::Forms::Label^  label11;
	private: System::Windows::Forms::CheckBox^  checkbox_hwmon_enable;
	private: System::Windows::Forms::CheckBox^ checkbox_maxgputemp;

	private: System::Windows::Forms::CheckBox^ checkbox_minrpm;
	private: System::Windows::Forms::Label^  label1;
	private: System::Windows::Forms::NumericUpDown^ nud_maxgputemp;

	private: System::Windows::Forms::Label^ lbl_rpm;

	private: System::Windows::Forms::NumericUpDown^  nud_rpmthreshold;
	private: System::Windows::Forms::Label^  lbl_onalarm;
	private: System::Windows::Forms::ComboBox^  combobox_alarmbehavior;

	private: System::Windows::Forms::Button^  button1;
	private: System::Windows::Forms::Button^  button2;

	private: System::Windows::Forms::GroupBox^  groupbox_safety;

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
			this->groupBox1 = (gcnew System::Windows::Forms::GroupBox());
			this->label11 = (gcnew System::Windows::Forms::Label());
			this->checkbox_hwmon_enable = (gcnew System::Windows::Forms::CheckBox());
			this->checkbox_maxgputemp = (gcnew System::Windows::Forms::CheckBox());
			this->checkbox_minrpm = (gcnew System::Windows::Forms::CheckBox());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->nud_maxgputemp = (gcnew System::Windows::Forms::NumericUpDown());
			this->lbl_rpm = (gcnew System::Windows::Forms::Label());
			this->nud_rpmthreshold = (gcnew System::Windows::Forms::NumericUpDown());
			this->lbl_onalarm = (gcnew System::Windows::Forms::Label());
			this->combobox_alarmbehavior = (gcnew System::Windows::Forms::ComboBox());
			this->button1 = (gcnew System::Windows::Forms::Button());
			this->button2 = (gcnew System::Windows::Forms::Button());
			this->groupbox_safety = (gcnew System::Windows::Forms::GroupBox());
			this->textbox_header = (gcnew System::Windows::Forms::TextBox());
			this->textBox1 = (gcnew System::Windows::Forms::TextBox());
			this->groupBox1->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->nud_maxgputemp))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->nud_rpmthreshold))->BeginInit();
			this->groupbox_safety->SuspendLayout();
			this->SuspendLayout();
			// 
			// groupBox1
			// 
			this->groupBox1->Controls->Add(this->label11);
			this->groupBox1->Controls->Add(this->checkbox_hwmon_enable);
			this->groupBox1->Location = System::Drawing::Point(12, 43);
			this->groupBox1->Name = L"groupBox1";
			this->groupBox1->Size = System::Drawing::Size(347, 102);
			this->groupBox1->TabIndex = 0;
			this->groupBox1->TabStop = false;
			this->groupBox1->Text = L"GPU Health Monitoring";
			// 
			// label11
			// 
			this->label11->AutoSize = true;
			this->label11->Location = System::Drawing::Point(11, 21);
			this->label11->Name = L"label11";
			this->label11->Size = System::Drawing::Size(279, 39);
			this->label11->TabIndex = 0;
			this->label11->Text = L"Monitor temperatures, power draw and fan/pump speeds.\r\nReadings are displayed in "
				L"the columns next to each GPU.\r\n(This feature may not be supported on all devices"
				L".)";
			// 
			// checkbox_hwmon_enable
			// 
			this->checkbox_hwmon_enable->AutoSize = true;
			this->checkbox_hwmon_enable->Location = System::Drawing::Point(23, 73);
			this->checkbox_hwmon_enable->Name = L"checkbox_hwmon_enable";
			this->checkbox_hwmon_enable->Size = System::Drawing::Size(59, 17);
			this->checkbox_hwmon_enable->TabIndex = 1;
			this->checkbox_hwmon_enable->Text = L"Enable";
			this->checkbox_hwmon_enable->UseVisualStyleBackColor = true;
			this->checkbox_hwmon_enable->CheckedChanged += gcnew System::EventHandler(this, &ConfigHwMon::checkbox_hwmon_enable_CheckedChanged);
			// 
			// checkbox_maxgputemp
			// 
			this->checkbox_maxgputemp->AutoSize = true;
			this->checkbox_maxgputemp->Location = System::Drawing::Point(23, 65);
			this->checkbox_maxgputemp->Name = L"checkbox_maxgputemp";
			this->checkbox_maxgputemp->Size = System::Drawing::Size(147, 17);
			this->checkbox_maxgputemp->TabIndex = 0;
			this->checkbox_maxgputemp->Text = L"GPU Temp. Alarm above:";
			this->checkbox_maxgputemp->UseVisualStyleBackColor = true;
			this->checkbox_maxgputemp->CheckedChanged += gcnew System::EventHandler(this, &ConfigHwMon::checkbox_maxtemp_CheckedChanged);
			// 
			// checkbox_minrpm
			// 
			this->checkbox_minrpm->AutoSize = true;
			this->checkbox_minrpm->Location = System::Drawing::Point(23, 88);
			this->checkbox_minrpm->Name = L"checkbox_minrpm";
			this->checkbox_minrpm->Size = System::Drawing::Size(174, 17);
			this->checkbox_minrpm->TabIndex = 2;
			this->checkbox_minrpm->Text = L"Fan/Pump RPM Alarm if below:";
			this->checkbox_minrpm->UseVisualStyleBackColor = true;
			this->checkbox_minrpm->CheckedChanged += gcnew System::EventHandler(this, &ConfigHwMon::checkbox_minrpm_CheckedChanged);
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Location = System::Drawing::Point(303, 67);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(18, 13);
			this->label1->TabIndex = 7;
			this->label1->Text = L"°C";
			// 
			// nud_maxgputemp
			// 
			this->nud_maxgputemp->Location = System::Drawing::Point(203, 64);
			this->nud_maxgputemp->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 999, 0, 0, System::Int32::MinValue });
			this->nud_maxgputemp->Name = L"nud_maxgputemp";
			this->nud_maxgputemp->Size = System::Drawing::Size(96, 20);
			this->nud_maxgputemp->TabIndex = 1;
			// 
			// lbl_rpm
			// 
			this->lbl_rpm->AutoSize = true;
			this->lbl_rpm->Enabled = false;
			this->lbl_rpm->Location = System::Drawing::Point(303, 90);
			this->lbl_rpm->Name = L"lbl_rpm";
			this->lbl_rpm->Size = System::Drawing::Size(31, 13);
			this->lbl_rpm->TabIndex = 8;
			this->lbl_rpm->Text = L"RPM";
			// 
			// nud_rpmthreshold
			// 
			this->nud_rpmthreshold->Enabled = false;
			this->nud_rpmthreshold->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 50, 0, 0, 0 });
			this->nud_rpmthreshold->Location = System::Drawing::Point(203, 87);
			this->nud_rpmthreshold->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 10000, 0, 0, 0 });
			this->nud_rpmthreshold->Name = L"nud_rpmthreshold";
			this->nud_rpmthreshold->Size = System::Drawing::Size(96, 20);
			this->nud_rpmthreshold->TabIndex = 3;
			this->nud_rpmthreshold->ThousandsSeparator = true;
			// 
			// lbl_onalarm
			// 
			this->lbl_onalarm->AutoSize = true;
			this->lbl_onalarm->Enabled = false;
			this->lbl_onalarm->Location = System::Drawing::Point(11, 116);
			this->lbl_onalarm->Name = L"lbl_onalarm";
			this->lbl_onalarm->Size = System::Drawing::Size(192, 13);
			this->lbl_onalarm->TabIndex = 6;
			this->lbl_onalarm->Text = L"In an Alarm condition, the miner should:";
			this->lbl_onalarm->Visible = false;
			// 
			// combobox_alarmbehavior
			// 
			this->combobox_alarmbehavior->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->combobox_alarmbehavior->Enabled = false;
			this->combobox_alarmbehavior->FlatStyle = System::Windows::Forms::FlatStyle::Popup;
			this->combobox_alarmbehavior->FormattingEnabled = true;
			this->combobox_alarmbehavior->Items->AddRange(gcnew cli::array< System::Object^  >(2) { L"Inform Only, Keep Mining", L"Pause Mining on this Device" });
			this->combobox_alarmbehavior->Location = System::Drawing::Point(203, 113);
			this->combobox_alarmbehavior->Name = L"combobox_alarmbehavior";
			this->combobox_alarmbehavior->Size = System::Drawing::Size(130, 21);
			this->combobox_alarmbehavior->TabIndex = 4;
			this->combobox_alarmbehavior->Visible = false;
			// 
			// button1
			// 
			this->button1->DialogResult = System::Windows::Forms::DialogResult::OK;
			this->button1->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			this->button1->Location = System::Drawing::Point(221, 327);
			this->button1->Name = L"button1";
			this->button1->Size = System::Drawing::Size(112, 23);
			this->button1->TabIndex = 2;
			this->button1->Text = L"Save Settings";
			this->button1->UseVisualStyleBackColor = true;
			this->button1->Click += gcnew System::EventHandler(this, &ConfigHwMon::button1_Click);
			// 
			// button2
			// 
			this->button2->DialogResult = System::Windows::Forms::DialogResult::Cancel;
			this->button2->Location = System::Drawing::Point(83, 327);
			this->button2->Name = L"button2";
			this->button2->Size = System::Drawing::Size(112, 23);
			this->button2->TabIndex = 3;
			this->button2->Text = L"Discard Changes";
			this->button2->UseVisualStyleBackColor = true;
			// 
			// groupbox_safety
			// 
			this->groupbox_safety->Controls->Add(this->textBox1);
			this->groupbox_safety->Controls->Add(this->checkbox_maxgputemp);
			this->groupbox_safety->Controls->Add(this->combobox_alarmbehavior);
			this->groupbox_safety->Controls->Add(this->checkbox_minrpm);
			this->groupbox_safety->Controls->Add(this->lbl_onalarm);
			this->groupbox_safety->Controls->Add(this->label1);
			this->groupbox_safety->Controls->Add(this->nud_rpmthreshold);
			this->groupbox_safety->Controls->Add(this->nud_maxgputemp);
			this->groupbox_safety->Controls->Add(this->lbl_rpm);
			this->groupbox_safety->Location = System::Drawing::Point(12, 156);
			this->groupbox_safety->Name = L"groupbox_safety";
			this->groupbox_safety->Size = System::Drawing::Size(347, 150);
			this->groupbox_safety->TabIndex = 1;
			this->groupbox_safety->TabStop = false;
			this->groupbox_safety->Text = L"Safety Features";
			// 
			// textbox_header
			// 
			this->textbox_header->BackColor = System::Drawing::SystemColors::ButtonFace;
			this->textbox_header->BorderStyle = System::Windows::Forms::BorderStyle::None;
			this->textbox_header->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Underline, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->textbox_header->Location = System::Drawing::Point(12, 13);
			this->textbox_header->MaxLength = 200;
			this->textbox_header->Multiline = true;
			this->textbox_header->Name = L"textbox_header";
			this->textbox_header->ReadOnly = true;
			this->textbox_header->Size = System::Drawing::Size(379, 20);
			this->textbox_header->TabIndex = 4;
			this->textbox_header->TabStop = false;
			this->textbox_header->Text = L"Configuring: CUDA Device # ";
			// 
			// textBox1
			// 
			this->textBox1->BorderStyle = System::Windows::Forms::BorderStyle::None;
			this->textBox1->Cursor = System::Windows::Forms::Cursors::Arrow;
			this->textBox1->Location = System::Drawing::Point(14, 25);
			this->textBox1->Multiline = true;
			this->textBox1->Name = L"textBox1";
			this->textBox1->ReadOnly = true;
			this->textBox1->Size = System::Drawing::Size(244, 34);
			this->textBox1->TabIndex = 9;
			this->textBox1->TabStop = false;
			this->textBox1->Text = L"If the limits set here are exceeded, that device will be automatically paused.";
			// 
			// ConfigHwMon
			// 
			this->AcceptButton = this->button1;
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->CancelButton = this->button2;
			this->ClientSize = System::Drawing::Size(371, 362);
			this->ControlBox = false;
			this->Controls->Add(this->textbox_header);
			this->Controls->Add(this->groupbox_safety);
			this->Controls->Add(this->button2);
			this->Controls->Add(this->button1);
			this->Controls->Add(this->groupBox1);
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::FixedDialog;
			this->MaximizeBox = false;
			this->MinimizeBox = false;
			this->Name = L"ConfigHwMon";
			this->ShowIcon = false;
			this->ShowInTaskbar = false;
			this->SizeGripStyle = System::Windows::Forms::SizeGripStyle::Hide;
			this->StartPosition = System::Windows::Forms::FormStartPosition::CenterParent;
			this->Text = L"COSMiC - Hardware Safety Settings";
			this->Load += gcnew System::EventHandler(this, &ConfigHwMon::ConfigHwMon_Load);
			this->groupBox1->ResumeLayout(false);
			this->groupBox1->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->nud_maxgputemp))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->nud_rpmthreshold))->EndInit();
			this->groupbox_safety->ResumeLayout(false);
			this->groupbox_safety->PerformLayout();
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
	private: System::Void button1_Click(System::Object^  sender, System::EventArgs^  e)
	{ // "Save Settings" button clicked
		// get handle to the Configuration:
		System::Configuration::Configuration ^configHndl = ConfigurationManager::OpenExeConfiguration(ConfigurationUserLevel::None);

		// max GPU temp alarm and minimum RPM alarm en/disabled (set internally):
		gWatchQat_Devices[hwMonFormDevID].health_maxgputemp_enable = checkbox_maxgputemp->Checked;
		gWatchQat_Devices[hwMonFormDevID].health_minrpm_enable = checkbox_minrpm->Checked;

		// and the threshold values:
		gWatchQat_Devices[hwMonFormDevID].health_maxgputemp = (int)nud_maxgputemp->Value;
		gWatchQat_Devices[hwMonFormDevID].health_minrpm = (unsigned int)nud_rpmthreshold->Value;

		// check for keys first, only tries to remove if present:
		if (ConfigurationManager::AppSettings[ConfigKeys_MinRPM])  // if key is already present in .Config
			configHndl->AppSettings->Settings->Remove(ConfigKeys_MinRPM);
		configHndl->AppSettings->Settings->Add( ConfigKeys_MinRPM, nud_rpmthreshold->Value.ToString() ); // ConfigKeys_MinRPM is built in Form's constructor
		//
		if (ConfigurationManager::AppSettings[ConfigKeys_MaxGPUTemp])			// if key is already present
			configHndl->AppSettings->Settings->Remove(ConfigKeys_MaxGPUTemp);	// remove it first
		configHndl->AppSettings->Settings->Add(ConfigKeys_MaxGPUTemp, nud_maxgputemp->Value.ToString());  // ditto

		// set watchqat en/disabled internally
		gWatchQat_Devices[hwMonFormDevID].watchqat_enabled = checkbox_hwmon_enable->Checked;  // from checked-state (bool)

		// remove the watchqat enable key if it's present
		if (ConfigurationManager::AppSettings[ConfigKeys_HwMonEnable])			// if key is already present
			configHndl->AppSettings->Settings->Remove(ConfigKeys_HwMonEnable);  // remove it first
		configHndl->AppSettings->Settings->Add(ConfigKeys_HwMonEnable, checkbox_hwmon_enable->Checked.ToString());
		// add key for enabled status.
		
		// write "min rpm" alarm enabled setting:
		if (ConfigurationManager::AppSettings[ConfigKeys_MinRPMEnable])
			configHndl->AppSettings->Settings->Remove(ConfigKeys_MinRPMEnable);
		configHndl->AppSettings->Settings->Add( ConfigKeys_MinRPMEnable, checkbox_minrpm->Checked.ToString() );
		// write "max gpu temp" alarm enabled setting:
		if (ConfigurationManager::AppSettings[ConfigKeys_MaxGPUTempEnable])
			configHndl->AppSettings->Settings->Remove(ConfigKeys_MaxGPUTempEnable);
		configHndl->AppSettings->Settings->Add(ConfigKeys_MaxGPUTempEnable, checkbox_maxgputemp->Checked.ToString() );
		//


		// save out the configuration
		configHndl->Save(ConfigurationSaveMode::Modified);
		ConfigurationManager::RefreshSection("appSettings");
		Console::WriteLine("ConfigHwMon: Updated config.");
	}

	private: System::Void ConfigHwMon_Load(System::Object^  sender, System::EventArgs^  e)
	{
		// Form Loaded: get the current settings.
		// TODO/WIP: accept more than one selected device at once, settings entered in this Form will
		//			 be set for all selected GPUs

		// show the device user is configuring:
		String^ scratchMString = gcnew String(gpuName[hwMonFormDevID].c_str() );
		textbox_header->Text += hwMonFormDevID.ToString() + " (" + scratchMString + ")";

		// debug:
		if (gVerbosity == V_DEBUG) {
			printf("ConfigHwMon: got CUDA device # %d Max GPU Temp setting: %d C \n", hwMonFormDevID, gWatchQat_Devices[hwMonFormDevID].health_maxgputemp);
			printf("ConfigHwMon: got CUDA device # %d Min Tachometer setting: %d RPM \n", hwMonFormDevID, gWatchQat_Devices[hwMonFormDevID].health_minrpm);
			printf("maxGPUtemp alarm enabled: %s \n", std::to_string(gWatchQat_Devices[hwMonFormDevID].health_maxgputemp_enable).c_str());
			printf("minRPM alarm enabled: %s \n", std::to_string(gWatchQat_Devices[hwMonFormDevID].health_minrpm_enable).c_str() );
		} // ^ remove

		// populate the controls with values from the internal settings in use:
		// checkbox control property of type bool <- bool
		checkbox_hwmon_enable->Checked = gWatchQat_Devices[hwMonFormDevID].watchqat_enabled;
		groupbox_safety->Enabled = checkbox_hwmon_enable->Enabled;
		lbl_rpm->Enabled = checkbox_hwmon_enable->Enabled;
		
		// if the GPU Max Temp setting is disabled : ...
		// or min rpm...
		
		// put the threshold values into the numericUpDown controls:
		nud_maxgputemp->Value = gWatchQat_Devices[hwMonFormDevID].health_maxgputemp;
		nud_rpmthreshold->Value = gWatchQat_Devices[hwMonFormDevID].health_minrpm;

		// set the checkboxes for MinRPM and MaxGPUTemp alarms en/disabled:
		checkbox_maxgputemp->Checked = gWatchQat_Devices[hwMonFormDevID].health_maxgputemp_enable;
		checkbox_minrpm->Checked = gWatchQat_Devices[hwMonFormDevID].health_minrpm_enable;

}

private: System::Void checkbox_maxtemp_CheckedChanged(System::Object^ sender, System::EventArgs^ e)
{
	// if unchecked, the value is set to 89C and the numericUpDown control is disabled
	if (checkbox_maxgputemp->Checked) {
		if (gWatchQat_Devices[hwMonFormDevID].health_maxgputemp_enable)	 // en/disabled (overheat threshold set?)
			nud_maxgputemp->Value = gWatchQat_Devices[hwMonFormDevID].health_maxgputemp;		// the internal value
		nud_maxgputemp->Enabled = true;
	}
	else
	{
		nud_maxgputemp->Enabled = false;
		nud_maxgputemp->Value = DEFAULT_MAXGPUTEMP;
	}

	// new:
	nud_maxgputemp->Enabled = checkbox_maxgputemp->Enabled;
}

private: System::Void checkbox_minrpm_CheckedChanged(System::Object^ sender, System::EventArgs^ e)
{
	nud_rpmthreshold->Enabled = checkbox_minrpm->Checked;
	lbl_rpm->Enabled = checkbox_minrpm->Checked;
}

private: System::Void checkbox_hwmon_enable_CheckedChanged(System::Object^ sender, System::EventArgs^ e)
{
	groupbox_safety->Enabled = checkbox_hwmon_enable->Checked;
}

};	//class ConfigHwMon

} //namespace

#else
#pragma message("Not re-including CONFIGHWMON_FORM")
#endif	//CONFIGHWMON_FORM