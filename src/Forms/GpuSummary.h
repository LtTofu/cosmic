// GpuSummary.h : windows form that shows some cuda device statistics
// 2020 LtTofu
#pragma once

#ifndef FORM_GPUSUMMARY
#define FORM_GPUSUMMARY
//#include "../Core/defs.h"
//#include "../Core/Core.h"  // optionally
//#include <cuda.h>
//#include <nvapi.h>
#include <nvml.h>
#include <cinttypes>  // was: #include <stdint.h>
#include <string>
#include <iostream>

//#include "network.hpp"
#include "hwmon.h"
#include "generic_solver.hpp"

//using namespace System;
//using namespace msclr::interop;

//#include "hashburner.cuh" ?
std::string Cuda_GetDeviceNames(int devIndex);						// hashburner.cu
std::string Cuda_GetDeviceBusInfo(int devIndex);  					// "

extern uint64_t gU64_DifficultyNo;  // or include "network.hpp"
extern unsigned short gVerbosity;
extern bool gCudaSolving;

extern double gNum_Hashrate[CUDA_MAX_DEVICES];						// Cosmic.cpp
extern uint64_t gCuda_HashCounts[CUDA_MAX_DEVICES];					// hashburner.cu
extern unsigned short gpusSummarized[CUDA_MAX_DEVICES];

//#include "cuda_device.hpp"
//extern bool gCudaDeviceHWMonEnabled[CUDA_MAX_DEVICES];			// CosmicWind.h
//extern struct WQdevice gWatchQat_Devices[CUDA_MAX_DEVICES];		// WatchQat.cpp <--- New!
//extern unsigned int  gCudaDeviceIntensities[CUDA_MAX_DEVICES];	// CosmicWind.cpp

//bool NvAPI_DoCall(/*const*/ NvAPI_Status nvapiResult, const std::string theTask, const unsigned short deviceNo);  // WatchQat.cpp
bool NVML_DoCall(/*const*/ nvmlReturn_t nvmlResult, const std::string theTask, const unsigned short deviceNo);		// "

namespace Cosmic {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Windows::Forms;
	//using namespace System::Data;
	//using namespace System::Drawing;
	//using namespace System::Configuration;

	public ref class GpuSummary : public System::Windows::Forms::Form
	{
	public:
		int gpuDetailsDeviceID = -1;								// -1 = null value (indicates >1 cards selected)
		unsigned short howManyDevices = 0;							// how many selected & being summarized.
		String^ gpusSummarized_MStr = "";
	//public:
	private: System::Windows::Forms::Timer^  timer_devicedetails;		// updates our double-buffered treeview's stats
	private: System::Windows::Forms::ToolTip^  tooltip_gpudetails;
	private: System::Windows::Forms::Panel^  panel_for_smoothtreeview;	// for placement and sizing of the treeview
	private: System::Windows::Forms::NumericUpDown^  numericUpDown1;
	private: System::Windows::Forms::GroupBox^  groupbox_gpudetails;
	private: System::Windows::Forms::GroupBox^  groupBox1;
	private: System::Windows::Forms::TextBox^ textBox1;
	private: System::Windows::Forms::Label^ label1;

	public:
		ref class SmoothTreeView : TreeView
		{
			 public:
			 SmoothTreeView()
			 {
				 //printf("\n- SmoothTreeView: spawning double-buffered treeview \n");
				 SetStyle(ControlStyles::OptimizedDoubleBuffer, true);
				 SetStyle(ControlStyles::DoubleBuffer, true);
				 SetStyle(ControlStyles::AllPaintingInWmPaint, true);
				 SetStyle(ControlStyles::Opaque, true);
				 UpdateStyles();
			 }
		 };

		// old: GpuSummary(int cudaDeviceID) {...}
		// new version allows for multiple selected cards in the Devices View to be passed in at once
		GpuSummary( int cudaDeviceID, unsigned short multipleDevices[] )
		{
			InitializeComponent();
			
			if (cudaDeviceID == -1)
			{
				unsigned short i{ 0 };
				// remove this (TESTME)
				if (gVerbosity == V_DEBUG) {
					printf("Multiple devices selected, GpuSummary() with CUDA device indices: ");
					for (i = 0; i < CUDA_MAX_DEVICES; ++i)
						printf(" %d ", multipleDevices[i] );
				}
				//
				String^ gpusSummarized_MStr = "CUDA GPUs ";
				for (i = 0; i < CUDA_MAX_DEVICES; ++i)
				{
					if (multipleDevices[i] == UINT8_MAX)		 // 0xFF = not a device
						continue;  // skip

					++howManyDevices;
					gpusSummarized[i] = multipleDevices[i];		 // copy real CUDA device #
					gpusSummarized_MStr += i.ToString() + "  ";  // write to devices mng'd string
				}
				
				if (NORMALVERBOSITY) { printf("%d CUDA Devices being summarized: \n", howManyDevices); } //dbg <--
				if (DEBUGMODE) { Console::WriteLine(gpusSummarized_MStr); }								 //dbg <--
				gpuDetailsDeviceID = -1;  // null, this var for individual device summary only
			}
			else
			{
				if (gVerbosity == V_DEBUG)  printf("Got single CUDA device index # %d. \n", cudaDeviceID);
				gpuDetailsDeviceID = cudaDeviceID;	// copy passed parameter (CUDA device index) to public var
				howManyDevices = 1;
			}

			//this->DBTreeView->Nodes[0]->Expand();  // TODO: expand the tree nodes to taste (consider screen area)
			//this->DBTreeView->Nodes[1]->Expand();
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~GpuSummary()
		{
			if (components)
			{
				delete components;
			}
		}

	protected:

	private: System::Windows::Forms::Button^  button2;
	private: System::ComponentModel::IContainer^  components;
	private: SmoothTreeView^ DBTreeView;	 // pointer for accessing the tree view in this form
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
			System::ComponentModel::ComponentResourceManager^ resources = (gcnew System::ComponentModel::ComponentResourceManager(GpuSummary::typeid));
			this->button2 = (gcnew System::Windows::Forms::Button());
			this->timer_devicedetails = (gcnew System::Windows::Forms::Timer(this->components));
			this->tooltip_gpudetails = (gcnew System::Windows::Forms::ToolTip(this->components));
			this->numericUpDown1 = (gcnew System::Windows::Forms::NumericUpDown());
			this->groupBox1 = (gcnew System::Windows::Forms::GroupBox());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->groupbox_gpudetails = (gcnew System::Windows::Forms::GroupBox());
			this->panel_for_smoothtreeview = (gcnew System::Windows::Forms::Panel());
			this->textBox1 = (gcnew System::Windows::Forms::TextBox());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown1))->BeginInit();
			this->groupBox1->SuspendLayout();
			this->groupbox_gpudetails->SuspendLayout();
			this->SuspendLayout();
			// 
			// button2
			// 
			this->button2->DialogResult = System::Windows::Forms::DialogResult::OK;
			this->button2->Location = System::Drawing::Point(355, 285);
			this->button2->Name = L"button2";
			this->button2->Size = System::Drawing::Size(134, 23);
			this->button2->TabIndex = 2;
			this->button2->Text = L"Close";
			this->button2->UseVisualStyleBackColor = true;
			this->button2->Click += gcnew System::EventHandler(this, &GpuSummary::button2_Click);
			// 
			// timer_devicedetails
			// 
			this->timer_devicedetails->Interval = 180;
			this->timer_devicedetails->Tick += gcnew System::EventHandler(this, &GpuSummary::timer_devicedetails_Tick);
			// 
			// tooltip_gpudetails
			// 
			this->tooltip_gpudetails->AutomaticDelay = 0;
			this->tooltip_gpudetails->AutoPopDelay = 32000;
			this->tooltip_gpudetails->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(191)),
				static_cast<System::Int32>(static_cast<System::Byte>(222)), static_cast<System::Int32>(static_cast<System::Byte>(255)));
			this->tooltip_gpudetails->InitialDelay = 400;
			this->tooltip_gpudetails->ReshowDelay = 150;
			this->tooltip_gpudetails->ToolTipIcon = System::Windows::Forms::ToolTipIcon::Info;
			this->tooltip_gpudetails->ToolTipTitle = L"GPU Details";
			this->tooltip_gpudetails->UseAnimation = false;
			this->tooltip_gpudetails->UseFading = false;
			// 
			// numericUpDown1
			// 
			this->numericUpDown1->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 25, 0, 0, 0 });
			this->numericUpDown1->Location = System::Drawing::Point(15, 25);
			this->numericUpDown1->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 10000, 0, 0, 0 });
			this->numericUpDown1->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 50, 0, 0, 0 });
			this->numericUpDown1->Name = L"numericUpDown1";
			this->numericUpDown1->Size = System::Drawing::Size(79, 20);
			this->numericUpDown1->TabIndex = 0;
			this->tooltip_gpudetails->SetToolTip(this->numericUpDown1, L"How often to update GPU summary info.\r\nUpdating more quickly uses somewhat more C"
				L"PU time.");
			this->numericUpDown1->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 175, 0, 0, 0 });
			this->numericUpDown1->ValueChanged += gcnew System::EventHandler(this, &GpuSummary::numericUpDown1_ValueChanged);
			// 
			// groupBox1
			// 
			this->groupBox1->Controls->Add(this->numericUpDown1);
			this->groupBox1->Controls->Add(this->label1);
			this->groupBox1->Location = System::Drawing::Point(355, 222);
			this->groupBox1->Name = L"groupBox1";
			this->groupBox1->Size = System::Drawing::Size(134, 57);
			this->groupBox1->TabIndex = 1;
			this->groupBox1->TabStop = false;
			this->groupBox1->Text = L"Stats Update Speed";
			this->tooltip_gpudetails->SetToolTip(this->groupBox1, L"How often to update GPU summary info.\r\nUpdating more quickly uses somewhat more C"
				L"PU time.");
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Location = System::Drawing::Point(99, 27);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(20, 13);
			this->label1->TabIndex = 1;
			this->label1->Text = L"ms";
			// 
			// groupbox_gpudetails
			// 
			this->groupbox_gpudetails->BackColor = System::Drawing::SystemColors::Control;
			this->groupbox_gpudetails->Controls->Add(this->panel_for_smoothtreeview);
			this->groupbox_gpudetails->Location = System::Drawing::Point(12, 12);
			this->groupbox_gpudetails->Name = L"groupbox_gpudetails";
			this->groupbox_gpudetails->Size = System::Drawing::Size(337, 302);
			this->groupbox_gpudetails->TabIndex = 0;
			this->groupbox_gpudetails->TabStop = false;
			this->groupbox_gpudetails->Text = L"Real-Time Info";
			// 
			// panel_for_smoothtreeview
			// 
			this->panel_for_smoothtreeview->BackColor = System::Drawing::SystemColors::Window;
			this->panel_for_smoothtreeview->Location = System::Drawing::Point(6, 19);
			this->panel_for_smoothtreeview->Name = L"panel_for_smoothtreeview";
			this->panel_for_smoothtreeview->Size = System::Drawing::Size(325, 277);
			this->panel_for_smoothtreeview->TabIndex = 0;
			this->panel_for_smoothtreeview->TabStop = true;
			// 
			// textBox1
			// 
			this->textBox1->BorderStyle = System::Windows::Forms::BorderStyle::None;
			this->textBox1->Cursor = System::Windows::Forms::Cursors::Arrow;
			this->textBox1->Location = System::Drawing::Point(355, 31);
			this->textBox1->MaxLength = 512;
			this->textBox1->Multiline = true;
			this->textBox1->Name = L"textBox1";
			this->textBox1->ReadOnly = true;
			this->textBox1->Size = System::Drawing::Size(134, 45);
			this->textBox1->TabIndex = 3;
			this->textBox1->TabStop = false;
			this->textBox1->Text = L"Note: nVidia rates power readings accuracy to ±5% as of FERMI architecture.";
			// 
			// GpuSummary
			// 
			this->AcceptButton = this->button2;
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(501, 320);
			this->Controls->Add(this->textBox1);
			this->Controls->Add(this->groupBox1);
			this->Controls->Add(this->groupbox_gpudetails);
			this->Controls->Add(this->button2);
			this->DoubleBuffered = true;
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::FixedSingle;
			this->HelpButton = true;
			this->Icon = (cli::safe_cast<System::Drawing::Icon^>(resources->GetObject(L"$this.Icon")));
			this->MaximizeBox = false;
			this->Name = L"GpuSummary";
			this->ShowIcon = false;
			this->SizeGripStyle = System::Windows::Forms::SizeGripStyle::Hide;
			this->StartPosition = System::Windows::Forms::FormStartPosition::CenterScreen;
			this->Text = L"COSMiC - CUDA Device Summary";
			this->FormClosing += gcnew System::Windows::Forms::FormClosingEventHandler(this, &GpuSummary::GpuSummary_FormClosing);
			this->Load += gcnew System::EventHandler(this, &GpuSummary::GpuSummary_Load);
			this->Shown += gcnew System::EventHandler(this, &GpuSummary::GpuSummary_Shown);
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown1))->EndInit();
			this->groupBox1->ResumeLayout(false);
			this->groupBox1->PerformLayout();
			this->groupbox_gpudetails->ResumeLayout(false);
			this->ResumeLayout(false);
			this->PerformLayout();
		}
#pragma endregion

private: System::Void button2_Click(System::Object^  sender, System::EventArgs^  e)
{
	// Close button clicked
	this->Close();
}
private: System::Void GpuSummary_Load(System::Object^  sender, System::EventArgs^  e)
{ // GPU Details form opened
	// update the title bar, groupbox name w/ the GPU Index # we're configuring
	if (howManyDevices > 1)
	{
		groupbox_gpudetails->Text = gpusSummarized_MStr + " Summary";
		this->Text = "COSMiC - " + gpusSummarized_MStr + " Summary";  // <---
	}
	else
	{
		groupbox_gpudetails->Text = "CUDA GPU # " + Convert::ToString(gpuDetailsDeviceID) + " Summary";
		this->Text = "COSMiC - CUDA GPU # " + Convert::ToString(gpuDetailsDeviceID) + " Summary";  // <---
	}

	SmoothTreeView^ ctrl = gcnew SmoothTreeView();
	panel_for_smoothtreeview->Controls->Add(ctrl);  // add to pre-placed panel

	this->DBTreeView = ctrl;						// quite important

	// Set up the smooth treeview
	this->DBTreeView->ShowNodeToolTips = true;
	this->DBTreeView->BackColor = System::Drawing::SystemColors::Control;
	this->DBTreeView->BorderStyle = System::Windows::Forms::BorderStyle::None;
	this->DBTreeView->Size = this->panel_for_smoothtreeview->Size;
	this->DBTreeView->Name = L"treeView1";
	this->DBTreeView->Nodes->Add("General");
	if (howManyDevices > 1)
	{ // multiple devices selected (WIP)
		this->DBTreeView->Nodes[0]->Nodes->Add("GPU Name: (multiple devices)");
		this->DBTreeView->Nodes[0]->Nodes->Add("PCIe ID: --");
		this->DBTreeView->Nodes[0]->Nodes->Add("Status: --");
	}
	else
	{ // single device (gpuDetailsDeviceID)
		this->DBTreeView->Nodes[0]->Nodes->Add("GPU Name: " + gcnew String(Cuda_GetDeviceNames(gpuDetailsDeviceID).c_str()));
		this->DBTreeView->Nodes[0]->Nodes->Add("PCIe ID: " + gcnew String(Cuda_GetDeviceBusInfo(gpuDetailsDeviceID).c_str()));
		this->DBTreeView->Nodes[0]->Nodes->Add("Status: ");
	}

	this->DBTreeView->Nodes->Add("Real-Time Performance");
	this->DBTreeView->Nodes[1]->Nodes->Add("GPU Utilization: --");
	//this->DBTreeView->Nodes[1]->Nodes[0]->ToolTipText = "This reading includes utilization by COSMiC Miner and other applications that \n"
		"perform calculations on the GPU. This can include web browsers, messaging clients and Windows itself.";
	
	this->DBTreeView->Nodes[1]->Nodes->Add("Core Clock: --");
	//this->DBTreeView->Nodes[1]->Nodes[1]->ToolTipText = "After the number of CUDA cores, the Core Clock frequency has the greatest impact on "
		"mining performance. Note: COSMiC Miner uses the high-speed memory inside the GPU instead of VRAM, \nso Video Memory clock does "
		"not have a significant impact on hash rate. Underclocking Video Memory can save power and potentially stabilize a higher Core Clock.";
	this->DBTreeView->Nodes[1]->Nodes->Add("Hash Rate: --");
	//this->DBTreeView->Nodes[1]->Nodes[2]->ToolTipText = "Speed at which this GPU is mining, measured in MegaHashes Per Second (MH/s). \n"
		"This number is an average since the last solve (share) and increases in accuracy the longer a solve progresses.";
	this->DBTreeView->Nodes[1]->Nodes->Add("Estimated Efficiency: --");
	//this->DBTreeView->Nodes[1]->Nodes[3]->ToolTipText = "The real-time (approximate) efficiency as computed from the current Hash Rate and device \n"
		"power draw. Depends upon the accuracy of the device's power measurement. \n";
	this->DBTreeView->Nodes[1]->Nodes->Add("Projected Avg. Solve Time: --");
	this->DBTreeView->Nodes[1]->Nodes->Add("Hashes This Solve: --");
	//this->DBTreeView->Nodes[1]->Nodes[4]->ToolTipText = "The computed approximate time required for this GPU to Solve (find a share/solution) with "
		"given its Hash Rate and current Pool Difficulty. Individual solutions may be found faster or slower than this projection.";
	this->DBTreeView->Nodes->Add("Device Sensors");
	this->DBTreeView->Nodes[2]->Nodes->Add("GPU Temperature: --");
	//this->DBTreeView->Nodes[2]->Nodes[0]->ToolTipText = "Note: Make sure your card's power delivery hardware ('VRMs') have adequate cooling! :)";
	this->DBTreeView->Nodes[2]->Nodes->Add("Tachometer Reading: --");
	this->DBTreeView->Nodes[2]->Nodes->Add("Power Usage: --");
	//this->DBTreeView->Nodes[2]->Nodes[2]->ToolTipText = "nVidia specifies a +/-5% accuracy for power measurement on Fermi GPUs.";
 
	this->DBTreeView->Nodes[1]->Nodes[0]->Text = "Utilization: --";
	this->DBTreeView->Nodes[1]->Nodes[1]->Text = "Core Clock: --";
	this->DBTreeView->Nodes[2]->Nodes[0]->Text = "GPU Temperature: --";
	this->DBTreeView->Nodes[2]->Nodes[1]->Text = "Tachometer Reading: --"; // <--- Consider putting a totally different stat here.

	this->DBTreeView->Nodes[0]->Expand();
	this->DBTreeView->Nodes[1]->Expand();
	this->DBTreeView->Nodes[2]->Expand();

	// 
	timer_devicedetails->Interval = (int)numericUpDown1->Value;
	this->timer_devicedetails->Enabled = true;

	// gpuDetailsDeviceID is device #, passed to this form when opened, public
}



//
// [WIP]: optimizing! break this function up if needed.
private: System::Void timer_devicedetails_Tick(System::Object^  sender, System::EventArgs^  e)
{
	if ( this->DBTreeView->GetNodeCount(true) < 1 )  // if treeview not populated yet
		return;

	// if first run ... 
	// (write some stuff only once)
	timer_devicedetails->Stop();  // stop timer for work
	//this->DBTreeView->BeginUpdate();
//	if (gVerbosity == V_DEBUG)  printf("howmany devices: %d \n", howManyDevices);
	if (howManyDevices == 1)
	{ // update device status.	[old]: gCudaDeviceStatus[device_no]
		String^ local_devicestatus = "Status: ";
		if (Solvers[gpuDetailsDeviceID]->solver_status == SolverStatus::Ready) { local_devicestatus += "Ready"; }
		else if (Solvers[gpuDetailsDeviceID]->solver_status == SolverStatus::NotSolving) { local_devicestatus += "Idle"; }
		else if (Solvers[gpuDetailsDeviceID]->solver_status == SolverStatus::Solving) { local_devicestatus += "Mining"; }
		 else if (Solvers[gpuDetailsDeviceID]->solver_status == SolverStatus::DeviceError) { local_devicestatus += "Device Error"; }
		 else if (Solvers[gpuDetailsDeviceID]->solver_status == SolverStatus::Null) { local_devicestatus += "Not Ready"; }
		 else if (Solvers[gpuDetailsDeviceID]->solver_status == SolverStatus::Resuming) { local_devicestatus += "Resuming"; }
		 else if (Solvers[gpuDetailsDeviceID]->solver_status == SolverStatus::UpdatingParams) { local_devicestatus += "Updating Params"; }
		 else if (Solvers[gpuDetailsDeviceID]->solver_status == SolverStatus::WaitingForNetwork) { local_devicestatus += "Waiting"; }
		 else  local_devicestatus += "--";	//unknown
		this->DBTreeView->Nodes[0]->Nodes[2]->Text = local_devicestatus;

		// local vars:
		nvmlUtilization_t theUsage;  // filled in by nvmlDeviceGetUtilizationRates()
		unsigned int theClock = 0;
		theUsage.gpu = 0;
		theUsage.memory = 0;

		this->DBTreeView->Nodes[1]->Nodes[2]->Text = "Hash Rate: " + Math::Round(gNum_Hashrate[gpuDetailsDeviceID], 3, MidpointRounding::ToEven).ToString("0.00") + " MH/s";  // 2 dec. places
		/* [TODO] / [FIXME]: MidpointRounding::ToNegativeInfinity ? for slightly more precise reading of hashrate's last decimal place? */

		//nvml_rslt = nvmlDeviceGetUtilizationRates(gWatchQat_DeviceHandles_NVML[gpuDetailsDeviceID], &theUsage);		  // get utilization for this device
		//NVML_DoCall( nvmlDeviceGetUtilizationRates(gWatchQat_Devices[i].nvml_handle), "getting device utilization", i );  //
		if ( NVML_DoCall( nvmlDeviceGetUtilizationRates(gWatchQat_Devices[gpuDetailsDeviceID].nvml_handle, &theUsage), "getting device utilization", gpuDetailsDeviceID ) == true)
			 DBTreeView->Nodes[1]->Nodes[0]->Text = "Utilization: " + Convert::ToString(theUsage.gpu) + "%";  // write into node's text
		 else  DBTreeView->Nodes[1]->Nodes[0]->Text = "Utilization: --";

		// update GPU clock speed in MHz
		if (NVML_DoCall( nvmlDeviceGetClock(gWatchQat_Devices[gpuDetailsDeviceID].nvml_handle, NVML_CLOCK_GRAPHICS, NVML_CLOCK_ID_CURRENT, &theClock),
			"getting gpu clock", gpuDetailsDeviceID ) == true) {
			// Look into NVAPI equivalent to get clock rate. Possibly remove NVML as a requirement?
			DBTreeView->Nodes[1]->Nodes[1]->Text = "Core Clock: " + theClock.ToString("0,000") + " MHz";
		} else
			DBTreeView->Nodes[1]->Nodes[1]->Text = "Core Clock: --";

		// NOT Error checked call (FIXME):
		const unsigned int local_temp = gWatchQat_Devices[gpuDetailsDeviceID].gputemp;
		DBTreeView->Nodes[2]->Nodes[0]->Text = "GPU Temperature: " + Convert::ToString(local_temp) + "°C";
		DBTreeView->Nodes[2]->Nodes[0]->Text += " / " + Convert::ToString((local_temp * 1.8) + 32) + "°F";

		DBTreeView->Nodes[2]->Nodes[1]->Text = "Tachometer Reading: " + gWatchQat_Devices[gpuDetailsDeviceID].fanspeed_rpm.ToString() + " RPM";
		DBTreeView->Nodes[2]->Nodes[2]->Text = "Board Power Usage: " + gWatchQat_Devices[gpuDetailsDeviceID].powerdraw_w.ToString("0.0") + " W";

		// [wip]:
		double local_efficiency{ 0 };
		const double local_powerdraw = gWatchQat_Devices[gpuDetailsDeviceID].powerdraw_w;
		if (local_powerdraw > 0)  // FIXME: Standardizing double/float, condensing/optimizing
		{
			if (gVerbosity == V_DEBUG)
				std::cout << "[debug]	efficiency calculation:	" << std::to_string(gNum_Hashrate[gpuDetailsDeviceID]) + " / " + std::to_string(local_powerdraw) << std::endl;
			
			local_efficiency = gNum_Hashrate[gpuDetailsDeviceID] / local_powerdraw;
			if (local_efficiency > 0) {
				DBTreeView->Nodes[1]->Nodes[3]->Text = "Estimated Efficiency: ~" +
					Math::Round(local_efficiency, 3, MidpointRounding::ToEven).ToString("0.0") + " MHashes/Sec. / Watt"; }
			 else  DBTreeView->Nodes[1]->Nodes[3]->Text = "Estimated Efficiency: not known";	// not actually displayed <--
		} else
			DBTreeView->Nodes[1]->Nodes[3]->Text = "Estimated Efficiency: not known"; // [new]
	}
	else
	{ // multiple devices selected:
		double dScratch_hr{ 0 };		// total hashrate (double)
		uint64_t scratch64_hc{ 0 };		// total hash count (uint64)
		unsigned short gpuIndex{ 0 };	// cuda gpu # (uint8)
		double dScratch_pwr{ 0 };		// power draw total (double)

		for (unsigned short d = 0; d < CUDA_MAX_DEVICES; ++d)
		{
			if (gpusSummarized[d] == UINT8_MAX) { break; }			 // if empty slot, stop here
			gpuIndex = gpusSummarized[d];
			dScratch_hr += gNum_Hashrate[gpuIndex];					 // add up hashrates (float)
			
			scratch64_hc += gCuda_HashCounts[gpuIndex];				 // add up hash counts (uint64)
//			p_cudaDevices[d]->solver->hash_count;					// [fixme]?

			dScratch_pwr = gWatchQat_Devices[gpuIndex].powerdraw_w;  // add up power draw (double)
			// ... [todo]: keep improving hwmonitor and summary code.
		}
		
		this->DBTreeView->Nodes[1]->Nodes[2]->Text = "Total Hash Rate: " + Math::Round(dScratch_hr, 3, MidpointRounding::ToEven).ToString("0.00") + " MH/s";  // 2 dec. places
		this->DBTreeView->Nodes[1]->Nodes[5]->Text = "Total Hash Count: " + scratch64_hc.ToString("N0");  // comma-separated for readability
		this->DBTreeView->Nodes[2]->Nodes[2]->Text = "GPUs' Total Power Usage: ~" + dScratch_pwr.ToString("0.0");  // FIXME: round? <------
		// ...

		// to here
		double local_efficiency{ 0 };
		if (dScratch_pwr > 0)  // don't div by 0
			local_efficiency = dScratch_hr / dScratch_pwr;
		this->DBTreeView->Nodes[1]->Nodes[3]->Text = "Avg. Estimated Efficiency: " + local_efficiency; // <--- WIP ...
		// ... WIP

	}
	
	//
	if (!gCudaSolving) {
		this->DBTreeView->Nodes[1]->Text = "Real-Time Info  (Not Mining)";
		return;  }
	else  this->DBTreeView->Nodes[1]->Text = "Real-Time Info";
	//
	// why is this here? [moved].
	// single device: show its hashrate
	//if (howManyDevices == 1)
		//this->DBTreeView->Nodes[1]->Nodes[2]->Text = "Hash Rate: " + Math::Round(Cuda_GetDeviceHashrate(gpuDetailsDeviceID), 3, MidpointRounding::ToEven).ToString("0.00") + " MH/s";
	//else {}

	// Allow for projected solution time even for very 'unlikely' solutions (high diff, low hashrate etc.)
	// seconds per solution: 2^22*pool difficulty number / hashrate in hashes/sec (TODO: consider maxtarget override)
	const uint64_t scratch64 = (uint64_t)(gNum_Hashrate[gpuDetailsDeviceID] * 1000000);  // scratch   <-------
	if (scratch64 > 0)  // don't / by 0
	{
		const uint64_t local_difficulty = gU64_DifficultyNo;  // [WIP] <--
		const uint64_t local_secspersoln = (4194304 * local_difficulty) / scratch64;  // calculate projected seconds needed to solve
		TimeSpan local_timespan = TimeSpan::FromSeconds((double)local_secspersoln);

		String^ local_projtimetosoln_str = "Projected Solve Time: ~";
		if (local_timespan.Days)    local_projtimetosoln_str += local_timespan.Days.ToString() + "d ";  // only show days if relevant
		local_projtimetosoln_str += local_timespan.Hours.ToString() + "h ";								// append hours
		local_projtimetosoln_str += local_timespan.Minutes.ToString() + "m ";							// append minutes
		local_projtimetosoln_str += local_timespan.Seconds.ToString() + "s";							// append seconds
		this->DBTreeView->Nodes[1]->Nodes[4]->Text = local_projtimetosoln_str;

		// update the hash count for this device (nonces tested), separated with commas for readability
		this->DBTreeView->Nodes[1]->Nodes[5]->Text = "Hashes This Solve: " + gCuda_HashCounts[gpuDetailsDeviceID].ToString("#,##0");
	}

	//DBTreeView->Nodes[1]->ToolTipText = "nodes no1 tool tip test";
	//DBTreeView->Nodes[1]->Nodes[2]->ToolTipText = "This reading includes other applications and the operating system.\n "
	//	"To maximize hashrate, try to get this number close to 100%.";

	//this->DBTreeView->EndUpdate();
	timer_devicedetails->Start();  // start it again
}

private: System::Void GpuSummary_Shown(System::Object^  sender, System::EventArgs^  e)
{
	// CUDA GPU Details Form now displayed
	//nvmlReturn_t nvml_rslt = NVML_SUCCESS;
	//unsigned int pwrlim_min{ 0 }, pwrlim_max{ 0 };
}

private: System::Void GpuSummary_FormClosing(System::Object^  sender, System::Windows::Forms::FormClosingEventArgs^  e)
{
	// the GPU Details form is closing (Save or Cancel not important)

	timer_devicedetails->Stop();  // stop updating treeview, its form is closing
}

private: System::Void numericUpDown1_ValueChanged(System::Object^  sender, System::EventArgs^  e)
{
	if (numericUpDown1->Value >= 1)  // must be a positive nonzero value
		timer_devicedetails->Interval = (int)numericUpDown1->Value;		// set timer's tick interval
}

};	//class GpuSummary

}	//namespace Cosmic

#endif	//FORM_GPUSUMMARY