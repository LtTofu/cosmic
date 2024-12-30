#pragma once
// COSMICWIND.H: Form for COSMiC's main Mining Window (HUD)
// controls, device status, mining status, menu bar, status bar, etc.
// 2020 LtTofu

#ifndef COSMICWIND_H
#define COSMICWIND_H
#pragma message("Including COSMICWIND, last modified " __TIMESTAMP__ ".")

constexpr auto MIN_NETINTERVAL_POOL = 300;
constexpr auto MIN_NETINTERVAL_SOLO = 100;
constexpr auto MAX_NETINTERVAL_POOL = 2000;	//
constexpr auto MAX_NETINTERVAL_SOLO = 2000;	// [WIP] adjust these as needed <---
constexpr auto /* size_t? */ EVENTTEXT_MAXLENGTH = 130;		// FIXME: use the width of the form? (consider resize)
constexpr unsigned int MAX_URLS_TO_OPEN = 4;

//constexpr int EVENTLINE_SBUILDER_STARTCAPACITY	= 130;	// see: HandleEventsListBox()
constexpr int EVENTLINE_SBUILDER_MAXCAPACITY		= 240;	// 
//#endif

#include "defs.hpp"
extern struct txViewItem gTxViewItems[DEF_TXVIEW_MAX_ITEMS];
extern bool gMiningStopStart;	// defined in CosmicWind.cpp <--- [Kludge]

//void Pause_ms(const unsigned long ms);
//extern std::queue<unsigned int> q_totxview;		// see ` std::mutex mtx_totxview ` in network.cpp

#include "generic_solver.hpp"			//<--
#include "cuda_device.hpp"				//<--
#include "ethereum-rlp/include/RLP.h"
//#include "ethereum-rlp/include/RLP_utils.h"
//#include "ethereum-rlp/include/RLP_test.h"

#include <queue>
#include <msclr/marshal.h>
#include <msclr/marshal_cppstd.h>	// for marshaling native<->managed types
#include <loguru/loguru.hpp>		// loguru (emilk.github.io)

//
#include "Forms/OptionsForm.h"			// General Options & Pool Mode config dialog
#include "Forms/AboutForm.h"			// About COSMiC Dialog
#include "Forms/GpuSummary.h"			// individual GPU statistics, etc
#include "Forms/ConfigIntensityForm.h"	// Set intensity of CUDA devices
#include "Forms/ConfigHwMon.h"			// Configure HW monitor/safety features
#include "Forms/ConfigSoloMining.h"		// Configure Solo Mining dialog
//#include "Forms/EnterPassword.h"		// Simple password input dialog (for Solo Mode)
#include "Forms/TxReceiptForm.h"		// Simple transaction receipt viewer wind

#include "hashburner.cuh"
#include "cpu_solver.h"
#include "network.hpp"				// note: compile "solutions" object without /clr.
#include "net_pool.h"
#include "net_solo.h"
//#include "net_rlp.hpp"
#include "hwmon.h"
#include "util.hpp"

#include "Forms/EnterPassword.h"

using namespace System;
using namespace System::Drawing;
using namespace System::Globalization;
using namespace System::Windows;
using namespace System::Windows::Forms;
using namespace System::Text;
using namespace System::Threading;
using namespace System::Diagnostics;	//<-- Debug Use Only!


//
// === function forward declarations  (TODO: consolidate these w/ Cosmic.cpp, etc.) ===
void ClearEventsQueue(void);					// cosmicwind_native.cpp <-- get rid of
int Cuda_GetNumberOfDevices(void);				// hashburner.cu

GetParmRslt UpdateTxCount(const std::string& for_address, std::string& hex_txcount_out);
int DoMiningSetup(const bool compute_target, const unsigned short maxtarget_exponent, miningParameters* params);  // Cosmic.cpp
//int SpawnSolverThread(const DeviceType device_type, const int device_num);
//void StopMining(uint8_t deviceID);

void CWind_DebugInfo ( miningParameters* params );
void SpawnConsoleOutput();						// "
/*_inline*/ Drawing::Color GetWarningColor(const unsigned short deviceIndex, int scratchTemp);

extern unsigned short gpusSummarized[CUDA_MAX_DEVICES];		// GpuSummary.h
extern bool gDeviceManuallyUnpaused[CUDA_MAX_DEVICES];
extern unsigned int gSoloNetInterval;						// Solo mode Network access interval, in ms
extern bool gNetPause; // hashburner.cu						// ^^ (todo: make sure this is populated at CosmicWind _Shown time) <--

//#include "Configuration.h"

//class genericSolver;	// generic_solver.cpp/.hpp
//class cudaDevice;	// hashburner.cu/cuh


namespace Cosmic {

// === redundant? ===
	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	//using namespace System::Data;
	using namespace System::Drawing;
	using namespace System::Threading;
	using namespace System::Configuration;

	//ref class SmoothListView;

/// <summary>
/// CosmicWind: COSMiC's main mining window
/// </summary>
// ---
	public ref class CosmicWind : System::Windows::Forms::Form
	{ // === Main Form ===
	public:
		CosmicWind(void)
		{
			//AllocateGpuSolvers();

			InitializeComponent();
		}

	protected:
		~CosmicWind()
		{
			LOG_IF_F(INFO, DEBUGMODE, " -- CosmicWind destructor -- ");
			//configuringDevice = UINT8_MAX;
			//memset(configuringDevices, 0xFF, MAX_SOLVERS);
			if (components)
			{
				delete components;
			}
		}

	//move this
	public:
		ref class SmoothListView : ListView
		{
			public:
			SmoothListView(void)
			{ //constructor:
				SetStyle(ControlStyles::OptimizedDoubleBuffer, true);
				SetStyle(ControlStyles::AllPaintingInWmPaint, true);
				SetStyle(ControlStyles::Opaque, true);
			//	SetStyle(ControlStyles::SupportsTransparentBackColor, true);
				UpdateStyles();

				// PASTE:
				//SmoothListView^ newLV = gcnew SmoothListView();  // instantiate the custom listview				[MOVEME] ?
				this->Visible = false;							 // not visible while building

				this->AutoSize = true;							// <-
				//this->BorderStyle = System::Windows::Forms::BorderStyle::None or ::BorderSingle	[REF] <--
				this->Anchor = Forms::AnchorStyles::Top;		// <-
				this->Dock = Forms::DockStyle::Fill;
				// REF: this->BorderStyle = System::Windows::Forms::BorderStyle::None  or  BorderSingle
				this->HeaderStyle = System::Windows::Forms::ColumnHeaderStyle::Nonclickable;
				this->Activation = System::Windows::Forms::ItemActivation::OneClick;
				this->HideSelection = true /* false? */;  // don't show "ghost" when unfocused control
				this->AllowColumnReorder = true;
				this->Scrollable = true;
				this->CheckForIllegalCrossThreadCalls = true;			// 

				//panel_deviceslist->Controls->Add(newLV);				// lower row, only one column (0) of table panel

			//	tablelayoutpanel_top->Controls->Add(newLV, 0, 1); 
				this->View = System::Windows::Forms::View::Details;		// detailed multi-column list in appearance
			//	DevicesView = newLV;

			// (todo)  consider collapsing the top panel above the Splitter if 0 CUDA devices detected (CPU mining only).
			// -- done setting up the DevicesView?	// ^ [MOVED HERE] ^

			}

			~SmoothListView(void) {
				//destructor:
			}

		}; //class SmoothListView

	private: SmoothListView^ DevicesView;	// gpu solvers
	private: SmoothListView^ threadsListView;	// cpu threads (phasing out)

	public: static System::Boolean CosmicWind::CosmicWind_ReadConfiguration(const unsigned short cuda_devicecount);	//<---

	private: System::Diagnostics::Stopwatch^ sw_miningTime;								// new-ish
	private: System::Windows::Forms::ContextMenuStrip^ trayContextMenu;
	private: System::Windows::Forms::ToolStripMenuItem^ cOSMiCVersionToolStripMenuItem; // old, clean up
	private: System::Windows::Forms::ToolStripSeparator^ toolStripSeparator6;
	private: System::Windows::Forms::ToolStripMenuItem^ showToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ quitToolStripMenuItem1;
	private: System::ComponentModel::BackgroundWorker^ bgWorker_SoloView;
	private: System::Windows::Forms::ContextMenuStrip^ txViewContextMenu;
	private: System::Windows::Forms::ToolStripMenuItem^ viewInBlockExplorerToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ clearToolStripMenuItem;
	private: System::Windows::Forms::ToolStripSeparator^ toolStripSeparator5;
	private: System::Windows::Forms::ToolStripMenuItem^ purgeTxViewMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ writePoWToFileToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ viewTxReceiptToolStripMenuItem;
	private: System::Windows::Forms::Timer^ timer_txview;
	private: System::Windows::Forms::ToolStripMenuItem^ enableDisableGpuToolStripMenuItem;
	private: System::Windows::Forms::ToolStripSeparator^ toolStripSeparator3;
	private: System::Windows::Forms::ToolStripSeparator^ toolStripSeparator8;
	private: System::Windows::Forms::ToolStripMenuItem^ gpuNameAndIndexMenuItem;
	private: System::Windows::Forms::Timer^ timer_cputhreadsview;
	private: System::Windows::Forms::PictureBox^ pictureBox1;
	private: System::Windows::Forms::Label^ label6;
	private: System::Windows::Forms::Label^ lbl_totalhashrate;
	private: System::Windows::Forms::Label^ lbl_totalsols;
	private: System::Windows::Forms::Label^ lbl_txncount;
	private: System::Windows::Forms::Label^ lbl_totalminetime;
	private: System::Windows::Forms::Button^ start1;
	private: System::Windows::Forms::Label^ lbl_totalpwr;
	private: System::Windows::Forms::Panel^ lowerPanel;

	private: System::Windows::Forms::GroupBox^ totalsGroupBox;
	private: System::Windows::Forms::ToolStripSeparator^ toolStripSeparator4;
	private: System::Windows::Forms::ToolStripMenuItem^ saveLogToolStripMenuItem;
	private: System::Windows::Forms::ToolStripSeparator^ toolStripSeparator7;
	private: System::Windows::Forms::ToolStripMenuItem^ viewToolStripMenuItem;

	private: System::Windows::Forms::ComboBox^ combobox_modeselect;

	private: System::Windows::Forms::ToolStripMenuItem^ resetHashrateCalcToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ devicesModesToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ nVidiaCUDAGPUsOnlyToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ cUDAGPUsCPUThreadsToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ cPUThreadsOnlyToolStripMenuItem;

	private: System::Windows::Forms::ToolStripMenuItem^ viewToolStripMenuItem1;
	private: System::Windows::Forms::ToolStripMenuItem^ minimizeToTrayToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ autoResizeColumnsToolStripMenuItem1;
	private: System::Windows::Forms::ToolStripMenuItem^ keyboardHotkeysToolStripMenuItem;
	private: System::Windows::Forms::TabControl^ tabControl1;
	private: System::Windows::Forms::TabPage^ eventsPage;
	private: System::Windows::Forms::ListBox^ listbox_events;
	private: System::Windows::Forms::TabPage^ tabSolns;
	private: System::Windows::Forms::ListView^ listview_solutionsview;

	private: System::Windows::Forms::ColumnHeader^ colhead_solnonce;
	private: System::Windows::Forms::ColumnHeader^ colhead_txnstatus;
	private: System::Windows::Forms::ColumnHeader^ colhead_deviceno;
	private: System::Windows::Forms::ColumnHeader^ colhead_challenge;
	private: System::Windows::Forms::ColumnHeader^ colhead_txviewslot;
	private: System::Windows::Forms::ColumnHeader^ colhead_bufslot;
	private: System::Windows::Forms::ColumnHeader^ columnHeader7;
	private: System::Windows::Forms::ColumnHeader^ columnHeader8;
	private: System::Windows::Forms::TabPage^ tabCpu;

	private: System::Windows::Forms::SplitContainer^ splitPanel;
	private: System::Windows::Forms::ToolStripSeparator^ toolStripSeparator10;
	private: System::Windows::Forms::TableLayoutPanel^ tablelayoutpanel_top;

	private: System::Windows::Forms::Panel^ panel_e2gg_r;
	private: System::Windows::Forms::PictureBox^ pictureBox2;
	private: System::Windows::Forms::TableLayoutPanel^ tableLayoutPanel2;
	private: System::Windows::Forms::TextBox^ textbox_challenge;
	private: System::Windows::Forms::Label^ labelChallenge;
	private: System::Windows::Forms::TextBox^ textbox_ethaddress;
	private: System::Windows::Forms::Label^ label1;

	private: System::Windows::Forms::ToolStripMenuItem^ columnsToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ devicesListToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ toolStripMenuItem4;
	private: System::Windows::Forms::ToolStripMenuItem^ toolStripMenuItem5;
	private: System::Windows::Forms::ToolStripMenuItem^ toolStripMenuItem6;
	private: System::Windows::Forms::ToolStripMenuItem^ toolStripMenuItem7;
	private: System::Windows::Forms::ToolStripMenuItem^ toolStripMenuItem1;
	private: System::Windows::Forms::ToolStripMenuItem^ toolStripMenuItem3;
	private: System::Windows::Forms::ToolStripSeparator^ toolStripSeparator9;
	private: System::Windows::Forms::ToolStripMenuItem^ toolStripMenuItem2;
	private: System::Windows::Forms::Label^ lbl_hashbrowns;
	private: System::Windows::Forms::TableLayoutPanel^ tableLayoutPanel3;
	private: System::Windows::Forms::Panel^ panel_e2gg_l;
	private: System::Windows::Forms::PictureBox^ pictureBox3;
	private: System::Windows::Forms::TableLayoutPanel^ tableLayoutPanel1;
	private: System::Windows::Forms::Label^ label3;
	private: System::Windows::Forms::Label^ label2;
	private: System::Windows::Forms::TextBox^ textbox_poolurl;
	private: System::Windows::Forms::TextBox^ textbox_difficulty;
	private: System::Windows::Forms::ToolStripMenuItem^ resubmitTransactionToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ cancelTransactionToolStripMenuItem;
	private: System::Windows::Forms::ToolStripSeparator^ toolStripSeparator11;
	private: System::Windows::Forms::Panel^ panel_deviceslist;
	private: System::Windows::Forms::ToolStripMenuItem^ donateToolStripMenuItem;
	private: System::Windows::Forms::StatusStrip^ statusBar;
	private: System::Windows::Forms::ToolStripStatusLabel^ statusbar_minerstate;
	private: System::Windows::Forms::ToolStripStatusLabel^ statusbar_balanceTokens;
	private: System::Windows::Forms::ToolStripStatusLabel^ statusbar_balanceEth;

	private: System::Windows::Forms::Panel^ panel_threads;
	private: System::Windows::Forms::NumericUpDown^ nud_numthreads;
	private: System::Windows::Forms::Label^ label_threads;
	private: System::Windows::Forms::NumericUpDown^ nud_updatespeed;
	private: System::Windows::Forms::Label^ label_updatespeed;
	private: System::Windows::Forms::Label^ label_uspd_ms;
	private: System::Windows::Forms::Button^ button_cpumine_startstop;
	private: System::Windows::Forms::TextBox^ textbox_cpu_infobox;
	private: System::Windows::Forms::ToolStripStatusLabel^ statusbar_anncNET;
	private: System::Windows::Forms::Panel^ panel2;
	private: System::Windows::Forms::CheckBox^ checkbox_useCPU;
	private: System::Windows::Forms::ToolStripStatusLabel^ statusbar_elapsedTime;
	private: System::Windows::Forms::ToolStripStatusLabel^ statusbar_anncTXN;
	private: System::Windows::Forms::ToolStripMenuItem^ menuitem_options_computetarg;
	
	private: System::Windows::Forms::ToolStripMenuItem^ miscToolStripMenuItem;					// "Misc" menu
	private: System::Windows::Forms::ToolStripMenuItem^ checkMiningRewardToolStripMenuItem;		// dbg

	private: System::Windows::Forms::ToolStripMenuItem^ summaryToolStripMenuItem;		// gpudetails?

//
	public: System::Void CosmicWind::CosmicWind_ResizeStuff(); /* [WIP] Called by the form's _ResizeEnd() event handler */

	private: System::Void CosmicWind::HandleEventsListBox();

// NET annunciator: enabled when NetworkBGWorker is active. text color  red: network error, rust: high latency.
	private: System::Void CosmicWind::Update_NET_Annunciator();

// args: cuda device# and DevicesView subitem #. blanks that info cell with "-"  (.clear()?)
	private: __inline System::Void CosmicWind::ClearDevicesListSubItem(const short row, const short col);

// - MAIN THREAD: -
//void PassSolutionsToSoloTxWorker(void)  // old name

private: System::Void CosmicWind::GetSolutionForTxView();	/* <- rename, condense w/ calling func timer1_Tick() ?    */
		
//
	// detects CUDA devices and populates arrays
	private: unsigned short CosmicWind::Detect_CUDA_Devices();

	private: System::Void CosmicWind::Init_PopulateDevicesList(const unsigned short num_devices);	// "

	private: bool CosmicWind::SetUpDevicesView(void);							// CosmicWind.cpp

	private: unsigned short CosmicWind::Form_DetectDevices(void);				//<----- functional overlap with DetectDevices! [FIXME] <-----
//

//protected:
//public:
private: System::Windows::Forms::Timer^  timer1;

private: System::ComponentModel::BackgroundWorker^ NetworkBGWorker;
//public: WrappedBGWorker^ NetworkBGWorker;

private: System::Windows::Forms::ToolStripMenuItem^  configureGPUToolStripMenuItem;
private: System::Windows::Forms::ToolStripMenuItem^  forceUnpauseToolStripMenuItem;
private: System::Windows::Forms::ToolTip^  toolTip1;
private: System::Windows::Forms::ToolStripMenuItem^  helpToolStripMenuItem;
private: System::Windows::Forms::ToolStripMenuItem^  aboutCOSMiCToolStripMenuItem;

private: System::Windows::Forms::MenuStrip^ menuBar;
private: System::Windows::Forms::ToolStripMenuItem^  fileToolStripMenuItem;
private: System::Windows::Forms::ToolStripMenuItem^  quitToolStripMenuItem;
private: System::Windows::Forms::ToolStripMenuItem^  optionsToolStripMenuItem;
private: System::Windows::Forms::ToolStripMenuItem^  configureToolStripMenuItem;
private: System::Windows::Forms::ToolStripSeparator^  toolStripSeparator1;
private: System::Windows::Forms::ToolStripMenuItem^  advancedToolStripMenuItem;
private: System::Windows::Forms::ToolStripMenuItem^  consoleOutputToolStripMenuItem;
private: System::Windows::Forms::ToolStripMenuItem^  miscCommandsToolStripMenuItem;
private: System::Windows::Forms::ToolStripMenuItem^  deviceCountersToolStripMenuItem;
private: System::Windows::Forms::ToolStripMenuItem^  nVMLStartupToolStripMenuItem;
private: System::Windows::Forms::ToolStripMenuItem^  pauseUnpauseToolStripMenuItem;
private: System::Windows::Forms::ToolStripMenuItem^  newMessageToolStripMenuItem;
private: System::Windows::Forms::ToolStripComboBox^  hUDUpdateSpeedToolStripMenuItem;
private: System::Windows::Forms::NotifyIcon^  notifyIcon1;
private: System::Windows::Forms::Timer^  timer_net_worker_restart;
private: System::Windows::Forms::ContextMenuStrip^  contextMenu_eventsBox;
private: System::Windows::Forms::ToolStripMenuItem^  contextMenu_copySelection;
private: System::Windows::Forms::ToolStripSeparator^  toolStripSeparator2;
private: System::Windows::Forms::ToolStripMenuItem^  contextMenu_clearEvents;
private: System::ComponentModel::BackgroundWorker^  bgworker_swiftmon;	// cpu solvers monitor bgworker
private: System::Windows::Forms::Timer^  timer_resumeafternetfail;
private: System::Windows::Forms::Timer^  timer_enableminingbutton;
private: System::Windows::Forms::Timer^  timer2;
private: System::Windows::Forms::ContextMenuStrip^ contextMenu_gpu;
private: System::Windows::Forms::ToolStripMenuItem^ gpuSummaryMenuItem;  // sort of a header
private: System::Windows::Forms::ToolStripMenuItem^  toolStripMenuItemForceUnpause;
private: System::Windows::Forms::ToolTip^ tooltip_COSMiC;
private: System::Windows::Forms::ToolStripComboBox^  executionStateToolStripMenuItem;
private: System::Windows::Forms::ToolStripMenuItem^  setIntensityToolStripMenuItem;
private: System::Windows::Forms::ToolStripMenuItem^  hWMonitoringAndAlarmsToolStripMenuItem;
private: System::Windows::Forms::ToolStripComboBox^  cUDAEngineToolStripMenuItem;
private: System::Windows::Forms::ToolStripComboBox^  hWMonitoringUpdateSpeedToolStripMenuItem;
private: System::Windows::Forms::ToolTip^  tooltip_NET;
private: System::Windows::Forms::ToolStripMenuItem^  configureSoloMiningToolStripMenuItem;

private: System::Windows::Forms::Timer^  timer_solobalance;

private: System::ComponentModel::IContainer^  components;
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
			System::ComponentModel::ComponentResourceManager^ resources = (gcnew System::ComponentModel::ComponentResourceManager(CosmicWind::typeid));
			this->timer1 = (gcnew System::Windows::Forms::Timer(this->components));
			this->NetworkBGWorker = (gcnew System::ComponentModel::BackgroundWorker());
		//	this->NetworkBGWorker = (gcnew WrappedBGWorker());
			this->configureGPUToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->forceUnpauseToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->toolTip1 = (gcnew System::Windows::Forms::ToolTip(this->components));
			this->pictureBox1 = (gcnew System::Windows::Forms::PictureBox());
			this->lbl_totalminetime = (gcnew System::Windows::Forms::Label());
			this->start1 = (gcnew System::Windows::Forms::Button());
			this->lbl_totalpwr = (gcnew System::Windows::Forms::Label());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->labelChallenge = (gcnew System::Windows::Forms::Label());
			this->label2 = (gcnew System::Windows::Forms::Label());
			this->label3 = (gcnew System::Windows::Forms::Label());
			this->button_cpumine_startstop = (gcnew System::Windows::Forms::Button());
			this->panel_threads = (gcnew System::Windows::Forms::Panel());
			this->panel2 = (gcnew System::Windows::Forms::Panel());
			this->label_threads = (gcnew System::Windows::Forms::Label());
			this->nud_numthreads = (gcnew System::Windows::Forms::NumericUpDown());
			this->textbox_cpu_infobox = (gcnew System::Windows::Forms::TextBox());
			this->label_updatespeed = (gcnew System::Windows::Forms::Label());
			this->nud_updatespeed = (gcnew System::Windows::Forms::NumericUpDown());
			this->label_uspd_ms = (gcnew System::Windows::Forms::Label());
			this->checkbox_useCPU = (gcnew System::Windows::Forms::CheckBox());
			this->contextMenu_eventsBox = (gcnew System::Windows::Forms::ContextMenuStrip(this->components));
			this->contextMenu_copySelection = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->toolStripSeparator2 = (gcnew System::Windows::Forms::ToolStripSeparator());
			this->contextMenu_clearEvents = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->helpToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->aboutCOSMiCToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->menuBar = (gcnew System::Windows::Forms::MenuStrip());
			this->fileToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->saveLogToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->donateToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->toolStripSeparator7 = (gcnew System::Windows::Forms::ToolStripSeparator());
			this->quitToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->optionsToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->configureToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->configureSoloMiningToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->toolStripSeparator1 = (gcnew System::Windows::Forms::ToolStripSeparator());
			this->advancedToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->executionStateToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripComboBox());
			this->hUDUpdateSpeedToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripComboBox());
			this->hWMonitoringUpdateSpeedToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripComboBox());
			this->consoleOutputToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->miscCommandsToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->deviceCountersToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->nVMLStartupToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->pauseUnpauseToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->newMessageToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->resetHashrateCalcToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->menuitem_options_computetarg = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->viewToolStripMenuItem1 = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->minimizeToTrayToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->autoResizeColumnsToolStripMenuItem1 = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->keyboardHotkeysToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->devicesModesToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->nVidiaCUDAGPUsOnlyToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->cUDAGPUsCPUThreadsToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->cPUThreadsOnlyToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->viewToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->columnsToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->devicesListToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->toolStripMenuItem4 = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->toolStripMenuItem5 = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->toolStripMenuItem6 = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->toolStripMenuItem7 = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->toolStripMenuItem1 = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->toolStripMenuItem3 = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->toolStripSeparator9 = (gcnew System::Windows::Forms::ToolStripSeparator());
			this->toolStripMenuItem2 = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->toolStripSeparator10 = (gcnew System::Windows::Forms::ToolStripSeparator());
			this->miscToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->checkMiningRewardToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->notifyIcon1 = (gcnew System::Windows::Forms::NotifyIcon(this->components));
			this->trayContextMenu = (gcnew System::Windows::Forms::ContextMenuStrip(this->components));
			this->cOSMiCVersionToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->toolStripSeparator6 = (gcnew System::Windows::Forms::ToolStripSeparator());
			this->summaryToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->showToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->quitToolStripMenuItem1 = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->timer_net_worker_restart = (gcnew System::Windows::Forms::Timer(this->components));
			this->bgworker_swiftmon = (gcnew System::ComponentModel::BackgroundWorker());
			this->timer_resumeafternetfail = (gcnew System::Windows::Forms::Timer(this->components));
			this->timer_enableminingbutton = (gcnew System::Windows::Forms::Timer(this->components));
			this->timer2 = (gcnew System::Windows::Forms::Timer(this->components));
			this->txViewContextMenu = (gcnew System::Windows::Forms::ContextMenuStrip(this->components));
			this->viewInBlockExplorerToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->clearToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->toolStripSeparator11 = (gcnew System::Windows::Forms::ToolStripSeparator());
			this->resubmitTransactionToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->cancelTransactionToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->viewTxReceiptToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->toolStripSeparator5 = (gcnew System::Windows::Forms::ToolStripSeparator());
			this->purgeTxViewMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->writePoWToFileToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->contextMenu_gpu = (gcnew System::Windows::Forms::ContextMenuStrip(this->components));
			this->gpuNameAndIndexMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->toolStripSeparator4 = (gcnew System::Windows::Forms::ToolStripSeparator());
			this->cUDAEngineToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripComboBox());
			this->toolStripSeparator8 = (gcnew System::Windows::Forms::ToolStripSeparator());
			this->gpuSummaryMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->setIntensityToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->hWMonitoringAndAlarmsToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->toolStripSeparator3 = (gcnew System::Windows::Forms::ToolStripSeparator());
			this->toolStripMenuItemForceUnpause = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->enableDisableGpuToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->tooltip_COSMiC = (gcnew System::Windows::Forms::ToolTip(this->components));
			this->tooltip_NET = (gcnew System::Windows::Forms::ToolTip(this->components));
			this->timer_solobalance = (gcnew System::Windows::Forms::Timer(this->components));
			this->bgWorker_SoloView = (gcnew System::ComponentModel::BackgroundWorker());
			this->timer_txview = (gcnew System::Windows::Forms::Timer(this->components));
			this->timer_cputhreadsview = (gcnew System::Windows::Forms::Timer(this->components));
			this->label6 = (gcnew System::Windows::Forms::Label());
			this->lbl_totalhashrate = (gcnew System::Windows::Forms::Label());
			this->lbl_totalsols = (gcnew System::Windows::Forms::Label());
			this->lbl_txncount = (gcnew System::Windows::Forms::Label());
			this->lowerPanel = (gcnew System::Windows::Forms::Panel());
			this->statusBar = (gcnew System::Windows::Forms::StatusStrip());
			this->statusbar_minerstate = (gcnew System::Windows::Forms::ToolStripStatusLabel());
			this->statusbar_anncTXN = (gcnew System::Windows::Forms::ToolStripStatusLabel());
			this->statusbar_anncNET = (gcnew System::Windows::Forms::ToolStripStatusLabel());
			this->statusbar_balanceEth = (gcnew System::Windows::Forms::ToolStripStatusLabel());
			this->statusbar_balanceTokens = (gcnew System::Windows::Forms::ToolStripStatusLabel());
			this->statusbar_elapsedTime = (gcnew System::Windows::Forms::ToolStripStatusLabel());
			this->lbl_hashbrowns = (gcnew System::Windows::Forms::Label());
			this->totalsGroupBox = (gcnew System::Windows::Forms::GroupBox());
			this->combobox_modeselect = (gcnew System::Windows::Forms::ComboBox());
			this->tabControl1 = (gcnew System::Windows::Forms::TabControl());
			this->eventsPage = (gcnew System::Windows::Forms::TabPage());
			this->listbox_events = (gcnew System::Windows::Forms::ListBox());
			this->tabSolns = (gcnew System::Windows::Forms::TabPage());
			this->listview_solutionsview = (gcnew System::Windows::Forms::ListView());
			this->colhead_solnonce = (gcnew System::Windows::Forms::ColumnHeader());
			this->colhead_txnstatus = (gcnew System::Windows::Forms::ColumnHeader());
			this->colhead_deviceno = (gcnew System::Windows::Forms::ColumnHeader());
			this->colhead_challenge = (gcnew System::Windows::Forms::ColumnHeader());
			this->colhead_txviewslot = (gcnew System::Windows::Forms::ColumnHeader());
			this->colhead_bufslot = (gcnew System::Windows::Forms::ColumnHeader());
			this->columnHeader7 = (gcnew System::Windows::Forms::ColumnHeader());
			this->columnHeader8 = (gcnew System::Windows::Forms::ColumnHeader());
			this->tabCpu = (gcnew System::Windows::Forms::TabPage());
			this->splitPanel = (gcnew System::Windows::Forms::SplitContainer());
			this->tablelayoutpanel_top = (gcnew System::Windows::Forms::TableLayoutPanel());
			this->tableLayoutPanel3 = (gcnew System::Windows::Forms::TableLayoutPanel());
			this->panel_e2gg_l = (gcnew System::Windows::Forms::Panel());
			this->pictureBox3 = (gcnew System::Windows::Forms::PictureBox());
			this->tableLayoutPanel1 = (gcnew System::Windows::Forms::TableLayoutPanel());
			this->textbox_poolurl = (gcnew System::Windows::Forms::TextBox());
			this->textbox_difficulty = (gcnew System::Windows::Forms::TextBox());
			this->panel_e2gg_r = (gcnew System::Windows::Forms::Panel());
			this->pictureBox2 = (gcnew System::Windows::Forms::PictureBox());
			this->tableLayoutPanel2 = (gcnew System::Windows::Forms::TableLayoutPanel());
			this->textbox_challenge = (gcnew System::Windows::Forms::TextBox());
			this->textbox_ethaddress = (gcnew System::Windows::Forms::TextBox());
			this->panel_deviceslist = (gcnew System::Windows::Forms::Panel());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->BeginInit();
			this->panel2->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->nud_numthreads))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->nud_updatespeed))->BeginInit();
			this->contextMenu_eventsBox->SuspendLayout();
			this->menuBar->SuspendLayout();
			this->trayContextMenu->SuspendLayout();
			this->txViewContextMenu->SuspendLayout();
			this->contextMenu_gpu->SuspendLayout();
			this->lowerPanel->SuspendLayout();
			this->statusBar->SuspendLayout();
			this->totalsGroupBox->SuspendLayout();
			this->tabControl1->SuspendLayout();
			this->eventsPage->SuspendLayout();
			this->tabSolns->SuspendLayout();
			this->tabCpu->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->splitPanel))->BeginInit();
			this->splitPanel->Panel1->SuspendLayout();
			this->splitPanel->Panel2->SuspendLayout();
			this->splitPanel->SuspendLayout();
			this->tablelayoutpanel_top->SuspendLayout();
			this->tableLayoutPanel3->SuspendLayout();
			this->panel_e2gg_l->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox3))->BeginInit();
			this->tableLayoutPanel1->SuspendLayout();
			this->panel_e2gg_r->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox2))->BeginInit();
			this->tableLayoutPanel2->SuspendLayout();
			this->SuspendLayout();
			// 
			// timer1
			// 
			this->timer1->Enabled = true;
			this->timer1->Interval = 150;
			this->timer1->Tick += gcnew System::EventHandler(this, &CosmicWind::timer1_Tick);
			// 
			// NetworkBGWorker
			// 
			this->NetworkBGWorker->WorkerReportsProgress = true;
			this->NetworkBGWorker->WorkerSupportsCancellation = true;
			this->NetworkBGWorker->DoWork += gcnew System::ComponentModel::DoWorkEventHandler(this, &CosmicWind::NetworkBGWorker_DoWork);
			this->NetworkBGWorker->RunWorkerCompleted += gcnew System::ComponentModel::RunWorkerCompletedEventHandler(this, &CosmicWind::NetworkBGWorker_RunWorkerCompleted);
			// 
			// configureGPUToolStripMenuItem
			// 
			this->configureGPUToolStripMenuItem->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"configureGPUToolStripMenuItem.Image")));
			this->configureGPUToolStripMenuItem->Name = L"configureGPUToolStripMenuItem";
			this->configureGPUToolStripMenuItem->Size = System::Drawing::Size(187, 22);
			this->configureGPUToolStripMenuItem->Text = L"Configure This GPU...";
			this->configureGPUToolStripMenuItem->Click += gcnew System::EventHandler(this, &CosmicWind::configureGPUToolStripMenuItem_Click);
			// 
			// forceUnpauseToolStripMenuItem
			// 
			this->forceUnpauseToolStripMenuItem->Name = L"forceUnpauseToolStripMenuItem";
			this->forceUnpauseToolStripMenuItem->Size = System::Drawing::Size(187, 22);
			this->forceUnpauseToolStripMenuItem->Text = L"Force Unpause";
			this->forceUnpauseToolStripMenuItem->Click += gcnew System::EventHandler(this, &CosmicWind::forceUnpauseToolStripMenuItem_Click);
			// 
			// toolTip1
			// 
			this->toolTip1->AutomaticDelay = 0;
			this->toolTip1->AutoPopDelay = 32000;
			this->toolTip1->BackColor = System::Drawing::SystemColors::ActiveCaption;
			this->toolTip1->InitialDelay = 350;
			this->toolTip1->ReshowDelay = 100;
			this->toolTip1->ToolTipIcon = System::Windows::Forms::ToolTipIcon::Info;
			this->toolTip1->ToolTipTitle = L"Info";
			this->toolTip1->UseAnimation = false;
			this->toolTip1->UseFading = false;
			// 
			// pictureBox1
			// 
			this->pictureBox1->Anchor = System::Windows::Forms::AnchorStyles::Left;
			this->pictureBox1->BackColor = System::Drawing::SystemColors::ButtonFace;
			this->pictureBox1->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"pictureBox1.Image")));
			this->pictureBox1->Location = System::Drawing::Point(0, 8);
			this->pictureBox1->Name = L"pictureBox1";
			this->pictureBox1->Padding = System::Windows::Forms::Padding(8, 0, 0, 0);
			this->pictureBox1->Size = System::Drawing::Size(156, 24);
			this->pictureBox1->TabIndex = 2;
			this->pictureBox1->TabStop = false;
			this->toolTip1->SetToolTip(this->pictureBox1, L"Thank You for using COSMiC!");
			// 
			// lbl_totalminetime
			// 
			this->lbl_totalminetime->Anchor = System::Windows::Forms::AnchorStyles::None;
			this->lbl_totalminetime->AutoSize = true;
			this->lbl_totalminetime->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Regular,
				System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
			this->lbl_totalminetime->Location = System::Drawing::Point(479, 35);
			this->lbl_totalminetime->Name = L"lbl_totalminetime";
			this->lbl_totalminetime->Size = System::Drawing::Size(13, 13);
			this->lbl_totalminetime->TabIndex = 6;
			this->lbl_totalminetime->Text = L"--";
			this->toolTip1->SetToolTip(this->lbl_totalminetime, L"The total time mining this session. Does not reset\r\non each Share/Solution find.");
			this->lbl_totalminetime->Visible = false;
			// 
			// start1
			// 
			this->start1->Anchor = System::Windows::Forms::AnchorStyles::Right;
			this->start1->BackColor = System::Drawing::Color::White;
			this->start1->FlatAppearance->BorderColor = System::Drawing::Color::Black;
			this->start1->FlatAppearance->MouseDownBackColor = System::Drawing::Color::White;
			this->start1->FlatAppearance->MouseOverBackColor = System::Drawing::Color::White;
			this->start1->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			this->start1->Font = (gcnew System::Drawing::Font(L"Segoe UI Semibold", 9, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->start1->ImageAlign = System::Drawing::ContentAlignment::MiddleLeft;
			this->start1->ImageKey = L"(none)";
			this->start1->Location = System::Drawing::Point(192, 16);
			this->start1->Margin = System::Windows::Forms::Padding(0);
			this->start1->Name = L"start1";
			this->start1->Size = System::Drawing::Size(132, 26);
			this->start1->TabIndex = 7;
			this->start1->Text = L"Start Mining!";
			this->start1->TextAlign = System::Drawing::ContentAlignment::TopCenter;
			this->toolTip1->SetToolTip(this->start1, L"Start / Stop mining. Uses the CUDA devices that are enabled.\r\nBe sure to configur"
				L"e your settings first (see the Options menu.)\r\n\r\n(Right-click a card in the list"
				L" to change its settings.)");
			this->start1->UseVisualStyleBackColor = false;
			this->start1->Click += gcnew System::EventHandler(this, &CosmicWind::start1_Click);
			// 
			// lbl_totalpwr
			// 
			this->lbl_totalpwr->AutoSize = true;
			this->lbl_totalpwr->Location = System::Drawing::Point(15, 52);
			this->lbl_totalpwr->Name = L"lbl_totalpwr";
			this->lbl_totalpwr->Size = System::Drawing::Size(27, 13);
			this->lbl_totalpwr->TabIndex = 4;
			this->lbl_totalpwr->Text = L"-- W";
			this->toolTip1->SetToolTip(this->lbl_totalpwr, L"Estimated total power usage of GPUs on which\r\nCOSMiC is active (in Watts) and the"
				L" power usage\r\nis known. nVidia rates power usage accuracy to\r\n+/- 5% (as of Ferm"
				L"i architecture).");
			// 
			// label1
			// 
			this->label1->Anchor = System::Windows::Forms::AnchorStyles::Left;
			this->label1->BackColor = System::Drawing::SystemColors::Window;
			this->label1->Font = (gcnew System::Drawing::Font(L"Segoe UI Semibold", 9.75F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->label1->Location = System::Drawing::Point(3, 0);
			this->label1->Margin = System::Windows::Forms::Padding(3, 0, 3, 2);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(87, 17);
			this->label1->TabIndex = 0;
			this->label1->Text = L"Eth. Address:";
			this->toolTip1->SetToolTip(this->label1, resources->GetString(L"label1.ToolTip"));
			// 
			// labelChallenge
			// 
			this->labelChallenge->Anchor = System::Windows::Forms::AnchorStyles::Left;
			this->labelChallenge->BackColor = System::Drawing::SystemColors::Window;
			this->labelChallenge->Font = (gcnew System::Drawing::Font(L"Segoe UI Semibold", 9.75F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->labelChallenge->Location = System::Drawing::Point(3, 19);
			this->labelChallenge->Margin = System::Windows::Forms::Padding(3, 0, 3, 2);
			this->labelChallenge->Name = L"labelChallenge";
			this->labelChallenge->Size = System::Drawing::Size(70, 17);
			this->labelChallenge->TabIndex = 2;
			this->labelChallenge->Text = L"Challenge:";
			this->toolTip1->SetToolTip(this->labelChallenge, L"The current network-wide Challenge. This changes whenever\r\na miner successfully m"
				L"ints a Solution at the Contract difficulty.\r\n\r\n");
			// 
			// label2
			// 
			this->label2->Anchor = System::Windows::Forms::AnchorStyles::Left;
			this->label2->Font = (gcnew System::Drawing::Font(L"Segoe UI Semibold", 9.75F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->label2->Location = System::Drawing::Point(3, 0);
			this->label2->Margin = System::Windows::Forms::Padding(3, 0, 3, 2);
			this->label2->Name = L"label2";
			this->label2->Size = System::Drawing::Size(71, 17);
			this->label2->TabIndex = 0;
			this->label2->Text = L"Mining To:";
			this->toolTip1->SetToolTip(this->label2, L"The currently-selected Pool. Check this Pool address,\r\nwithout the port, in a Web"
				L" Browser for information\r\nabout your shares/payouts.");
			// 
			// label3
			// 
			this->label3->Anchor = System::Windows::Forms::AnchorStyles::Left;
			this->label3->Font = (gcnew System::Drawing::Font(L"Segoe UI Semibold", 9.75F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->label3->Location = System::Drawing::Point(3, 19);
			this->label3->Margin = System::Windows::Forms::Padding(3, 0, 3, 2);
			this->label3->Name = L"label3";
			this->label3->Size = System::Drawing::Size(63, 17);
			this->label3->TabIndex = 2;
			this->label3->Text = L"Difficulty:";
			this->toolTip1->SetToolTip(this->label3, L"Difficulty set by the pool- can be Variable Difficulty (VARDIFF), fixed or\r\ncusto"
				L"mized on a per-miner basis. Contact your Pool op for more info.");
			// 
			// button_cpumine_startstop
			// 
			this->button_cpumine_startstop->Location = System::Drawing::Point(158, 91);
			this->button_cpumine_startstop->Name = L"button_cpumine_startstop";
			this->button_cpumine_startstop->Size = System::Drawing::Size(75, 23);
			this->button_cpumine_startstop->TabIndex = 25;
			this->button_cpumine_startstop->Text = L"Start / Stop";
			this->toolTip1->SetToolTip(this->button_cpumine_startstop, L"Start or Stop mining on the CPU.");
			this->button_cpumine_startstop->UseVisualStyleBackColor = true;
			this->button_cpumine_startstop->Click += gcnew System::EventHandler(this, &CosmicWind::button_cpumine_startstop_Click);
			// 
			// panel_threads
			// 
			this->panel_threads->AutoSizeMode = System::Windows::Forms::AutoSizeMode::GrowAndShrink;
			this->panel_threads->Enabled = false;
			this->panel_threads->Location = System::Drawing::Point(247, 3);
			this->panel_threads->Name = L"panel_threads";
			this->panel_threads->Size = System::Drawing::Size(555, 118);
			this->panel_threads->TabIndex = 29;
			this->toolTip1->SetToolTip(this->panel_threads, L"CPU threads used to mine are listed here");
			// 
			// panel2
			// 
			this->panel2->Controls->Add(this->label_threads);
			this->panel2->Controls->Add(this->nud_numthreads);
			this->panel2->Controls->Add(this->button_cpumine_startstop);
			this->panel2->Controls->Add(this->textbox_cpu_infobox);
			this->panel2->Controls->Add(this->label_updatespeed);
			this->panel2->Controls->Add(this->nud_updatespeed);
			this->panel2->Controls->Add(this->label_uspd_ms);
			this->panel2->Controls->Add(this->checkbox_useCPU);
			this->panel2->Enabled = false;
			this->panel2->Location = System::Drawing::Point(0, 0);
			this->panel2->Name = L"panel2";
			this->panel2->Size = System::Drawing::Size(241, 121);
			this->panel2->TabIndex = 29;
			// 
			// label_threads
			// 
			this->label_threads->AutoSize = true;
			this->label_threads->Enabled = false;
			this->label_threads->Location = System::Drawing::Point(12, 36);
			this->label_threads->Name = L"label_threads";
			this->label_threads->Size = System::Drawing::Size(55, 15);
			this->label_threads->TabIndex = 21;
			this->label_threads->Text = L"Threads:";
			// 
			// nud_numthreads
			// 
			this->nud_numthreads->Location = System::Drawing::Point(133, 34);
			this->nud_numthreads->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 512, 0, 0, 0 });
			this->nud_numthreads->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
			this->nud_numthreads->Name = L"nud_numthreads";
			this->nud_numthreads->Size = System::Drawing::Size(74, 21);
			this->nud_numthreads->TabIndex = 22;
			this->nud_numthreads->ThousandsSeparator = true;
			this->nud_numthreads->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
			// 
			// textbox_cpu_infobox
			// 
			this->textbox_cpu_infobox->BackColor = System::Drawing::SystemColors::ButtonFace;
			this->textbox_cpu_infobox->Location = System::Drawing::Point(8, 92);
			this->textbox_cpu_infobox->Multiline = true;
			this->textbox_cpu_infobox->Name = L"textbox_cpu_infobox";
			this->textbox_cpu_infobox->ReadOnly = true;
			this->textbox_cpu_infobox->ShortcutsEnabled = false;
			this->textbox_cpu_infobox->Size = System::Drawing::Size(144, 22);
			this->textbox_cpu_infobox->TabIndex = 23;
			// 
			// label_updatespeed
			// 
			this->label_updatespeed->AutoSize = true;
			this->label_updatespeed->Enabled = false;
			this->label_updatespeed->Location = System::Drawing::Point(12, 61);
			this->label_updatespeed->Name = L"label_updatespeed";
			this->label_updatespeed->Size = System::Drawing::Size(119, 15);
			this->label_updatespeed->TabIndex = 26;
			this->label_updatespeed->Text = L"Stats Update Speed:";
			// 
			// nud_updatespeed
			// 
			this->nud_updatespeed->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 10, 0, 0, 0 });
			this->nud_updatespeed->Location = System::Drawing::Point(133, 59);
			this->nud_updatespeed->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 5000, 0, 0, 0 });
			this->nud_updatespeed->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 20, 0, 0, 0 });
			this->nud_updatespeed->Name = L"nud_updatespeed";
			this->nud_updatespeed->Size = System::Drawing::Size(74, 21);
			this->nud_updatespeed->TabIndex = 24;
			this->nud_updatespeed->ThousandsSeparator = true;
			this->nud_updatespeed->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 175, 0, 0, 0 });
			// 
			// label_uspd_ms
			// 
			this->label_uspd_ms->AutoSize = true;
			this->label_uspd_ms->Enabled = false;
			this->label_uspd_ms->Location = System::Drawing::Point(209, 61);
			this->label_uspd_ms->Name = L"label_uspd_ms";
			this->label_uspd_ms->Size = System::Drawing::Size(24, 15);
			this->label_uspd_ms->TabIndex = 27;
			this->label_uspd_ms->Text = L"ms";
			// 
			// checkbox_useCPU
			// 
			this->checkbox_useCPU->Anchor = System::Windows::Forms::AnchorStyles::None;
			this->checkbox_useCPU->Checked = true;
			this->checkbox_useCPU->CheckState = System::Windows::Forms::CheckState::Checked;
			this->checkbox_useCPU->ImageAlign = System::Drawing::ContentAlignment::MiddleLeft;
			this->checkbox_useCPU->Location = System::Drawing::Point(4, 9);
			this->checkbox_useCPU->Name = L"checkbox_useCPU";
			this->checkbox_useCPU->Padding = System::Windows::Forms::Padding(5, 0, 0, 0);
			this->checkbox_useCPU->Size = System::Drawing::Size(234, 19);
			this->checkbox_useCPU->TabIndex = 30;
			this->checkbox_useCPU->Text = L"Mine with CPU   (experimental)";
			this->checkbox_useCPU->UseVisualStyleBackColor = true;
			// 
			// contextMenu_eventsBox
			// 
			this->contextMenu_eventsBox->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(3) {
				this->contextMenu_copySelection,
					this->toolStripSeparator2, this->contextMenu_clearEvents
			});
			this->contextMenu_eventsBox->Name = L"contextMenu_eventsBox";
			this->contextMenu_eventsBox->Size = System::Drawing::Size(187, 54);
			this->contextMenu_eventsBox->Text = L"Selected Event(s)";
			this->contextMenu_eventsBox->Closing += gcnew System::Windows::Forms::ToolStripDropDownClosingEventHandler(this, &CosmicWind::contextMenu_eventsBox_Closing);
			this->contextMenu_eventsBox->Opening += gcnew System::ComponentModel::CancelEventHandler(this, &CosmicWind::contextMenu_eventsBox_Opening);
			// 
			// contextMenu_copySelection
			// 
			this->contextMenu_copySelection->Enabled = false;
			this->contextMenu_copySelection->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"contextMenu_copySelection.Image")));
			this->contextMenu_copySelection->Name = L"contextMenu_copySelection";
			this->contextMenu_copySelection->Size = System::Drawing::Size(186, 22);
			this->contextMenu_copySelection->Text = L"Copy Selected Events";
			this->contextMenu_copySelection->ToolTipText = L"Temporarily disabled for debugging.";
			this->contextMenu_copySelection->Click += gcnew System::EventHandler(this, &CosmicWind::copyEventsToolStripMenuItem_Click);
			// 
			// toolStripSeparator2
			// 
			this->toolStripSeparator2->Name = L"toolStripSeparator2";
			this->toolStripSeparator2->Size = System::Drawing::Size(183, 6);
			// 
			// contextMenu_clearEvents
			// 
			this->contextMenu_clearEvents->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"contextMenu_clearEvents.Image")));
			this->contextMenu_clearEvents->Name = L"contextMenu_clearEvents";
			this->contextMenu_clearEvents->Size = System::Drawing::Size(186, 22);
			this->contextMenu_clearEvents->Text = L"Clear Events List";
			this->contextMenu_clearEvents->Click += gcnew System::EventHandler(this, &CosmicWind::contextMenu_clearEvents_Click);
			// 
			// helpToolStripMenuItem
			// 
			this->helpToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(1) { this->aboutCOSMiCToolStripMenuItem });
			this->helpToolStripMenuItem->Name = L"helpToolStripMenuItem";
			this->helpToolStripMenuItem->Size = System::Drawing::Size(45, 20);
			this->helpToolStripMenuItem->Text = L"&Help";
			this->helpToolStripMenuItem->Click += gcnew System::EventHandler(this, &CosmicWind::helpToolStripMenuItem_Click);
			// 
			// aboutCOSMiCToolStripMenuItem
			// 
			this->aboutCOSMiCToolStripMenuItem->Font = (gcnew System::Drawing::Font(L"Segoe UI", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->aboutCOSMiCToolStripMenuItem->ForeColor = System::Drawing::Color::Black;
			this->aboutCOSMiCToolStripMenuItem->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"aboutCOSMiCToolStripMenuItem.Image")));
			this->aboutCOSMiCToolStripMenuItem->Name = L"aboutCOSMiCToolStripMenuItem";
			this->aboutCOSMiCToolStripMenuItem->ShortcutKeys = System::Windows::Forms::Keys::F1;
			this->aboutCOSMiCToolStripMenuItem->Size = System::Drawing::Size(174, 22);
			this->aboutCOSMiCToolStripMenuItem->Text = L"&About / Help ...";
			this->aboutCOSMiCToolStripMenuItem->Click += gcnew System::EventHandler(this, &CosmicWind::aboutCOSMiCToolStripMenuItem_Click);
			// 
			// menuBar
			// 
			this->menuBar->AccessibleRole = System::Windows::Forms::AccessibleRole::MenuBar;
			this->menuBar->AllowItemReorder = true;
			this->menuBar->AllowMerge = false;
			this->menuBar->BackColor = System::Drawing::SystemColors::Window;
			this->menuBar->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->menuBar->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(5) {
				this->fileToolStripMenuItem,
					this->optionsToolStripMenuItem, this->viewToolStripMenuItem, this->miscToolStripMenuItem, this->helpToolStripMenuItem
			});
			this->menuBar->Location = System::Drawing::Point(0, 0);
			this->menuBar->Name = L"menuBar";
			this->menuBar->RightToLeft = System::Windows::Forms::RightToLeft::No;
			this->menuBar->ShowItemToolTips = true;
			this->menuBar->Size = System::Drawing::Size(880, 24);
			this->menuBar->TabIndex = 0;
			this->menuBar->TabStop = true;
			this->menuBar->Text = L"menuBar";
			// 
			// fileToolStripMenuItem
			// 
			this->fileToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(4) {
				this->saveLogToolStripMenuItem,
					this->donateToolStripMenuItem, this->toolStripSeparator7, this->quitToolStripMenuItem
			});
			this->fileToolStripMenuItem->Name = L"fileToolStripMenuItem";
			this->fileToolStripMenuItem->Size = System::Drawing::Size(39, 20);
			this->fileToolStripMenuItem->Text = L"&File";
			// 
			// saveLogToolStripMenuItem
			// 
			this->saveLogToolStripMenuItem->CheckOnClick = true;
			this->saveLogToolStripMenuItem->Name = L"saveLogToolStripMenuItem";
			this->saveLogToolStripMenuItem->Size = System::Drawing::Size(137, 22);
			this->saveLogToolStripMenuItem->Text = L"&Save Log";
			this->saveLogToolStripMenuItem->Visible = false;
			// 
			// donateToolStripMenuItem
			// 
			this->donateToolStripMenuItem->Name = L"donateToolStripMenuItem";
			this->donateToolStripMenuItem->Size = System::Drawing::Size(137, 22);
			this->donateToolStripMenuItem->Text = L"Donate...";
			this->donateToolStripMenuItem->Visible = false;
			// 
			// toolStripSeparator7
			// 
			this->toolStripSeparator7->Name = L"toolStripSeparator7";
			this->toolStripSeparator7->Size = System::Drawing::Size(134, 6);
			this->toolStripSeparator7->Visible = false;
			// 
			// quitToolStripMenuItem
			// 
			this->quitToolStripMenuItem->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"quitToolStripMenuItem.Image")));
			this->quitToolStripMenuItem->Name = L"quitToolStripMenuItem";
			this->quitToolStripMenuItem->ShortcutKeys = static_cast<System::Windows::Forms::Keys>((System::Windows::Forms::Keys::Control | System::Windows::Forms::Keys::Q));
			this->quitToolStripMenuItem->Size = System::Drawing::Size(137, 22);
			this->quitToolStripMenuItem->Text = L"&Quit";
			this->quitToolStripMenuItem->Click += gcnew System::EventHandler(this, &CosmicWind::quitToolStripMenuItem_Click);
			// 
			// optionsToolStripMenuItem
			// 
			this->optionsToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(6) {
				this->configureToolStripMenuItem,
					this->configureSoloMiningToolStripMenuItem, this->toolStripSeparator1, this->advancedToolStripMenuItem, this->viewToolStripMenuItem1,
					this->devicesModesToolStripMenuItem
			});
			this->optionsToolStripMenuItem->Name = L"optionsToolStripMenuItem";
			this->optionsToolStripMenuItem->Size = System::Drawing::Size(61, 20);
			this->optionsToolStripMenuItem->Text = L"&Options";
			// 
			// configureToolStripMenuItem
			// 
			this->configureToolStripMenuItem->Font = (gcnew System::Drawing::Font(L"Segoe UI", 9, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->configureToolStripMenuItem->ForeColor = System::Drawing::SystemColors::HotTrack;
			this->configureToolStripMenuItem->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"configureToolStripMenuItem.Image")));
			this->configureToolStripMenuItem->Name = L"configureToolStripMenuItem";
			this->configureToolStripMenuItem->RightToLeftAutoMirrorImage = true;
			this->configureToolStripMenuItem->ShortcutKeyDisplayString = L"";
			this->configureToolStripMenuItem->ShortcutKeys = static_cast<System::Windows::Forms::Keys>((System::Windows::Forms::Keys::Control | System::Windows::Forms::Keys::O));
			this->configureToolStripMenuItem->Size = System::Drawing::Size(207, 22);
			this->configureToolStripMenuItem->Text = L"&General Setup...";
			this->configureToolStripMenuItem->ToolTipText = L"Configure Pool address, your Ethereum mining address\r\nand other general options.";
			this->configureToolStripMenuItem->Click += gcnew System::EventHandler(this, &CosmicWind::configureToolStripMenuItem_Click);
			// 
			// configureSoloMiningToolStripMenuItem
			// 
			this->configureSoloMiningToolStripMenuItem->Font = (gcnew System::Drawing::Font(L"Segoe UI Semibold", 9, System::Drawing::FontStyle::Bold,
				System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
			this->configureSoloMiningToolStripMenuItem->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"configureSoloMiningToolStripMenuItem.Image")));
			this->configureSoloMiningToolStripMenuItem->Name = L"configureSoloMiningToolStripMenuItem";
			this->configureSoloMiningToolStripMenuItem->Size = System::Drawing::Size(207, 22);
			this->configureSoloMiningToolStripMenuItem->Text = L"&Solo Mining Setup...";
			this->configureSoloMiningToolStripMenuItem->Click += gcnew System::EventHandler(this, &CosmicWind::configureSoloMiningToolStripMenuItem_Click);
			// 
			// toolStripSeparator1
			// 
			this->toolStripSeparator1->Name = L"toolStripSeparator1";
			this->toolStripSeparator1->Size = System::Drawing::Size(204, 6);
			// 
			// advancedToolStripMenuItem
			// 
			this->advancedToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(6) {
				this->executionStateToolStripMenuItem,
					this->hUDUpdateSpeedToolStripMenuItem, this->hWMonitoringUpdateSpeedToolStripMenuItem, this->consoleOutputToolStripMenuItem,
					this->miscCommandsToolStripMenuItem, this->menuitem_options_computetarg
			});
			this->advancedToolStripMenuItem->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"advancedToolStripMenuItem.Image")));
			this->advancedToolStripMenuItem->Name = L"advancedToolStripMenuItem";
			this->advancedToolStripMenuItem->Size = System::Drawing::Size(207, 22);
			this->advancedToolStripMenuItem->Text = L"&Advanced Settings...";
			// 
			// executionStateToolStripMenuItem
			// 
			this->executionStateToolStripMenuItem->AutoCompleteCustomSource->AddRange(gcnew cli::array< System::String^  >(3) {
				L"Allow Suspend & Display Standby",
					L"Allow Display Standby Only", L"Prevent Both (\"Always On\")"
			});
			this->executionStateToolStripMenuItem->Items->AddRange(gcnew cli::array< System::Object^  >(3) {
				L"Normal Windows Behavior",
					L"Allow Display Powerdown, Avoid Suspend", L"Avoid Suspend & Display Powerdown"
			});
			this->executionStateToolStripMenuItem->Name = L"executionStateToolStripMenuItem";
			this->executionStateToolStripMenuItem->Size = System::Drawing::Size(245, 23);
			this->executionStateToolStripMenuItem->Text = L"Interruption Prevention:";
			this->executionStateToolStripMenuItem->ToolTipText = resources->GetString(L"executionStateToolStripMenuItem.ToolTipText");
			this->executionStateToolStripMenuItem->SelectedIndexChanged += gcnew System::EventHandler(this, &CosmicWind::executionStateToolStripMenuItem_SelectedIndexChanged);
			// 
			// hUDUpdateSpeedToolStripMenuItem
			// 
			this->hUDUpdateSpeedToolStripMenuItem->AutoToolTip = true;
			this->hUDUpdateSpeedToolStripMenuItem->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->hUDUpdateSpeedToolStripMenuItem->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			this->hUDUpdateSpeedToolStripMenuItem->Items->AddRange(gcnew cli::array< System::Object^  >(5) {
				L"UI Update Speed: 100 ms",
					L"UI Update Speed: 150 ms", L"UI Update Speed: 200 ms", L"UI Update Speed: 400 ms", L"UI Refresh Speed: 1 sec."
			});
			this->hUDUpdateSpeedToolStripMenuItem->Name = L"hUDUpdateSpeedToolStripMenuItem";
			this->hUDUpdateSpeedToolStripMenuItem->Size = System::Drawing::Size(245, 23);
			this->hUDUpdateSpeedToolStripMenuItem->ToolTipText = L"How often (in ms) to update the Heads-Up Display.\r\nThis setting has a slight effe"
				L"ct on CPU usage.";
			this->hUDUpdateSpeedToolStripMenuItem->SelectedIndexChanged += gcnew System::EventHandler(this, &CosmicWind::hUDUpdateSpeedToolStripMenuItem_SelectedIndexChanged);
			// 
			// hWMonitoringUpdateSpeedToolStripMenuItem
			// 
			this->hWMonitoringUpdateSpeedToolStripMenuItem->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->hWMonitoringUpdateSpeedToolStripMenuItem->Enabled = false;
			this->hWMonitoringUpdateSpeedToolStripMenuItem->Items->AddRange(gcnew cli::array< System::Object^  >(7) {
				L"50ms (Fastest)",
					L"75ms (Faster)", L"100ms (Fast)", L"150ms (Regular)", L"200ms (Slow)", L"500ms (Slower)", L"1sec (Slowest)"
			});
			this->hWMonitoringUpdateSpeedToolStripMenuItem->Name = L"hWMonitoringUpdateSpeedToolStripMenuItem";
			this->hWMonitoringUpdateSpeedToolStripMenuItem->Size = System::Drawing::Size(245, 23);
			this->hWMonitoringUpdateSpeedToolStripMenuItem->ToolTipText = L"How often (in ms) to update hardware monitoring readings\r\nin the Heads-Up Display"
				L" (such as temperature, power draw, etc.)\r\nThis setting has a slight effect on CP"
				L"U usage.";
			this->hWMonitoringUpdateSpeedToolStripMenuItem->Visible = false;
			// 
			// consoleOutputToolStripMenuItem
			// 
			this->consoleOutputToolStripMenuItem->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"consoleOutputToolStripMenuItem.Image")));
			this->consoleOutputToolStripMenuItem->Name = L"consoleOutputToolStripMenuItem";
			this->consoleOutputToolStripMenuItem->ShortcutKeys = static_cast<System::Windows::Forms::Keys>(((System::Windows::Forms::Keys::Control | System::Windows::Forms::Keys::Shift)
				| System::Windows::Forms::Keys::O));
			this->consoleOutputToolStripMenuItem->Size = System::Drawing::Size(305, 22);
			this->consoleOutputToolStripMenuItem->Text = L"Debug &Output...";
			this->consoleOutputToolStripMenuItem->ToolTipText = L"Displays a window with STDOUT stream output.";
			this->consoleOutputToolStripMenuItem->Visible = false;
			this->consoleOutputToolStripMenuItem->Click += gcnew System::EventHandler(this, &CosmicWind::consoleOutputToolStripMenuItem_Click);
			// 
			// miscCommandsToolStripMenuItem
			// 
			this->miscCommandsToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(5) {
				this->deviceCountersToolStripMenuItem,
					this->nVMLStartupToolStripMenuItem, this->pauseUnpauseToolStripMenuItem, this->newMessageToolStripMenuItem, this->resetHashrateCalcToolStripMenuItem
			});
			this->miscCommandsToolStripMenuItem->Enabled = false;
			this->miscCommandsToolStripMenuItem->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"miscCommandsToolStripMenuItem.Image")));
			this->miscCommandsToolStripMenuItem->Name = L"miscCommandsToolStripMenuItem";
			this->miscCommandsToolStripMenuItem->Size = System::Drawing::Size(305, 22);
			this->miscCommandsToolStripMenuItem->Text = L"&Misc Commands...";
			this->miscCommandsToolStripMenuItem->Visible = false;
			// 
			// deviceCountersToolStripMenuItem
			// 
			this->deviceCountersToolStripMenuItem->Name = L"deviceCountersToolStripMenuItem";
			this->deviceCountersToolStripMenuItem->ShortcutKeys = static_cast<System::Windows::Forms::Keys>(((System::Windows::Forms::Keys::Control | System::Windows::Forms::Keys::Shift)
				| System::Windows::Forms::Keys::C));
			this->deviceCountersToolStripMenuItem->Size = System::Drawing::Size(270, 22);
			this->deviceCountersToolStripMenuItem->Text = L"CUDA Device Counters";
			this->deviceCountersToolStripMenuItem->ToolTipText = L"Displays CUDA devices\' 64-bit counters to STDOUT.";
			this->deviceCountersToolStripMenuItem->Visible = false;
			this->deviceCountersToolStripMenuItem->Click += gcnew System::EventHandler(this, &CosmicWind::deviceCountersToolStripMenuItem_Click);
			// 
			// nVMLStartupToolStripMenuItem
			// 
			this->nVMLStartupToolStripMenuItem->Name = L"nVMLStartupToolStripMenuItem";
			this->nVMLStartupToolStripMenuItem->ShortcutKeys = static_cast<System::Windows::Forms::Keys>((System::Windows::Forms::Keys::Control | System::Windows::Forms::Keys::H));
			this->nVMLStartupToolStripMenuItem->Size = System::Drawing::Size(270, 22);
			this->nVMLStartupToolStripMenuItem->Text = L"Hardware Monitoring Start";
			this->nVMLStartupToolStripMenuItem->ToolTipText = L"Starts hardware monitoring code manually.";
			// 
			// pauseUnpauseToolStripMenuItem
			// 
			this->pauseUnpauseToolStripMenuItem->Name = L"pauseUnpauseToolStripMenuItem";
			this->pauseUnpauseToolStripMenuItem->Size = System::Drawing::Size(270, 22);
			this->pauseUnpauseToolStripMenuItem->Text = L"Network (Un)pause";
			this->pauseUnpauseToolStripMenuItem->ToolTipText = L"If mining has been globally paused due to difficulty\r\ncontacting the pool for an "
				L"extended period, this option\r\nwill resume mining normally.";
			this->pauseUnpauseToolStripMenuItem->Click += gcnew System::EventHandler(this, &CosmicWind::pauseUnpauseToolStripMenuItem_Click);
			// 
			// newMessageToolStripMenuItem
			// 
			this->newMessageToolStripMenuItem->Enabled = false;
			this->newMessageToolStripMenuItem->Name = L"newMessageToolStripMenuItem";
			this->newMessageToolStripMenuItem->Size = System::Drawing::Size(270, 22);
			this->newMessageToolStripMenuItem->Text = L"Generate New Message";
			this->newMessageToolStripMenuItem->ToolTipText = L"Generate new mining message for all GPUs.";
			// 
			// resetHashrateCalcToolStripMenuItem
			// 
			this->resetHashrateCalcToolStripMenuItem->Name = L"resetHashrateCalcToolStripMenuItem";
			this->resetHashrateCalcToolStripMenuItem->ShortcutKeys = System::Windows::Forms::Keys::F5;
			this->resetHashrateCalcToolStripMenuItem->Size = System::Drawing::Size(270, 22);
			this->resetHashrateCalcToolStripMenuItem->Text = L"Refresh Hashrate Calc";
			this->resetHashrateCalcToolStripMenuItem->Click += gcnew System::EventHandler(this, &CosmicWind::resetHashrateCalcToolStripMenuItem_Click);
			// 
			// menuitem_options_computetarg
			// 
			this->menuitem_options_computetarg->AutoToolTip = true;
			this->menuitem_options_computetarg->Checked = true;
			this->menuitem_options_computetarg->CheckOnClick = true;
			this->menuitem_options_computetarg->CheckState = System::Windows::Forms::CheckState::Checked;
			this->menuitem_options_computetarg->Name = L"menuitem_options_computetarg";
			this->menuitem_options_computetarg->Size = System::Drawing::Size(305, 22);
			this->menuitem_options_computetarg->Text = L"Compute Target Locally";
			this->menuitem_options_computetarg->ToolTipText = L"- If checked, the mining target will be computed from\r\nthe difficulty# and maxTar"
				L"get, locally.\r\n- If unchecked, the target will be retrieved from the\r\npool or to"
				L"ken contract.";
			// 
			// viewToolStripMenuItem1
			// 
			this->viewToolStripMenuItem1->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(3) {
				this->minimizeToTrayToolStripMenuItem,
					this->autoResizeColumnsToolStripMenuItem1, this->keyboardHotkeysToolStripMenuItem
			});
			this->viewToolStripMenuItem1->Name = L"viewToolStripMenuItem1";
			this->viewToolStripMenuItem1->Size = System::Drawing::Size(207, 22);
			this->viewToolStripMenuItem1->Text = L"View Settings";
			// 
			// minimizeToTrayToolStripMenuItem
			// 
			this->minimizeToTrayToolStripMenuItem->CheckOnClick = true;
			this->minimizeToTrayToolStripMenuItem->Name = L"minimizeToTrayToolStripMenuItem";
			this->minimizeToTrayToolStripMenuItem->Size = System::Drawing::Size(192, 22);
			this->minimizeToTrayToolStripMenuItem->Text = L"Minimize to Tray";
			this->minimizeToTrayToolStripMenuItem->CheckStateChanged += gcnew System::EventHandler(this, &CosmicWind::minimizeToTrayToolStripMenuItem_CheckStateChanged);
			// 
			// autoResizeColumnsToolStripMenuItem1
			// 
			this->autoResizeColumnsToolStripMenuItem1->CheckOnClick = true;
			this->autoResizeColumnsToolStripMenuItem1->Name = L"autoResizeColumnsToolStripMenuItem1";
			this->autoResizeColumnsToolStripMenuItem1->Size = System::Drawing::Size(192, 22);
			this->autoResizeColumnsToolStripMenuItem1->Text = L"Auto-Resize Columns";
			// 
			// keyboardHotkeysToolStripMenuItem
			// 
			this->keyboardHotkeysToolStripMenuItem->Enabled = false;
			this->keyboardHotkeysToolStripMenuItem->Name = L"keyboardHotkeysToolStripMenuItem";
			this->keyboardHotkeysToolStripMenuItem->Size = System::Drawing::Size(192, 22);
			this->keyboardHotkeysToolStripMenuItem->Text = L"Keyboard Hotkeys...";
			this->keyboardHotkeysToolStripMenuItem->Visible = false;
			// 
			// devicesModesToolStripMenuItem
			// 
			this->devicesModesToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(3) {
				this->nVidiaCUDAGPUsOnlyToolStripMenuItem,
					this->cUDAGPUsCPUThreadsToolStripMenuItem, this->cPUThreadsOnlyToolStripMenuItem
			});
			this->devicesModesToolStripMenuItem->Enabled = false;
			this->devicesModesToolStripMenuItem->Name = L"devicesModesToolStripMenuItem";
			this->devicesModesToolStripMenuItem->Size = System::Drawing::Size(207, 22);
			this->devicesModesToolStripMenuItem->Text = L"Devices/Modes";
			this->devicesModesToolStripMenuItem->Visible = false;
			// 
			// nVidiaCUDAGPUsOnlyToolStripMenuItem
			// 
			this->nVidiaCUDAGPUsOnlyToolStripMenuItem->Name = L"nVidiaCUDAGPUsOnlyToolStripMenuItem";
			this->nVidiaCUDAGPUsOnlyToolStripMenuItem->Size = System::Drawing::Size(228, 22);
			this->nVidiaCUDAGPUsOnlyToolStripMenuItem->Text = L"nVidia/CUDA GPUs Only";
			// 
			// cUDAGPUsCPUThreadsToolStripMenuItem
			// 
			this->cUDAGPUsCPUThreadsToolStripMenuItem->Name = L"cUDAGPUsCPUThreadsToolStripMenuItem";
			this->cUDAGPUsCPUThreadsToolStripMenuItem->Size = System::Drawing::Size(228, 22);
			this->cUDAGPUsCPUThreadsToolStripMenuItem->Text = L"CUDA GPUs + CPU Threads";
			// 
			// cPUThreadsOnlyToolStripMenuItem
			// 
			this->cPUThreadsOnlyToolStripMenuItem->Name = L"cPUThreadsOnlyToolStripMenuItem";
			this->cPUThreadsOnlyToolStripMenuItem->Size = System::Drawing::Size(228, 22);
			this->cPUThreadsOnlyToolStripMenuItem->Text = L"CPU Threads Only";
			// 
			// viewToolStripMenuItem
			// 
			this->viewToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(2) {
				this->columnsToolStripMenuItem,
					this->toolStripSeparator10
			});
			this->viewToolStripMenuItem->Name = L"viewToolStripMenuItem";
			this->viewToolStripMenuItem->Size = System::Drawing::Size(45, 20);
			this->viewToolStripMenuItem->Text = L"&View";
			this->viewToolStripMenuItem->Visible = false;
			// 
			// columnsToolStripMenuItem
			// 
			this->columnsToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(9) {
				this->devicesListToolStripMenuItem,
					this->toolStripMenuItem4, this->toolStripMenuItem5, this->toolStripMenuItem6, this->toolStripMenuItem7, this->toolStripMenuItem1,
					this->toolStripMenuItem3, this->toolStripSeparator9, this->toolStripMenuItem2
			});
			this->columnsToolStripMenuItem->Enabled = false;
			this->columnsToolStripMenuItem->Name = L"columnsToolStripMenuItem";
			this->columnsToolStripMenuItem->Size = System::Drawing::Size(156, 22);
			this->columnsToolStripMenuItem->Text = L"Columns (WIP)";
			// 
			// devicesListToolStripMenuItem
			// 
			this->devicesListToolStripMenuItem->Font = (gcnew System::Drawing::Font(L"Segoe UI", 9, static_cast<System::Drawing::FontStyle>((System::Drawing::FontStyle::Bold | System::Drawing::FontStyle::Underline)),
				System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
			this->devicesListToolStripMenuItem->Name = L"devicesListToolStripMenuItem";
			this->devicesListToolStripMenuItem->Size = System::Drawing::Size(247, 22);
			this->devicesListToolStripMenuItem->Text = L"Devices List";
			// 
			// toolStripMenuItem4
			// 
			this->toolStripMenuItem4->CheckOnClick = true;
			this->toolStripMenuItem4->Name = L"toolStripMenuItem4";
			this->toolStripMenuItem4->Size = System::Drawing::Size(247, 22);
			this->toolStripMenuItem4->Text = L"GPU Temperature";
			// 
			// toolStripMenuItem5
			// 
			this->toolStripMenuItem5->CheckOnClick = true;
			this->toolStripMenuItem5->Name = L"toolStripMenuItem5";
			this->toolStripMenuItem5->Size = System::Drawing::Size(247, 22);
			this->toolStripMenuItem5->Text = L"Card Power Usage";
			// 
			// toolStripMenuItem6
			// 
			this->toolStripMenuItem6->CheckOnClick = true;
			this->toolStripMenuItem6->Name = L"toolStripMenuItem6";
			this->toolStripMenuItem6->Size = System::Drawing::Size(247, 22);
			this->toolStripMenuItem6->Text = L"Fan/Pump Speed (Tachometer)";
			// 
			// toolStripMenuItem7
			// 
			this->toolStripMenuItem7->CheckOnClick = true;
			this->toolStripMenuItem7->Name = L"toolStripMenuItem7";
			this->toolStripMenuItem7->Size = System::Drawing::Size(247, 22);
			this->toolStripMenuItem7->Text = L"Fan/Pump Setting (%)";
			// 
			// toolStripMenuItem1
			// 
			this->toolStripMenuItem1->CheckOnClick = true;
			this->toolStripMenuItem1->Name = L"toolStripMenuItem1";
			this->toolStripMenuItem1->Size = System::Drawing::Size(247, 22);
			this->toolStripMenuItem1->Text = L"Solve Time";
			// 
			// toolStripMenuItem3
			// 
			this->toolStripMenuItem3->CheckOnClick = true;
			this->toolStripMenuItem3->Name = L"toolStripMenuItem3";
			this->toolStripMenuItem3->Size = System::Drawing::Size(247, 22);
			this->toolStripMenuItem3->Text = L"CUDA Utilization";
			// 
			// toolStripSeparator9
			// 
			this->toolStripSeparator9->Name = L"toolStripSeparator9";
			this->toolStripSeparator9->Size = System::Drawing::Size(244, 6);
			// 
			// toolStripMenuItem2
			// 
			this->toolStripMenuItem2->Font = (gcnew System::Drawing::Font(L"Segoe UI", 9, System::Drawing::FontStyle::Underline, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->toolStripMenuItem2->Name = L"toolStripMenuItem2";
			this->toolStripMenuItem2->Size = System::Drawing::Size(247, 22);
			this->toolStripMenuItem2->Text = L"Solutions View";
			// 
			// toolStripSeparator10
			// 
			this->toolStripSeparator10->Name = L"toolStripSeparator10";
			this->toolStripSeparator10->Size = System::Drawing::Size(153, 6);
			// 
			// miscToolStripMenuItem
			// 
			this->miscToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(1) { this->checkMiningRewardToolStripMenuItem });
			this->miscToolStripMenuItem->Name = L"miscToolStripMenuItem";
			this->miscToolStripMenuItem->Size = System::Drawing::Size(45, 20);
			this->miscToolStripMenuItem->Text = L"Misc";
			// 
			// checkMiningRewardToolStripMenuItem
			// 
			this->checkMiningRewardToolStripMenuItem->Name = L"checkMiningRewardToolStripMenuItem";
			this->checkMiningRewardToolStripMenuItem->Size = System::Drawing::Size(195, 22);
			this->checkMiningRewardToolStripMenuItem->Text = L"Check Mining Reward";
			this->checkMiningRewardToolStripMenuItem->Click += gcnew System::EventHandler(this, &CosmicWind::checkMiningRewardToolStripMenuItem_Click);
			// 
			// notifyIcon1
			// 
			this->notifyIcon1->BalloonTipIcon = System::Windows::Forms::ToolTipIcon::Info;
			this->notifyIcon1->BalloonTipTitle = L"COSMiC v4.1.5";
			this->notifyIcon1->ContextMenuStrip = this->trayContextMenu;
			this->notifyIcon1->Icon = (cli::safe_cast<System::Drawing::Icon^>(resources->GetObject(L"notifyIcon1.Icon")));
			this->notifyIcon1->Text = L"COSMiC ";
			this->notifyIcon1->DoubleClick += gcnew System::EventHandler(this, &CosmicWind::NotifyIcon1_DoubleClick);
			// 
			// trayContextMenu
			// 
			this->trayContextMenu->Font = (gcnew System::Drawing::Font(L"Segoe UI", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->trayContextMenu->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(5) {
				this->cOSMiCVersionToolStripMenuItem,
					this->toolStripSeparator6, this->summaryToolStripMenuItem, this->showToolStripMenuItem, this->quitToolStripMenuItem1
			});
			this->trayContextMenu->Name = L"trayContextMenu";
			this->trayContextMenu->Size = System::Drawing::Size(193, 98);
			// 
			// cOSMiCVersionToolStripMenuItem
			// 
			this->cOSMiCVersionToolStripMenuItem->BackColor = System::Drawing::SystemColors::Control;
			this->cOSMiCVersionToolStripMenuItem->Enabled = false;
			this->cOSMiCVersionToolStripMenuItem->Font = (gcnew System::Drawing::Font(L"Segoe UI Semibold", 9.75F, System::Drawing::FontStyle::Bold,
				System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
			this->cOSMiCVersionToolStripMenuItem->ForeColor = System::Drawing::SystemColors::WindowText;
			this->cOSMiCVersionToolStripMenuItem->Name = L"cOSMiCVersionToolStripMenuItem";
			this->cOSMiCVersionToolStripMenuItem->Size = System::Drawing::Size(192, 22);
			this->cOSMiCVersionToolStripMenuItem->Text = L"COSMiC v4.1.5 TEST";
			// 
			// toolStripSeparator6
			// 
			this->toolStripSeparator6->Name = L"toolStripSeparator6";
			this->toolStripSeparator6->Size = System::Drawing::Size(189, 6);
			// 
			// summaryToolStripMenuItem
			// 
			this->summaryToolStripMenuItem->Font = (gcnew System::Drawing::Font(L"Segoe UI", 9));
			this->summaryToolStripMenuItem->Name = L"summaryToolStripMenuItem";
			this->summaryToolStripMenuItem->Size = System::Drawing::Size(192, 22);
			this->summaryToolStripMenuItem->Text = L"Summary...";
			this->summaryToolStripMenuItem->Visible = false;
			this->summaryToolStripMenuItem->Click += gcnew System::EventHandler(this, &CosmicWind::SummaryToolStripMenuItem_Click);
			// 
			// showToolStripMenuItem
			// 
			this->showToolStripMenuItem->Font = (gcnew System::Drawing::Font(L"Segoe UI", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->showToolStripMenuItem->Name = L"showToolStripMenuItem";
			this->showToolStripMenuItem->Size = System::Drawing::Size(192, 22);
			this->showToolStripMenuItem->Text = L"Show...";
			this->showToolStripMenuItem->Click += gcnew System::EventHandler(this, &CosmicWind::ShowToolStripMenuItem_Click);
			// 
			// quitToolStripMenuItem1
			// 
			this->quitToolStripMenuItem1->Font = (gcnew System::Drawing::Font(L"Segoe UI", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->quitToolStripMenuItem1->Name = L"quitToolStripMenuItem1";
			this->quitToolStripMenuItem1->Size = System::Drawing::Size(192, 22);
			this->quitToolStripMenuItem1->Text = L"Quit";
			this->quitToolStripMenuItem1->Click += gcnew System::EventHandler(this, &CosmicWind::QuitToolStripMenuItem1_Click);
			// 
			// timer_net_worker_restart
			// 
			this->timer_net_worker_restart->Interval = 5000;
			this->timer_net_worker_restart->Tick += gcnew System::EventHandler(this, &CosmicWind::timer_net_worker_restart_Tick);
			// 
			// timer_resumeafternetfail
			// 
			this->timer_resumeafternetfail->Interval = 25000;
			this->timer_resumeafternetfail->Tick += gcnew System::EventHandler(this, &CosmicWind::timer_resumeafternetfail_Tick);
			// 
			// timer_enableminingbutton
			// 
			this->timer_enableminingbutton->Interval = 4250;
			this->timer_enableminingbutton->Tick += gcnew System::EventHandler(this, &CosmicWind::timer_enableminingbutton_Tick);
			// 
			// timer2
			// 
			this->timer2->Enabled = true;
			this->timer2->Interval = 1500;
			this->timer2->Tick += gcnew System::EventHandler(this, &CosmicWind::timer2_Tick);
			// 
			// txViewContextMenu
			// 
			this->txViewContextMenu->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(9) {
				this->viewInBlockExplorerToolStripMenuItem,
					this->clearToolStripMenuItem, this->toolStripSeparator11, this->resubmitTransactionToolStripMenuItem, this->cancelTransactionToolStripMenuItem,
					this->viewTxReceiptToolStripMenuItem, this->toolStripSeparator5, this->purgeTxViewMenuItem, this->writePoWToFileToolStripMenuItem
			});
			this->txViewContextMenu->Name = L"txViewContextMenu";
			this->txViewContextMenu->Size = System::Drawing::Size(200, 170);
			this->txViewContextMenu->Opening += gcnew System::ComponentModel::CancelEventHandler(this, &CosmicWind::TxViewContextMenu_Opening);
			// 
			// viewInBlockExplorerToolStripMenuItem
			// 
			this->viewInBlockExplorerToolStripMenuItem->Name = L"viewInBlockExplorerToolStripMenuItem";
			this->viewInBlockExplorerToolStripMenuItem->Size = System::Drawing::Size(199, 22);
			this->viewInBlockExplorerToolStripMenuItem->Text = L"View in Block Explorer...";
			this->viewInBlockExplorerToolStripMenuItem->Click += gcnew System::EventHandler(this, &CosmicWind::ViewInBlockExplorerToolStripMenuItem_Click);
			// 
			// clearToolStripMenuItem
			// 
			this->clearToolStripMenuItem->Name = L"clearToolStripMenuItem";
			this->clearToolStripMenuItem->Size = System::Drawing::Size(199, 22);
			this->clearToolStripMenuItem->Text = L"Clear Item(s)";
			this->clearToolStripMenuItem->Visible = false;
			this->clearToolStripMenuItem->Click += gcnew System::EventHandler(this, &CosmicWind::ClearToolStripMenuItem_Click);
			// 
			// toolStripSeparator11
			// 
			this->toolStripSeparator11->Name = L"toolStripSeparator11";
			this->toolStripSeparator11->Size = System::Drawing::Size(196, 6);
			// 
			// resubmitTransactionToolStripMenuItem
			// 
			this->resubmitTransactionToolStripMenuItem->Name = L"resubmitTransactionToolStripMenuItem";
			this->resubmitTransactionToolStripMenuItem->Size = System::Drawing::Size(199, 22);
			this->resubmitTransactionToolStripMenuItem->Text = L"Re-submit Transaction";
			this->resubmitTransactionToolStripMenuItem->Click += gcnew System::EventHandler(this, &CosmicWind::resubmitTransactionToolStripMenuItem_Click);
			// 
			// cancelTransactionToolStripMenuItem
			// 
			this->cancelTransactionToolStripMenuItem->Enabled = false;
			this->cancelTransactionToolStripMenuItem->Name = L"cancelTransactionToolStripMenuItem";
			this->cancelTransactionToolStripMenuItem->Size = System::Drawing::Size(199, 22);
			this->cancelTransactionToolStripMenuItem->Text = L"Cancel Transaction";
			// 
			// viewTxReceiptToolStripMenuItem
			// 
			this->viewTxReceiptToolStripMenuItem->Name = L"viewTxReceiptToolStripMenuItem";
			this->viewTxReceiptToolStripMenuItem->Size = System::Drawing::Size(199, 22);
			this->viewTxReceiptToolStripMenuItem->Text = L"View Tx Receipt...";
			this->viewTxReceiptToolStripMenuItem->Click += gcnew System::EventHandler(this, &CosmicWind::ViewTxReceiptToolStripMenuItem_Click);
			// 
			// toolStripSeparator5
			// 
			this->toolStripSeparator5->Name = L"toolStripSeparator5";
			this->toolStripSeparator5->Size = System::Drawing::Size(196, 6);
			// 
			// purgeTxViewMenuItem
			// 
			this->purgeTxViewMenuItem->Name = L"purgeTxViewMenuItem";
			this->purgeTxViewMenuItem->Size = System::Drawing::Size(199, 22);
			this->purgeTxViewMenuItem->Text = L"Clear Solutions";
			this->purgeTxViewMenuItem->Click += gcnew System::EventHandler(this, &CosmicWind::TxViewPurgeList_Click);
			// 
			// writePoWToFileToolStripMenuItem
			// 
			this->writePoWToFileToolStripMenuItem->Enabled = false;
			this->writePoWToFileToolStripMenuItem->Name = L"writePoWToFileToolStripMenuItem";
			this->writePoWToFileToolStripMenuItem->Size = System::Drawing::Size(199, 22);
			this->writePoWToFileToolStripMenuItem->Text = L"Write PoW to File...";
			this->writePoWToFileToolStripMenuItem->Visible = false;
			// 
			// contextMenu_gpu
			// 
			this->contextMenu_gpu->AllowMerge = false;
			this->contextMenu_gpu->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(10) {
				this->gpuNameAndIndexMenuItem,
					this->toolStripSeparator4, this->cUDAEngineToolStripMenuItem, this->toolStripSeparator8, this->gpuSummaryMenuItem, this->setIntensityToolStripMenuItem,
					this->hWMonitoringAndAlarmsToolStripMenuItem, this->toolStripSeparator3, this->toolStripMenuItemForceUnpause, this->enableDisableGpuToolStripMenuItem
			});
			this->contextMenu_gpu->Name = L"contextMenu_gpu";
			this->contextMenu_gpu->Size = System::Drawing::Size(297, 181);
			this->contextMenu_gpu->Text = L"GPU Options";
			this->contextMenu_gpu->Closing += gcnew System::Windows::Forms::ToolStripDropDownClosingEventHandler(this, &CosmicWind::contextMenu_gpu_Closing);
			this->contextMenu_gpu->Opening += gcnew System::ComponentModel::CancelEventHandler(this, &CosmicWind::contextMenu_gpu_Opening);
			// 
			// gpuNameAndIndexMenuItem
			// 
			this->gpuNameAndIndexMenuItem->BackColor = System::Drawing::SystemColors::ButtonHighlight;
			this->gpuNameAndIndexMenuItem->Font = (gcnew System::Drawing::Font(L"Segoe UI Semibold", 9.75F, System::Drawing::FontStyle::Bold,
				System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
			this->gpuNameAndIndexMenuItem->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"gpuNameAndIndexMenuItem.Image")));
			this->gpuNameAndIndexMenuItem->Name = L"gpuNameAndIndexMenuItem";
			this->gpuNameAndIndexMenuItem->RightToLeft = System::Windows::Forms::RightToLeft::No;
			this->gpuNameAndIndexMenuItem->Size = System::Drawing::Size(296, 22);
			this->gpuNameAndIndexMenuItem->Text = L"CUDA Device Index/Name";
			this->gpuNameAndIndexMenuItem->TextDirection = System::Windows::Forms::ToolStripTextDirection::Horizontal;
			// 
			// toolStripSeparator4
			// 
			this->toolStripSeparator4->Name = L"toolStripSeparator4";
			this->toolStripSeparator4->Size = System::Drawing::Size(293, 6);
			// 
			// cUDAEngineToolStripMenuItem
			// 
			this->cUDAEngineToolStripMenuItem->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->cUDAEngineToolStripMenuItem->Items->AddRange(gcnew cli::array< System::Object^  >(2) { L"CUDA Engine: Hashburner", L"CUDA Engine: Compatibility" });
			this->cUDAEngineToolStripMenuItem->Name = L"cUDAEngineToolStripMenuItem";
			this->cUDAEngineToolStripMenuItem->Size = System::Drawing::Size(236, 23);
			this->cUDAEngineToolStripMenuItem->ToolTipText = L"Hashburner: select for highest performance.\r\nCompatibility: for older/unusual GPU"
				L" architectures.\r\n";
			this->cUDAEngineToolStripMenuItem->SelectedIndexChanged += gcnew System::EventHandler(this, &CosmicWind::cUDAEngineToolStripMenuItem_SelectedIndexChanged);
			// 
			// toolStripSeparator8
			// 
			this->toolStripSeparator8->Name = L"toolStripSeparator8";
			this->toolStripSeparator8->Size = System::Drawing::Size(293, 6);
			// 
			// gpuSummaryMenuItem
			// 
			this->gpuSummaryMenuItem->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"gpuSummaryMenuItem.Image")));
			this->gpuSummaryMenuItem->Name = L"gpuSummaryMenuItem";
			this->gpuSummaryMenuItem->Size = System::Drawing::Size(296, 22);
			this->gpuSummaryMenuItem->Text = L"GPU Summary...";
			this->gpuSummaryMenuItem->ToolTipText = L"View real-time readings and mining statistics for this GPU.";
			this->gpuSummaryMenuItem->Click += gcnew System::EventHandler(this, &CosmicWind::configureGPUToolStripMenuItem_Click);
			// 
			// setIntensityToolStripMenuItem
			// 
			this->setIntensityToolStripMenuItem->Enabled = false;
			this->setIntensityToolStripMenuItem->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"setIntensityToolStripMenuItem.Image")));
			this->setIntensityToolStripMenuItem->Name = L"setIntensityToolStripMenuItem";
			this->setIntensityToolStripMenuItem->Size = System::Drawing::Size(296, 22);
			this->setIntensityToolStripMenuItem->Text = L"Set Mining Intensity...";
			this->setIntensityToolStripMenuItem->ToolTipText = L"Configure the Mining Intensity for this GPU. Intensity\r\ndetermines how hard the G"
				L"PU works and should be\r\nadjusted for maximum performance and stability.";
			this->setIntensityToolStripMenuItem->Click += gcnew System::EventHandler(this, &CosmicWind::setIntensityToolStripMenuItem_Click);
			// 
			// hWMonitoringAndAlarmsToolStripMenuItem
			// 
			this->hWMonitoringAndAlarmsToolStripMenuItem->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"hWMonitoringAndAlarmsToolStripMenuItem.Image")));
			this->hWMonitoringAndAlarmsToolStripMenuItem->Name = L"hWMonitoringAndAlarmsToolStripMenuItem";
			this->hWMonitoringAndAlarmsToolStripMenuItem->Size = System::Drawing::Size(296, 22);
			this->hWMonitoringAndAlarmsToolStripMenuItem->Text = L"HW Health  / Alarms...";
			this->hWMonitoringAndAlarmsToolStripMenuItem->ToolTipText = L"Configure temperature, fan/pump and power monitoring\r\nsettings, as well as Safety"
				L" Features like overheating protection\r\nand fan fail detection.";
			this->hWMonitoringAndAlarmsToolStripMenuItem->Click += gcnew System::EventHandler(this, &CosmicWind::hWMonitoringAndAlarmsToolStripMenuItem_Click);
			// 
			// toolStripSeparator3
			// 
			this->toolStripSeparator3->Name = L"toolStripSeparator3";
			this->toolStripSeparator3->Size = System::Drawing::Size(293, 6);
			// 
			// toolStripMenuItemForceUnpause
			// 
			this->toolStripMenuItemForceUnpause->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"toolStripMenuItemForceUnpause.Image")));
			this->toolStripMenuItemForceUnpause->Name = L"toolStripMenuItemForceUnpause";
			this->toolStripMenuItemForceUnpause->Size = System::Drawing::Size(296, 22);
			this->toolStripMenuItemForceUnpause->Text = L"Force Unpause";
			this->toolStripMenuItemForceUnpause->Click += gcnew System::EventHandler(this, &CosmicWind::forceUnpauseToolStripMenuItem_Click);
			// 
			// enableDisableGpuToolStripMenuItem
			// 
			this->enableDisableGpuToolStripMenuItem->Font = (gcnew System::Drawing::Font(L"Segoe UI", 9, System::Drawing::FontStyle::Regular,
				System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
			this->enableDisableGpuToolStripMenuItem->ForeColor = System::Drawing::Color::Black;
			this->enableDisableGpuToolStripMenuItem->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"enableDisableGpuToolStripMenuItem.Image")));
			this->enableDisableGpuToolStripMenuItem->Name = L"enableDisableGpuToolStripMenuItem";
			this->enableDisableGpuToolStripMenuItem->ShowShortcutKeys = false;
			this->enableDisableGpuToolStripMenuItem->Size = System::Drawing::Size(296, 22);
			this->enableDisableGpuToolStripMenuItem->Text = L"Disable Device";
			this->enableDisableGpuToolStripMenuItem->ToolTipText = L"Enable / Disable this card. If Disabled, it will not be used for mining.";
			this->enableDisableGpuToolStripMenuItem->Click += gcnew System::EventHandler(this, &CosmicWind::enableDisableGpuToolStripMenuItem_Click);
			// 
			// tooltip_COSMiC
			// 
			this->tooltip_COSMiC->AutomaticDelay = 0;
			this->tooltip_COSMiC->AutoPopDelay = 32000;
			this->tooltip_COSMiC->BackColor = System::Drawing::SystemColors::ActiveCaption;
			this->tooltip_COSMiC->InitialDelay = 350;
			this->tooltip_COSMiC->ReshowDelay = 150;
			this->tooltip_COSMiC->ToolTipIcon = System::Windows::Forms::ToolTipIcon::Info;
			this->tooltip_COSMiC->ToolTipTitle = L"Info";
			this->tooltip_COSMiC->UseAnimation = false;
			this->tooltip_COSMiC->UseFading = false;
			// 
			// tooltip_NET
			// 
			this->tooltip_NET->AutomaticDelay = 0;
			this->tooltip_NET->AutoPopDelay = 32000;
			this->tooltip_NET->BackColor = System::Drawing::SystemColors::ActiveCaption;
			this->tooltip_NET->InitialDelay = 350;
			this->tooltip_NET->ReshowDelay = 150;
			this->tooltip_NET->ToolTipIcon = System::Windows::Forms::ToolTipIcon::Info;
			this->tooltip_NET->ToolTipTitle = L"Network Worker";
			this->tooltip_NET->UseAnimation = false;
			this->tooltip_NET->UseFading = false;
			// 
			// timer_solobalance
			// 
			this->timer_solobalance->Interval = 10000;
			this->timer_solobalance->Tick += gcnew System::EventHandler(this, &CosmicWind::timer_solobalance_Tick);
			// 
			// bgWorker_SoloView
			// 
			this->bgWorker_SoloView->WorkerReportsProgress = true;
			this->bgWorker_SoloView->DoWork += gcnew System::ComponentModel::DoWorkEventHandler(this, &CosmicWind::BgWorker_SoloView_DoWork);
			this->bgWorker_SoloView->ProgressChanged += gcnew System::ComponentModel::ProgressChangedEventHandler(this, &CosmicWind::BgWorker_SoloView_ProgressChanged);
			this->bgWorker_SoloView->RunWorkerCompleted += gcnew System::ComponentModel::RunWorkerCompletedEventHandler(this, &CosmicWind::BgWorker_SoloView_RunWorkerCompleted);
			// 
			// timer_txview
			// 
			this->timer_txview->Interval = 1250;
			this->timer_txview->Tick += gcnew System::EventHandler(this, &CosmicWind::Timer_txview_Tick);
			// 
			// timer_cputhreadsview
			// 
			this->timer_cputhreadsview->Tick += gcnew System::EventHandler(this, &CosmicWind::Timer_cputhreadsview_Tick);
			// 
			// label6
			// 
			this->label6->Anchor = System::Windows::Forms::AnchorStyles::Left;
			this->label6->AutoSize = true;
			this->label6->BackColor = System::Drawing::SystemColors::ButtonFace;
			this->label6->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Underline, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->label6->Location = System::Drawing::Point(-3, 35);
			this->label6->Name = L"label6";
			this->label6->Padding = System::Windows::Forms::Padding(8, 0, 0, 0);
			this->label6->Size = System::Drawing::Size(185, 15);
			this->label6->TabIndex = 7;
			this->label6->Text = L"ERC918 Token Miner by LtTofu";
			this->label6->Click += gcnew System::EventHandler(this, &CosmicWind::label6_Click);
			// 
			// lbl_totalhashrate
			// 
			this->lbl_totalhashrate->AutoSize = true;
			this->lbl_totalhashrate->Location = System::Drawing::Point(15, 22);
			this->lbl_totalhashrate->Name = L"lbl_totalhashrate";
			this->lbl_totalhashrate->Size = System::Drawing::Size(13, 13);
			this->lbl_totalhashrate->TabIndex = 2;
			this->lbl_totalhashrate->Text = L"--";
			// 
			// lbl_totalsols
			// 
			this->lbl_totalsols->AutoSize = true;
			this->lbl_totalsols->Location = System::Drawing::Point(15, 37);
			this->lbl_totalsols->Name = L"lbl_totalsols";
			this->lbl_totalsols->Size = System::Drawing::Size(13, 13);
			this->lbl_totalsols->TabIndex = 3;
			this->lbl_totalsols->Text = L"--";
			// 
			// lbl_txncount
			// 
			this->lbl_txncount->Anchor = System::Windows::Forms::AnchorStyles::None;
			this->lbl_txncount->AutoSize = true;
			this->lbl_txncount->Location = System::Drawing::Point(479, 22);
			this->lbl_txncount->Name = L"lbl_txncount";
			this->lbl_txncount->Size = System::Drawing::Size(13, 13);
			this->lbl_txncount->TabIndex = 2;
			this->lbl_txncount->Text = L"--";
			this->lbl_txncount->Visible = false;
			// 
			// lowerPanel
			// 
			this->lowerPanel->Controls->Add(this->statusBar);
			this->lowerPanel->Controls->Add(this->lbl_hashbrowns);
			this->lowerPanel->Controls->Add(this->totalsGroupBox);
			this->lowerPanel->Controls->Add(this->lbl_totalminetime);
			this->lowerPanel->Controls->Add(this->pictureBox1);
			this->lowerPanel->Controls->Add(this->lbl_txncount);
			this->lowerPanel->Controls->Add(this->label6);
			this->lowerPanel->Dock = System::Windows::Forms::DockStyle::Bottom;
			this->lowerPanel->Location = System::Drawing::Point(0, 407);
			this->lowerPanel->Name = L"lowerPanel";
			this->lowerPanel->Size = System::Drawing::Size(880, 112);
			this->lowerPanel->TabIndex = 17;
			this->lowerPanel->Click += gcnew System::EventHandler(this, &CosmicWind::lowerPanel_Click);
			// 
			// statusBar
			// 
			this->statusBar->BackColor = System::Drawing::SystemColors::Window;
			this->statusBar->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(6) {
				this->statusbar_minerstate,
					this->statusbar_anncTXN, this->statusbar_anncNET, this->statusbar_balanceEth, this->statusbar_balanceTokens, this->statusbar_elapsedTime
			});
			this->statusBar->Location = System::Drawing::Point(0, 90);
			this->statusBar->Name = L"statusBar";
			this->statusBar->ShowItemToolTips = true;
			this->statusBar->Size = System::Drawing::Size(880, 22);
			this->statusBar->SizingGrip = false;
			this->statusBar->TabIndex = 15;
			this->statusBar->Text = L"status bar";
			// 
			// statusbar_minerstate
			// 
			this->statusbar_minerstate->AutoSize = false;
			this->statusbar_minerstate->Name = L"statusbar_minerstate";
			this->statusbar_minerstate->Overflow = System::Windows::Forms::ToolStripItemOverflow::Never;
			this->statusbar_minerstate->Size = System::Drawing::Size(150, 17);
			this->statusbar_minerstate->Text = L"Status: Initing CosmicWind...";
			this->statusbar_minerstate->TextAlign = System::Drawing::ContentAlignment::MiddleLeft;
			// 
			// statusbar_anncTXN
			// 
			this->statusbar_anncTXN->AutoSize = false;
			this->statusbar_anncTXN->AutoToolTip = true;
			this->statusbar_anncTXN->BorderStyle = System::Windows::Forms::Border3DStyle::Etched;
			this->statusbar_anncTXN->Enabled = false;
			this->statusbar_anncTXN->Name = L"statusbar_anncTXN";
			this->statusbar_anncTXN->Overflow = System::Windows::Forms::ToolStripItemOverflow::Never;
			this->statusbar_anncTXN->Size = System::Drawing::Size(150, 17);
			this->statusbar_anncTXN->Text = L"TXN";
			this->statusbar_anncTXN->TextAlign = System::Drawing::ContentAlignment::MiddleRight;
			this->statusbar_anncTXN->ToolTipText = L"TXhelper";
			// 
			// statusbar_anncNET
			// 
			this->statusbar_anncNET->AutoSize = false;
			this->statusbar_anncNET->AutoToolTip = true;
			this->statusbar_anncNET->BorderStyle = System::Windows::Forms::Border3DStyle::Etched;
			this->statusbar_anncNET->Enabled = false;
			this->statusbar_anncNET->Name = L"statusbar_anncNET";
			this->statusbar_anncNET->Overflow = System::Windows::Forms::ToolStripItemOverflow::Never;
			this->statusbar_anncNET->Size = System::Drawing::Size(80, 17);
			this->statusbar_anncNET->Text = L"NET";
			this->statusbar_anncNET->TextAlign = System::Drawing::ContentAlignment::MiddleLeft;
			this->statusbar_anncNET->ToolTipText = L"Network BGWorker";
			// 
			// statusbar_balanceEth
			// 
			this->statusbar_balanceEth->AutoSize = false;
			this->statusbar_balanceEth->Name = L"statusbar_balanceEth";
			this->statusbar_balanceEth->Overflow = System::Windows::Forms::ToolStripItemOverflow::Never;
			this->statusbar_balanceEth->Size = System::Drawing::Size(165, 17);
			this->statusbar_balanceEth->Spring = true;
			this->statusbar_balanceEth->Text = L"Ether Balance:  ";
			this->statusbar_balanceEth->TextAlign = System::Drawing::ContentAlignment::MiddleLeft;
			// 
			// statusbar_balanceTokens
			// 
			this->statusbar_balanceTokens->AutoSize = false;
			this->statusbar_balanceTokens->Name = L"statusbar_balanceTokens";
			this->statusbar_balanceTokens->Overflow = System::Windows::Forms::ToolStripItemOverflow::Never;
			this->statusbar_balanceTokens->Size = System::Drawing::Size(154, 17);
			this->statusbar_balanceTokens->Text = L"Token Balance:  ";
			this->statusbar_balanceTokens->TextAlign = System::Drawing::ContentAlignment::MiddleLeft;
			// 
			// statusbar_elapsedTime
			// 
			this->statusbar_elapsedTime->AutoSize = false;
			this->statusbar_elapsedTime->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Text;
			this->statusbar_elapsedTime->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"statusbar_elapsedTime.Image")));
			this->statusbar_elapsedTime->ImageAlign = System::Drawing::ContentAlignment::MiddleRight;
			this->statusbar_elapsedTime->Name = L"statusbar_elapsedTime";
			this->statusbar_elapsedTime->Overflow = System::Windows::Forms::ToolStripItemOverflow::Never;
			this->statusbar_elapsedTime->RightToLeft = System::Windows::Forms::RightToLeft::No;
			this->statusbar_elapsedTime->Size = System::Drawing::Size(165, 17);
			this->statusbar_elapsedTime->Spring = true;
			this->statusbar_elapsedTime->Text = L"v4.1.5 DEV TEST";
			this->statusbar_elapsedTime->TextAlign = System::Drawing::ContentAlignment::MiddleRight;
			// 
			// lbl_hashbrowns
			// 
			this->lbl_hashbrowns->AutoSize = true;
			this->lbl_hashbrowns->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->lbl_hashbrowns->Location = System::Drawing::Point(5, 70);
			this->lbl_hashbrowns->Name = L"lbl_hashbrowns";
			this->lbl_hashbrowns->Size = System::Drawing::Size(132, 13);
			this->lbl_hashbrowns->TabIndex = 13;
			this->lbl_hashbrowns->Text = L"Cookin\' Your Hashbrowns!";
			// 
			// totalsGroupBox
			// 
			this->totalsGroupBox->Anchor = System::Windows::Forms::AnchorStyles::Right;
			this->totalsGroupBox->Controls->Add(this->combobox_modeselect);
			this->totalsGroupBox->Controls->Add(this->lbl_totalsols);
			this->totalsGroupBox->Controls->Add(this->lbl_totalhashrate);
			this->totalsGroupBox->Controls->Add(this->lbl_totalpwr);
			this->totalsGroupBox->Controls->Add(this->start1);
			this->totalsGroupBox->Location = System::Drawing::Point(532, 8);
			this->totalsGroupBox->Name = L"totalsGroupBox";
			this->totalsGroupBox->Size = System::Drawing::Size(339, 78);
			this->totalsGroupBox->TabIndex = 12;
			this->totalsGroupBox->TabStop = false;
			this->totalsGroupBox->Text = L"Totals: ";
			// 
			// combobox_modeselect
			// 
			this->combobox_modeselect->Anchor = System::Windows::Forms::AnchorStyles::Right;
			this->combobox_modeselect->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->combobox_modeselect->Font = (gcnew System::Drawing::Font(L"Segoe UI", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->combobox_modeselect->FormattingEnabled = true;
			this->combobox_modeselect->Items->AddRange(gcnew cli::array< System::Object^  >(2) { L"Pool Mode", L"Solo Mode (TEST)" });
			this->combobox_modeselect->Location = System::Drawing::Point(192, 45);
			this->combobox_modeselect->Name = L"combobox_modeselect";
			this->combobox_modeselect->Size = System::Drawing::Size(132, 23);
			this->combobox_modeselect->TabIndex = 13;
			this->combobox_modeselect->SelectedIndexChanged += gcnew System::EventHandler(this, &CosmicWind::combobox_modeselect_SelectedIndexChanged);
			// 
			// tabControl1
			// 
			this->tabControl1->Controls->Add(this->eventsPage);
			this->tabControl1->Controls->Add(this->tabSolns);
			this->tabControl1->Controls->Add(this->tabCpu);
			this->tabControl1->Dock = System::Windows::Forms::DockStyle::Fill;
			this->tabControl1->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->tabControl1->Location = System::Drawing::Point(0, 0);
			this->tabControl1->MinimumSize = System::Drawing::Size(60, 60);
			this->tabControl1->Name = L"tabControl1";
			this->tabControl1->Padding = System::Drawing::Point(10, 6);
			this->tabControl1->SelectedIndex = 0;
			this->tabControl1->ShowToolTips = true;
			this->tabControl1->Size = System::Drawing::Size(880, 177);
			this->tabControl1->TabIndex = 13;
			// 
			// eventsPage
			// 
			this->eventsPage->Controls->Add(this->listbox_events);
			this->eventsPage->Location = System::Drawing::Point(4, 30);
			this->eventsPage->Name = L"eventsPage";
			this->eventsPage->Padding = System::Windows::Forms::Padding(3);
			this->eventsPage->Size = System::Drawing::Size(872, 143);
			this->eventsPage->TabIndex = 0;
			this->eventsPage->Text = L"Event Log";
			this->eventsPage->UseVisualStyleBackColor = true;
			// 
			// listbox_events
			// 
			this->listbox_events->BackColor = System::Drawing::Color::White;
			this->listbox_events->BorderStyle = System::Windows::Forms::BorderStyle::None;
			this->listbox_events->ContextMenuStrip = this->contextMenu_eventsBox;
			this->listbox_events->Dock = System::Windows::Forms::DockStyle::Fill;
			this->listbox_events->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->listbox_events->Location = System::Drawing::Point(3, 3);
			this->listbox_events->Name = L"listbox_events";
			this->listbox_events->SelectionMode = System::Windows::Forms::SelectionMode::MultiExtended;
			this->listbox_events->Size = System::Drawing::Size(866, 137);
			this->listbox_events->TabIndex = 1;
			this->listbox_events->UseTabStops = false;
			// 
			// tabSolns
			// 
			this->tabSolns->Controls->Add(this->listview_solutionsview);
			this->tabSolns->Location = System::Drawing::Point(4, 30);
			this->tabSolns->Name = L"tabSolns";
			this->tabSolns->Size = System::Drawing::Size(872, 143);
			this->tabSolns->TabIndex = 1;
			this->tabSolns->Text = L"Solutions (solo)";
			this->tabSolns->ToolTipText = L"View and manage network transactions for Solutions found in Solo Mode.";
			this->tabSolns->UseVisualStyleBackColor = true;
			// 
			// listview_solutionsview
			// 
			this->listview_solutionsview->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->listview_solutionsview->Columns->AddRange(gcnew cli::array< System::Windows::Forms::ColumnHeader^  >(8) {
				this->colhead_solnonce,
					this->colhead_txnstatus, this->colhead_deviceno, this->colhead_challenge, this->colhead_txviewslot, this->colhead_bufslot, this->columnHeader7,
					this->columnHeader8
			});
			this->listview_solutionsview->ContextMenuStrip = this->txViewContextMenu;
			this->listview_solutionsview->Dock = System::Windows::Forms::DockStyle::Fill;
			this->listview_solutionsview->FullRowSelect = true;
			this->listview_solutionsview->HeaderStyle = System::Windows::Forms::ColumnHeaderStyle::Nonclickable;
			this->listview_solutionsview->HideSelection = false;
			this->listview_solutionsview->Location = System::Drawing::Point(0, 0);
			this->listview_solutionsview->MultiSelect = false;
			this->listview_solutionsview->Name = L"listview_solutionsview";
			this->listview_solutionsview->Size = System::Drawing::Size(872, 143);
			this->listview_solutionsview->TabIndex = 0;
			this->listview_solutionsview->UseCompatibleStateImageBehavior = false;
			this->listview_solutionsview->View = System::Windows::Forms::View::Details;
			// 
			// colhead_solnonce
			// 
			this->colhead_solnonce->Text = L"Solution Nonce";
			this->colhead_solnonce->Width = 162;
			// 
			// colhead_txnstatus
			// 
			this->colhead_txnstatus->Text = L"Transaction Status";
			this->colhead_txnstatus->Width = 130;
			// 
			// colhead_deviceno
			// 
			this->colhead_deviceno->Text = L"Device #";
			this->colhead_deviceno->Width = 65;
			// 
			// colhead_challenge
			// 
			this->colhead_challenge->Text = L"Challenge";
			this->colhead_challenge->Width = 116;
			// 
			// colhead_txviewslot
			// 
			this->colhead_txviewslot->Text = L"-";
			this->colhead_txviewslot->Width = 36;
			// 
			// colhead_bufslot
			// 
			this->colhead_bufslot->Text = L"-";
			this->colhead_bufslot->Width = 36;
			// 
			// columnHeader7
			// 
			this->columnHeader7->Text = L"Node Response (TxHash / Error)";
			this->columnHeader7->Width = 214;
			// 
			// columnHeader8
			// 
			this->columnHeader8->Text = L"Network Nonce";
			this->columnHeader8->Width = 100;
			// 
			// tabCpu
			// 
			this->tabCpu->AutoScroll = true;
			this->tabCpu->Controls->Add(this->panel_threads);
			this->tabCpu->Controls->Add(this->panel2);
			this->tabCpu->Location = System::Drawing::Point(4, 30);
			this->tabCpu->Name = L"tabCpu";
			this->tabCpu->Size = System::Drawing::Size(872, 143);
			this->tabCpu->TabIndex = 0;
			this->tabCpu->Text = L"CPU Threads";
			this->tabCpu->ToolTipText = L"CPU Mining controls and status (experimental).";
			this->tabCpu->UseVisualStyleBackColor = true;
			// 
			// splitPanel
			// 
			this->splitPanel->BackColor = System::Drawing::SystemColors::ButtonFace;
			this->splitPanel->Dock = System::Windows::Forms::DockStyle::Fill;
			this->splitPanel->Location = System::Drawing::Point(0, 24);
			this->splitPanel->Margin = System::Windows::Forms::Padding(0);
			this->splitPanel->Name = L"splitPanel";
			this->splitPanel->Orientation = System::Windows::Forms::Orientation::Horizontal;
			// 
			// splitPanel.Panel1
			// 
			this->splitPanel->Panel1->Controls->Add(this->tablelayoutpanel_top);
			this->splitPanel->Panel1MinSize = 165;
			// 
			// splitPanel.Panel2
			// 
			this->splitPanel->Panel2->Controls->Add(this->tabControl1);
			this->splitPanel->Size = System::Drawing::Size(880, 383);
			this->splitPanel->SplitterDistance = 196;
			this->splitPanel->SplitterWidth = 10;
			this->splitPanel->TabIndex = 11;
			this->splitPanel->SplitterMoved += gcnew System::Windows::Forms::SplitterEventHandler(this, &CosmicWind::splitPanel_SplitterMoved);
			this->splitPanel->Click += gcnew System::EventHandler(this, &CosmicWind::bg_panel_Click);
			// 
			// tablelayoutpanel_top
			// 
			this->tablelayoutpanel_top->ColumnCount = 1;
			this->tablelayoutpanel_top->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle()));
			this->tablelayoutpanel_top->Controls->Add(this->tableLayoutPanel3, 0, 0);
			this->tablelayoutpanel_top->Controls->Add(this->panel_deviceslist, 0, 1);
			this->tablelayoutpanel_top->Dock = System::Windows::Forms::DockStyle::Fill;
			this->tablelayoutpanel_top->Location = System::Drawing::Point(0, 0);
			this->tablelayoutpanel_top->Name = L"tablelayoutpanel_top";
			this->tablelayoutpanel_top->RowCount = 2;
			this->tablelayoutpanel_top->RowStyles->Add((gcnew System::Windows::Forms::RowStyle()));
			this->tablelayoutpanel_top->RowStyles->Add((gcnew System::Windows::Forms::RowStyle()));
			this->tablelayoutpanel_top->Size = System::Drawing::Size(880, 196);
			this->tablelayoutpanel_top->TabIndex = 15;
			// 
			// tableLayoutPanel3
			// 
			this->tableLayoutPanel3->AutoSize = true;
			this->tableLayoutPanel3->AutoSizeMode = System::Windows::Forms::AutoSizeMode::GrowAndShrink;
			this->tableLayoutPanel3->ColumnCount = 2;
			this->tableLayoutPanel3->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle()));
			this->tableLayoutPanel3->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle()));
			this->tableLayoutPanel3->Controls->Add(this->panel_e2gg_l, 0, 0);
			this->tableLayoutPanel3->Controls->Add(this->panel_e2gg_r, 1, 0);
			this->tableLayoutPanel3->Dock = System::Windows::Forms::DockStyle::Fill;
			this->tableLayoutPanel3->Location = System::Drawing::Point(3, 10);
			this->tableLayoutPanel3->Margin = System::Windows::Forms::Padding(3, 10, 3, 3);
			this->tableLayoutPanel3->Name = L"tableLayoutPanel3";
			this->tableLayoutPanel3->RowCount = 1;
			this->tableLayoutPanel3->RowStyles->Add((gcnew System::Windows::Forms::RowStyle()));
			this->tableLayoutPanel3->Size = System::Drawing::Size(874, 55);
			this->tableLayoutPanel3->TabIndex = 17;
			// 
			// panel_e2gg_l
			// 
			this->panel_e2gg_l->AutoSize = true;
			this->panel_e2gg_l->AutoSizeMode = System::Windows::Forms::AutoSizeMode::GrowAndShrink;
			this->panel_e2gg_l->BackColor = System::Drawing::SystemColors::Window;
			this->panel_e2gg_l->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->panel_e2gg_l->Controls->Add(this->pictureBox3);
			this->panel_e2gg_l->Controls->Add(this->tableLayoutPanel1);
			this->panel_e2gg_l->Dock = System::Windows::Forms::DockStyle::Fill;
			this->panel_e2gg_l->Location = System::Drawing::Point(3, 3);
			this->panel_e2gg_l->Name = L"panel_e2gg_l";
			this->panel_e2gg_l->Size = System::Drawing::Size(357, 49);
			this->panel_e2gg_l->TabIndex = 17;
			// 
			// pictureBox3
			// 
			this->pictureBox3->Anchor = System::Windows::Forms::AnchorStyles::Left;
			this->pictureBox3->BackColor = System::Drawing::SystemColors::Window;
			this->pictureBox3->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"pictureBox3.Image")));
			this->pictureBox3->Location = System::Drawing::Point(8, 5);
			this->pictureBox3->Name = L"pictureBox3";
			this->pictureBox3->Size = System::Drawing::Size(53, 36);
			this->pictureBox3->TabIndex = 13;
			this->pictureBox3->TabStop = false;
			// 
			// tableLayoutPanel1
			// 
			this->tableLayoutPanel1->AutoSizeMode = System::Windows::Forms::AutoSizeMode::GrowAndShrink;
			this->tableLayoutPanel1->BackColor = System::Drawing::SystemColors::Window;
			this->tableLayoutPanel1->ColumnCount = 2;
			this->tableLayoutPanel1->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle()));
			this->tableLayoutPanel1->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle()));
			this->tableLayoutPanel1->Controls->Add(this->label3, 0, 1);
			this->tableLayoutPanel1->Controls->Add(this->label2, 0, 0);
			this->tableLayoutPanel1->Controls->Add(this->textbox_poolurl, 1, 0);
			this->tableLayoutPanel1->Controls->Add(this->textbox_difficulty, 1, 1);
			this->tableLayoutPanel1->GrowStyle = System::Windows::Forms::TableLayoutPanelGrowStyle::FixedSize;
			this->tableLayoutPanel1->Location = System::Drawing::Point(64, 5);
			this->tableLayoutPanel1->Name = L"tableLayoutPanel1";
			this->tableLayoutPanel1->RowCount = 2;
			this->tableLayoutPanel1->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent, 50)));
			this->tableLayoutPanel1->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent, 50)));
			this->tableLayoutPanel1->Size = System::Drawing::Size(288, 39);
			this->tableLayoutPanel1->TabIndex = 12;
			// 
			// textbox_poolurl
			// 
			this->textbox_poolurl->AcceptsTab = true;
			this->textbox_poolurl->Anchor = System::Windows::Forms::AnchorStyles::Left;
			this->textbox_poolurl->BackColor = System::Drawing::SystemColors::Window;
			this->textbox_poolurl->BorderStyle = System::Windows::Forms::BorderStyle::None;
			this->textbox_poolurl->Location = System::Drawing::Point(80, 3);
			this->textbox_poolurl->MaxLength = 66;
			this->textbox_poolurl->Name = L"textbox_poolurl";
			this->textbox_poolurl->ReadOnly = true;
			this->textbox_poolurl->Size = System::Drawing::Size(215, 13);
			this->textbox_poolurl->TabIndex = 15;
			this->textbox_poolurl->Text = L"--";
			this->textbox_poolurl->WordWrap = false;
			// 
			// textbox_difficulty
			// 
			this->textbox_difficulty->AcceptsTab = true;
			this->textbox_difficulty->Anchor = System::Windows::Forms::AnchorStyles::Left;
			this->textbox_difficulty->BackColor = System::Drawing::SystemColors::Window;
			this->textbox_difficulty->BorderStyle = System::Windows::Forms::BorderStyle::None;
			this->textbox_difficulty->Location = System::Drawing::Point(80, 22);
			this->textbox_difficulty->MaxLength = 66;
			this->textbox_difficulty->Name = L"textbox_difficulty";
			this->textbox_difficulty->ReadOnly = true;
			this->textbox_difficulty->Size = System::Drawing::Size(215, 13);
			this->textbox_difficulty->TabIndex = 14;
			this->textbox_difficulty->Text = L"--";
			this->textbox_difficulty->WordWrap = false;
			// 
			// panel_e2gg_r
			// 
			this->panel_e2gg_r->AutoSize = true;
			this->panel_e2gg_r->AutoSizeMode = System::Windows::Forms::AutoSizeMode::GrowAndShrink;
			this->panel_e2gg_r->BackColor = System::Drawing::SystemColors::Window;
			this->panel_e2gg_r->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->panel_e2gg_r->Controls->Add(this->pictureBox2);
			this->panel_e2gg_r->Controls->Add(this->tableLayoutPanel2);
			this->panel_e2gg_r->Dock = System::Windows::Forms::DockStyle::Fill;
			this->panel_e2gg_r->Location = System::Drawing::Point(366, 3);
			this->panel_e2gg_r->Name = L"panel_e2gg_r";
			this->panel_e2gg_r->Size = System::Drawing::Size(505, 49);
			this->panel_e2gg_r->TabIndex = 18;
			// 
			// pictureBox2
			// 
			this->pictureBox2->BackColor = System::Drawing::SystemColors::Window;
			this->pictureBox2->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"pictureBox2.Image")));
			this->pictureBox2->Location = System::Drawing::Point(3, 5);
			this->pictureBox2->Name = L"pictureBox2";
			this->pictureBox2->Size = System::Drawing::Size(62, 35);
			this->pictureBox2->TabIndex = 3;
			this->pictureBox2->TabStop = false;
			// 
			// tableLayoutPanel2
			// 
			this->tableLayoutPanel2->AutoSizeMode = System::Windows::Forms::AutoSizeMode::GrowAndShrink;
			this->tableLayoutPanel2->BackColor = System::Drawing::SystemColors::Window;
			this->tableLayoutPanel2->ColumnCount = 2;
			this->tableLayoutPanel2->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle()));
			this->tableLayoutPanel2->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle()));
			this->tableLayoutPanel2->Controls->Add(this->textbox_challenge, 1, 1);
			this->tableLayoutPanel2->Controls->Add(this->textbox_ethaddress, 1, 0);
			this->tableLayoutPanel2->Controls->Add(this->labelChallenge, 0, 1);
			this->tableLayoutPanel2->Controls->Add(this->label1, 0, 0);
			this->tableLayoutPanel2->GrowStyle = System::Windows::Forms::TableLayoutPanelGrowStyle::FixedSize;
			this->tableLayoutPanel2->Location = System::Drawing::Point(66, 5);
			this->tableLayoutPanel2->Name = L"tableLayoutPanel2";
			this->tableLayoutPanel2->RowCount = 2;
			this->tableLayoutPanel2->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent, 50)));
			this->tableLayoutPanel2->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent, 50)));
			this->tableLayoutPanel2->Size = System::Drawing::Size(432, 39);
			this->tableLayoutPanel2->TabIndex = 14;
			// 
			// textbox_challenge
			// 
			this->textbox_challenge->AcceptsTab = true;
			this->textbox_challenge->Anchor = System::Windows::Forms::AnchorStyles::Left;
			this->textbox_challenge->BackColor = System::Drawing::SystemColors::Window;
			this->textbox_challenge->BorderStyle = System::Windows::Forms::BorderStyle::None;
			this->textbox_challenge->Location = System::Drawing::Point(96, 22);
			this->textbox_challenge->MaxLength = 66;
			this->textbox_challenge->Name = L"textbox_challenge";
			this->textbox_challenge->ReadOnly = true;
			this->textbox_challenge->Size = System::Drawing::Size(333, 13);
			this->textbox_challenge->TabIndex = 13;
			this->textbox_challenge->Text = L"--";
			this->textbox_challenge->WordWrap = false;
			// 
			// textbox_ethaddress
			// 
			this->textbox_ethaddress->AcceptsTab = true;
			this->textbox_ethaddress->Anchor = System::Windows::Forms::AnchorStyles::Left;
			this->textbox_ethaddress->BackColor = System::Drawing::SystemColors::Window;
			this->textbox_ethaddress->BorderStyle = System::Windows::Forms::BorderStyle::None;
			this->textbox_ethaddress->Location = System::Drawing::Point(96, 3);
			this->textbox_ethaddress->MaxLength = 66;
			this->textbox_ethaddress->Name = L"textbox_ethaddress";
			this->textbox_ethaddress->ReadOnly = true;
			this->textbox_ethaddress->Size = System::Drawing::Size(333, 13);
			this->textbox_ethaddress->TabIndex = 4;
			this->textbox_ethaddress->Text = L"--";
			this->textbox_ethaddress->WordWrap = false;
			// 
			// panel_deviceslist
			// 
			this->panel_deviceslist->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
				| System::Windows::Forms::AnchorStyles::Left)
				| System::Windows::Forms::AnchorStyles::Right));
			this->panel_deviceslist->AutoSize = true;
			this->panel_deviceslist->AutoSizeMode = System::Windows::Forms::AutoSizeMode::GrowAndShrink;
			this->panel_deviceslist->BackColor = System::Drawing::SystemColors::ButtonFace;
			this->panel_deviceslist->Location = System::Drawing::Point(3, 71);
			this->panel_deviceslist->Name = L"panel_deviceslist";
			this->panel_deviceslist->Padding = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->panel_deviceslist->Size = System::Drawing::Size(874, 122);
			this->panel_deviceslist->TabIndex = 18;
			// 
			// CosmicWind
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->AutoSizeMode = System::Windows::Forms::AutoSizeMode::GrowAndShrink;
			this->BackColor = System::Drawing::SystemColors::ButtonFace;
			this->ClientSize = System::Drawing::Size(880, 519);
			this->Controls->Add(this->splitPanel);
			this->Controls->Add(this->lowerPanel);
			this->Controls->Add(this->menuBar);
			this->DoubleBuffered = true;
			this->Icon = (cli::safe_cast<System::Drawing::Icon^>(resources->GetObject(L"$this.Icon")));
			this->KeyPreview = true;
			this->MainMenuStrip = this->menuBar;
			this->MinimumSize = System::Drawing::Size(896, 558);
			this->Name = L"CosmicWind";
			this->SizeGripStyle = System::Windows::Forms::SizeGripStyle::Show;
			this->StartPosition = System::Windows::Forms::FormStartPosition::CenterScreen;
			this->Text = L"COSMiC v4.1.5 Dev TEST";
			this->FormClosing += gcnew System::Windows::Forms::FormClosingEventHandler(this, &CosmicWind::CosmicWind_FormClosing);
			this->FormClosed += gcnew System::Windows::Forms::FormClosedEventHandler(this, &CosmicWind::CosmicWind_FormClosed);
			this->Load += gcnew System::EventHandler(this, &CosmicWind::CosmicWind_Load);
			this->Shown += gcnew System::EventHandler(this, &CosmicWind::CosmicWind_Shown);
			this->ResizeEnd += gcnew System::EventHandler(this, &CosmicWind::CosmicWind_ResizeEnd);
			this->Click += gcnew System::EventHandler(this, &CosmicWind::CosmicWind_Click);
			this->Resize += gcnew System::EventHandler(this, &CosmicWind::CosmicWind_Resize);
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->EndInit();
			this->panel2->ResumeLayout(false);
			this->panel2->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->nud_numthreads))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->nud_updatespeed))->EndInit();
			this->contextMenu_eventsBox->ResumeLayout(false);
			this->menuBar->ResumeLayout(false);
			this->menuBar->PerformLayout();
			this->trayContextMenu->ResumeLayout(false);
			this->txViewContextMenu->ResumeLayout(false);
			this->contextMenu_gpu->ResumeLayout(false);
			this->lowerPanel->ResumeLayout(false);
			this->lowerPanel->PerformLayout();
			this->statusBar->ResumeLayout(false);
			this->statusBar->PerformLayout();
			this->totalsGroupBox->ResumeLayout(false);
			this->totalsGroupBox->PerformLayout();
			this->tabControl1->ResumeLayout(false);
			this->eventsPage->ResumeLayout(false);
			this->tabSolns->ResumeLayout(false);
			this->tabCpu->ResumeLayout(false);
			this->splitPanel->Panel1->ResumeLayout(false);
			this->splitPanel->Panel2->ResumeLayout(false);
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->splitPanel))->EndInit();
			this->splitPanel->ResumeLayout(false);
			this->tablelayoutpanel_top->ResumeLayout(false);
			this->tablelayoutpanel_top->PerformLayout();
			this->tableLayoutPanel3->ResumeLayout(false);
			this->tableLayoutPanel3->PerformLayout();
			this->panel_e2gg_l->ResumeLayout(false);
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox3))->EndInit();
			this->tableLayoutPanel1->ResumeLayout(false);
			this->tableLayoutPanel1->PerformLayout();
			this->panel_e2gg_r->ResumeLayout(false);
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox2))->EndInit();
			this->tableLayoutPanel2->ResumeLayout(false);
			this->tableLayoutPanel2->PerformLayout();
			this->ResumeLayout(false);
			this->PerformLayout();

			//
			this->DevicesView = gcnew SmoothListView();  // instantiate the custom listview		[MOVEME] ?
			//	DevicesView->LostFocus += gcnew System::EventHandler(this, &CosmicWind::DevicesView_LostFocus);
			//	DevicesView->GotFocus += gcnew System::EventHandler(this, &CosmicWind::DevicesView_GotFocus);
			//	DevicesView->Click += gcnew System::EventHandler(this, &CosmicWind::DevicesView_Click);

		}

#pragma endregion
// [MOVEME] ?


// TIMER1_NETERRORS_CHECK(): helper func for `timer1`
private: System::Boolean Timer1_NetErrors_Check(void)
{ // checks for and handles miner pausing when the network/pool can't be contacted. Pause on many consecutive network errors.
  // (todo)  Move where this is called, make user-configurable.

	if (stat_net_consecutive_errors >= DEFAULT_PAUSE_ON_NETERRORS) // <-----
	{
		if (!gNetPause)  // if not paused
		{  gNetPause = true;  // stop launching mining kernels
			statusbar_minerstate->Text = "Status: Waiting for Network";  // status bar update
			domesg_verb("Paused mining due to network errors. Will keep trying.", true, V_NORM);  // event, minor err
			return true;  // now paused to save power
		} else  /* if netpaused, resume: */
		{  gNetPause = false;
			domesg_verb("Pool contacted successfully. Resuming mining!", true, V_NORM);
			stat_net_consecutive_errors = 0;
			return false;  /* no longer paused */
		}
	}
	return gNetPause;  // bool
}


// [TODO}: roll the Hardware Monitoring columns updating into this func? or split it into two.
private: System::Void timer1_updateListedDeviceInfo(const unsigned short cosmicDevNo, const unsigned short apiDeviceNo)
{
	double hashrate{ 0.0 };
	uint64_t hashcount{ 0 };
	Solvers[cosmicDevNo]->GetSolverStatus(&hashrate, &hashcount);	//anything else?

// condense with the calling function. [TODO]
	unsigned int time_scratch_secs{ 0 }, time_scratch_mins{ 0 }, time_scratch_hours{ 0 };
	double doub_hashesPerSec{ 0 }, lcl_totalhashrate_d{ 0 }, lcl_totalpwr_d{ 0 }, lcl_totalPowerDraw{ 0 };  // d for double
	StringBuilder^ sb_work = gcnew StringBuilder("", 200);  // s.b. for managed string. 200 characters max
	uint64_t lcl_totalsolutions = 0;
	
	time_scratch_mins = time_scratch_secs /* = time_scratch_hours */ = 0;  // reset time scratchvars to zero

	/* was: `gCudaDeviceEnabled[apiDeviceNo] ` */
	if (Solvers[cosmicDevNo]->enabled == false) {
		DevicesView->Items[cosmicDevNo]->SubItems[9]->Text = "Disabled";	// todo: once
		DevicesView->Items[cosmicDevNo]->SubItems[10]->Text = "-";			// null usage %
		return;		//continue;
	}

// - if Network/Pool outage: -
	if (gNetPause) {  // applies to any enabled device:
		DevicesView->Items[cosmicDevNo]->SubItems[9]->Text = "Waiting";  // only once
		DevicesView->Items[cosmicDevNo]->SubItems[2]->Text = "0 MH/s";
		ClearDevicesListSubItem(cosmicDevNo, 2);
		ClearDevicesListSubItem(cosmicDevNo, 4);
		ClearDevicesListSubItem(cosmicDevNo, 5);
		ClearDevicesListSubItem(cosmicDevNo, 6);
		return;		//continue;		(to next iteration of loop in calling func)
	}
	// if mining and not paused, add this device's hashrate and solution count to the total
	//if (gCuda_Pause[apiDeviceNo] == DEVICE_PAUSE_NONE && Solvers[cosmicDevNo]->solver_status == SolverStatus::Solving) {
	if (Solvers[cosmicDevNo]->pause == PauseType::NotPaused && Solvers[cosmicDevNo]->solver_status == SolverStatus::Solving) {
		lcl_totalhashrate_d += Solvers[cosmicDevNo]->hash_rate;		// add this device's hashrate to system total
		lcl_totalsolutions += gNum_SolutionCount[apiDeviceNo];		// and the number of sol'ns found
	}
	

// -- Check for a fault condition: --
	if (Solvers[cosmicDevNo]->solver_status == SolverStatus::Solving)  //mining, ready, starting or idle
	{ // [old]: if (status < DEVICE_STATUS_FAULT)
		if (!gSolving[cosmicDevNo])  // solver CPU thread (CUDA device host thread) is stopping:
			DevicesView->Items[cosmicDevNo]->SubItems[9]->Text = "Finishing";	// once

		if(Solvers[cosmicDevNo]->pause == PauseType::NotPaused)		//if (gCuda_Pause[apiDeviceNo] == DEVICE_PAUSE_NONE)
		{ // device is not in any pause condition:
			if (Solvers[cosmicDevNo]->solve_time > 0)  /* don't div by 0: */
				doub_hashesPerSec = static_cast<double>(Solvers[cosmicDevNo]->hash_count) / Solvers[cosmicDevNo]->solve_time;
		//	Solvers[cosmicDevNo]->hash_rate = doub_hashesPerSec / 1000000;  // <- store as megahashes/sec. (Set in the Solver itself after each do-while{} run) [WIP] / [TODO] <---
			if (Solvers[cosmicDevNo]->hash_rate > 0) {
				sb_work->Clear()->Append(Solvers[cosmicDevNo]->hash_rate.ToString("0.00"))->Append(" MH/s");	//out to 2 decimal places	[FIXME]? ensure that the last digit is rounded. <--
				DevicesView->Items[cosmicDevNo]->SubItems[2]->Text = sb_work->ToString();						//update hashrate display text
				DevicesView->Items[cosmicDevNo]->SubItems[9]->Text = "Mining";
			} else {
				DevicesView->Items[cosmicDevNo]->SubItems[2]->Text = "? MH/s";		//
				DevicesView->Items[cosmicDevNo]->SubItems[9]->Text = "Not Mining";	// consider reorganizing this part
			}
		} else { /* the device is paused: */
			sb_work->Clear()->Append("Paused (");	/* pause message with reason in the Status column */
			if (Solvers[cosmicDevNo]->pause == PauseType::GPUTempAlarm) { sb_work->Append("Temp Alarm)"); }		/* kernel launching on    */
			 else if (Solvers[cosmicDevNo]->pause == PauseType::FanAlarm ) { sb_work->Append("Fan Alarm)"); }	/* this device will pause */
			DevicesView->Items[cosmicDevNo]->SubItems[9]->Text = sb_work->ToString();  // from stringbuilder
			DevicesView->Items[cosmicDevNo]->SubItems[2]->Text = "0 MH/s";
		}
	}
	else if (Solvers[cosmicDevNo]->solver_status == SolverStatus::Starting) {
		this->DevicesView->Items[cosmicDevNo]->SubItems[9]->Text = "Starting";
		this->DevicesView->Items[cosmicDevNo]->SubItems[2]->Text = "0 MH/s";
		//..
	}
	else if (Solvers[cosmicDevNo]->solver_status == SolverStatus::NotSolving) {
		this->DevicesView->Items[cosmicDevNo]->SubItems[9]->Text = "Not Mining";
		this->DevicesView->Items[cosmicDevNo]->SubItems[2]->Text = "0 MH/s";
	}
	else if (Solvers[cosmicDevNo]->solver_status == SolverStatus::Resuming) {
		this->DevicesView->Items[cosmicDevNo]->SubItems[9]->Text = "Resuming";
		this->DevicesView->Items[cosmicDevNo]->SubItems[2]->Text = "0 MH/s";
	}
	else if (Solvers[cosmicDevNo]->solver_status == SolverStatus::Ready) {
		this->DevicesView->Items[cosmicDevNo]->SubItems[9]->Text = "Ready";
		this->DevicesView->Items[cosmicDevNo]->SubItems[2]->Text = "0 MH/s";
	}
	else if (Solvers[cosmicDevNo]->solver_status == SolverStatus::Error) {
		this->DevicesView->Items[cosmicDevNo]->SubItems[9]->Text = "Error";
		this->DevicesView->Items[cosmicDevNo]->SubItems[2]->Text = "0 MH/s";
	}
	else if (Solvers[cosmicDevNo]->solver_status == SolverStatus::DeviceError) //<-- 20190804: revised, was loop iterator `i`, should be cuda device#
	{ /* device error */
		gSolving[cosmicDevNo] = false;  // matching solver loop will stop
		this->DevicesView->Items[cosmicDevNo]->SubItems[9]->Text = "Device Error";
		this->DevicesView->Items[cosmicDevNo]->SubItems[2]->Text = "0 MH/s";
		domesg_verb("Stopping CUDA Device #" + std::to_string(apiDeviceNo) + " (" + gpuName[apiDeviceNo] + ") because of a device error.", true, V_LESS);
		gStopCondition = true;  // trigger miner stop because in testing, a device failure generally interrupts mining on the other devices (such as unstable OC.) [TESTME]
	}
	else if (Solvers[cosmicDevNo]->solver_status == SolverStatus::Null) {
		this->DevicesView->Items[cosmicDevNo]->SubItems[9]->Text = "Null";
		this->DevicesView->Items[cosmicDevNo]->SubItems[2]->Text = "0 MH/s";
	}
	else if (Solvers[cosmicDevNo]->solver_status == SolverStatus::WaitingForNetwork) {
		this->DevicesView->Items[cosmicDevNo]->SubItems[9]->Text = "Waiting for Network";
		this->DevicesView->Items[cosmicDevNo]->SubItems[2]->Text = "0 MH/s";
	}
	else if (Solvers[cosmicDevNo]->solver_status == SolverStatus::UpdatingParams) {
		this->DevicesView->Items[cosmicDevNo]->SubItems[9]->Text = "Updating Parameters";
		this->DevicesView->Items[cosmicDevNo]->SubItems[2]->Text = "0 MH/s";
	}
	else if (Solvers[cosmicDevNo]->solver_status == SolverStatus::DeviceNotInited) {
		this->DevicesView->Items[cosmicDevNo]->SubItems[9]->Text = "Waiting for Device";	//<--
		this->DevicesView->Items[cosmicDevNo]->SubItems[2]->Text = "0 MH/s";
	}
	else { /* stopping, because... */
		// [todo]: detect erroneous hashrates. usually unnecessary because we catch a memory error first.
		gSolving[cosmicDevNo] = false;  // matching do-while loop will stop
		this->DevicesView->Items[cosmicDevNo]->SubItems[9]->Text = "Unknown";
		this->DevicesView->Items[cosmicDevNo]->SubItems[2]->Text = "0 MH/s";
		domesg_verb("Stopping Solver #" + std::to_string(cosmicDevNo) + " (CUDA device #" + std::to_string(apiDeviceNo) + "), unknown error.", true, V_LESS);  // err
		LOG_F(ERROR, "Stopping Solver# %u, unknown error!" BUGIFHAPPENED, cosmicDevNo);
		Solvers[cosmicDevNo]->solver_status = SolverStatus::Error;
		Solvers[apiDeviceNo]->device_status = DeviceStatus::Fault;
		gStopCondition = true;  // trigger miner stop	<------------------ set Stop Condition on any other statuses? <-----
	}

	// - Solutions Count  (invalid or rejected shares in parentheses) -
	sb_work->Clear()->Append(static_cast<unsigned long long>(gNum_SolutionCount[cosmicDevNo]));
	sb_work->Append(" (")->Append(static_cast<unsigned long long>(gNum_InvalidSolutionCount[cosmicDevNo]))->Append(" invalid) ");
	DevicesView->Items[cosmicDevNo]->SubItems[3]->Text = sb_work->ToString();

	// - Solve Time -
	// [todo]:	use a timespan or similar structure and a function that CUDA host code can update/reset via function call.
	time_scratch_hours = 0;																// <-- [todo].
	time_scratch_secs = static_cast<unsigned int>(Solvers[cosmicDevNo]->solve_time);	// <-- [wip]: time display improvement
	time_scratch_mins = time_scratch_secs / 60;
	time_scratch_secs -= time_scratch_mins * 60;  // hehe

	// update the new simple time string under device's "solve time" column (TODO: add hours for solo mode & tough solves).
	sb_work->Clear()->Append(time_scratch_mins)->Append("m:")->Append(time_scratch_secs)->Append("s");  // mins:secs (TODO: hrs)
	DevicesView->Items[cosmicDevNo]->SubItems[8]->Text = sb_work->ToString();

	// -- Hardware Monitor readings) --
	//  - Device Utilization (%) -
	sb_work->Clear()->Append(gWatchQat_Devices[cosmicDevNo].utilization)->Append("%");
	if (gWatchQat_Devices[cosmicDevNo].utilization >= 0)  DevicesView->Items[cosmicDevNo]->SubItems[10]->Text = sb_work->ToString();  // the utilization %.

	// if Hardware Monitoring is enabled this device:
	if (gWatchQat_Devices[cosmicDevNo].watchqat_enabled && !gStopCondition)	/*<---- apiDevNo ?	[WIP / FIXME] */
	{  // TODO: Fahrenheit option too? available in GPU Summary form as F.
		//StringBuilder^ sbf = gcnew StringBuilder( (int)gWatchQat_Devices[thisDevice].fanspeed_rpm );
		sb_work->Clear()->Append(gWatchQat_Devices[cosmicDevNo].fanspeed_rpm.ToString())->Append(" RPM");
		DevicesView->Items[cosmicDevNo]->SubItems[6]->Text = sb_work->ToString();  //gWatchQat_Devices[thisDevice].fanspeed_rpm.ToString() + " RPM";  // write RPMs
		sb_work->Clear()->Append(gWatchQat_Devices[cosmicDevNo].gputemp)->Append(" °C");
		DevicesView->Items[cosmicDevNo]->SubItems[4]->Text = sb_work->ToString();  // write GPU temp, from stringbuilder

		DevicesView->Items[cosmicDevNo]->SubItems[4]->BackColor = GetWarningColor(cosmicDevNo, 0);  // get appropriate color. args: device index, 0 scratch param. WIP: do only once
		// white text if 'warning'-colored BG, so user can easily read the number:
		DevicesView->Items[cosmicDevNo]->SubItems[4]->ForeColor = gWatchQat_Devices[cosmicDevNo].gputemp > 84 ? Color::White : Color::Black;
		
// These params yield 1 decimal place of precision. Round UP. Ex. 0.5 becomes 1, which apparently is not the default behavior.
// const double display_pwrdraw = Math::Round(gWatchQat_Devices[thisDevice].powerdraw_w, 1, MidpointRounding::AwayFromZero);

		//if paused due to device health alarm, powerdraw_w should be -1. not mining on this device, so don't add its power usage reading to the total.
		if (gWatchQat_Devices[cosmicDevNo].powerdraw_w > 0 && Solvers[cosmicDevNo]->solver_status == SolverStatus::Solving && 
			Solvers[cosmicDevNo]->pause == PauseType::NotPaused) {		// [FIXME]: simplify genericSolver access. <--
			// [todo]: do AppendFormat() and use a StringBuilder constructor which takes a decimal type? <--
			sb_work->Clear()->Append(Math::Round(gWatchQat_Devices[cosmicDevNo].powerdraw_w, 1, MidpointRounding::AwayFromZero).ToString("0.0"))->Append(" W");
			DevicesView->Items[cosmicDevNo]->SubItems[5]->Text = sb_work->ToString();	//update device's power usage in watts in devices list
			lcl_totalPowerDraw += gWatchQat_Devices[cosmicDevNo].powerdraw_w;			//add device's power usage to total power usage estimate
		}
		else  ClearDevicesListSubItem(cosmicDevNo, 5);	//if no power reading
	}
	else { //hardware monitoring disabled or Stop Condition:
		ClearDevicesListSubItem(cosmicDevNo, 4);
		ClearDevicesListSubItem(cosmicDevNo, 5);
		ClearDevicesListSubItem(cosmicDevNo, 6);
	}

}


// FIXME: Finish refactoring this function.
private: System::Void timer1_Tick(System::Object^ sender, System::EventArgs^ e)
{ // function runs when timer1 has ticked (its interval is up).
	timer1->Stop();  // stop the timer while doing our work.
 // WIP: breaking this up into functions and optimizing.
#ifdef TEST_BEHAVIOR
	Stopwatch^ sw_timer1 = Stopwatch::StartNew();	// Profiling
	const bool profile_timer1 = checkDoEveryCalls(Timings::profTimer1);
#endif

	// TODO: move mostly-static elements elsewhere.
	// TODO: timespan structure instead, push values from CUDA C host code?
	// be sure relevant scratch vars cleared each loop iteration (device in list).
	StringBuilder^ sb_work = gcnew StringBuilder("", 200);  // s.b. for managed string. 200 characters max
	unsigned int time_scratch_secs{ 0 }, time_scratch_mins{ 0 }, time_scratch_hours{ 0 };
	double dScratch{ 0 }, lcl_totalhashrate_d{ 0 }, lcl_totalpwr_d{ 0 };  // d for double
	uint64_t lcl_totalsolutions = 0;
	double lcl_totalPowerDraw = 0;

	if (gStopCondition /* && gCudaSolving*/) {
		if (gVerbosity > V_NORM)  printf("Timer1: Stop condition. Stopping mining... \n");
		domesg_verb("CUDA Device fault detected: Mining has stopped. Please check devices and intensity settings, then re-launch COSMiC.", true, V_LESS);

		start1->Enabled = false;
		start1->Text = "Start Mining!";
		combobox_modeselect->Enabled = true;
		timer_enableminingbutton->Enabled = true;
		statusbar_minerstate->Text = "Status: Stopping...";
		NetworkBGWorker->CancelAsync();					// stop network worker

		gCudaSolving = false;  // any running CUDA solver threads should stop now
		ClearEventsQueue();
		ClearSolutionsQueue();

//	leaving gStopCondition as `true`. Start button will check this and
//	inform user to restart the miner, if a GPU(s) become unstable.
		return;
	}
//
// - Status Bar, Mining Time, Network Pause -
//
	if ( !Timer1_NetErrors_Check() && gCudaSolving)  /* contains consecutive network errors / pause check					*/
	{												 /* not paused. append elapsed mining time, status, mode to status bar	*/
		// new: using stringbuilder for speed
		sb_work->Clear()->Append(sw_miningTime->Elapsed.Days)->Append("d:")->Append(sw_miningTime->Elapsed.Hours)->Append("h:")->
			Append(sw_miningTime->Elapsed.Minutes.ToString("00"))->Append("m:")->Append(sw_miningTime->Elapsed.Seconds.ToString("00"))->Append("s");
		statusbar_elapsedTime->Text = sb_work->ToString();	// from stringbuilder
		if (gSoloMiningMode) {								// valid TXview Item#: 0 or higher.
			if (gTxView_WaitForTx >= NOT_WAITING_FOR_TX)
				statusbar_minerstate->Text = "Status: Waiting for Transaction (Sol'n #" + gTxView_WaitForTx.ToString() + ")";	// network nonce?
			 else { statusbar_minerstate->Text = "Status: Mining (Solo Mode)"; }
		}
		else { statusbar_minerstate->Text = "Status: Mining (Pool Mode)"; }
	} // do elsewhere. thus rarely running this part ^

// - Status Bar, Challenge -
	if (gStopCondition)  statusbar_minerstate->Text = "Status: Error State";
	if (!gCudaSolving)  statusbar_minerstate->Text = "Status: Idle";
	else {
		// scratch_mstring = gcnew String(gStr_Challenge.c_str());  /*->Substring(0, 42) + "...";
		// todo: just have the textbox size hide the rest, whole Challenge shown when maximized. "..." if not fully shown.
		if ( gMiningParameters.challenge_str.length() == 66 )  /* [TESTME]  appropriate length incl. "0x" hex specifier */
			textbox_challenge->Text = gcnew String( gMiningParameters.challenge_str.c_str() );	/* TESTME */
		 else { if (DEBUGMODE) { printf("- Bad challenge string in timer1_Tick(). \n");  /* textbox_challenge->Text = "?"; */ } }
	}
	
	textbox_difficulty->Text = gU64_DifficultyNo.ToString();  // update DIFFICULTY textbox

// - Updates to Devices List -
// check MAX_SOLVERS ? the listview should not contain more devices than supported <--
// [TODO]: consider the device type, check solver class instead? <---
	if (DevicesView->Items->Count > 0 && DevicesView->Items->Count < MAX_SOLVERS && gCudaSolving)	// <= MAX_SOLVERS ?	[WIP] / [FIXME] <---
	{ // iterating through the list items in order:
		for (unsigned short i = 0; i < this->DevicesView->Items->Count; ++i)
		{// "i": Devices List Row Number from 0 (`i` might coincide with devices' CUDA API numbering, might not.)
		 // `thisDevice` was managed type `Int16` <-- # of solvers supported in the ListView is 0-254, 0xFF=NULL device#! //use UINT8_MAX (0xFF) as NULL value! [WIP] <--
			unsigned short thisDevice = UINT8_MAX;
			if (Solvers[i]->device_type == DeviceType::Type_CUDA) {
				thisDevice = static_cast<unsigned short>(Solvers[i]->cuda_device->dev_no);
				if (thisDevice != UINT8_MAX && thisDevice < CUDA_MAX_DEVICES) { // 0xFF=init value, was -1
					if (Solvers[thisDevice]->enabled)	/* was `gCudaDeviceEnabled[thisDevice]` */
						timer1_updateListedDeviceInfo(i, thisDevice);	//<-- passing the COSMiC-ordered device# (`i`) and CUDA
																		// ...and CUDA or other API's ID # assigned to this GPU.
				}
				else {//error:
					Console::WriteLine("Timer1: bad cuda device index in devices listview item #{0}!", thisDevice);
					continue;
				}
			}
			//else if (Solvers[i]->device_type == DeviceType::? )	//other device types [TODO]. <--
			else {
				LOG_F(ERROR, "Timer1: bad device type in devices listview item# %u!", i);
				break;	// stop updating the listview! <--	[WIP].
			}
			
		}
	}

	if (gStopCondition)		 // 20190724: now don't do this per-device ;)
		ClearEventsQueue();	 // for events list box

// === write total hashrate and total solutions found to UI labels: ===
// if CPU mining... (threads numbered serially from 0.)
	double lcl_totalcpuhashrate_d = 0;
	sb_work->Clear()->Append(lcl_totalhashrate_d.ToString("0.00"))->Append(" MH/s");  // 2 decimal places displaying hashrate
	if (lcl_totalhashrate_d > 0)
	{ // add up the running CPU threads' hashrates (double type)
		for (unsigned int i = 0; i < gCpuSolverThreads; ++i)
			lcl_totalcpuhashrate_d += cpuThreadHashRates[i];
		//
		sb_work->Append("     ")->Append((lcl_totalcpuhashrate_d/1000).ToString("0.00"))->Append(" KH/s");  // append to s.b. w/ 2 decimals
	} // add CPU hashrate if CPU solvers are running. ^
	lbl_totalhashrate->Text = sb_work->ToString();  // from Stringbuilder contents to control's text
	
	// append singular or plural version, "share" for pool (meets min-share difficulty) or "solution" for solo (full contract solution).
	sb_work->Clear();
	if (gSoloMiningMode)
		lcl_totalsolutions == 1 ? sb_work->Append(lcl_totalsolutions)->Append(" Solution") :		 // solo mode
			sb_work->Append(lcl_totalsolutions)->Append(" Solutions");								 // "
	else
		lcl_totalsolutions == 1 ? sb_work->Append(lcl_totalsolutions)->Append(" Share") : sb_work->Append(lcl_totalsolutions)->Append(" Shares");  // pool mode
	//
	lbl_totalsols->Text = sb_work->ToString();  // update the label adjacent the start/stop button.
	// WIP: can we skip at point A above, some converting/comparing, if # of shares has not changed? <-- as in convert/compare

	if (lcl_totalPowerDraw > 0)
		lbl_totalpwr->Text = lcl_totalPowerDraw.ToString() + " W";

// === Notification Area / Updates to Systray icon's Summary tooltip ===
	if (this->WindowState == Windows::Forms::FormWindowState::Minimized && notifyIcon1->Visible)
	{ // update the notifyIcon (system tray icon)'s tooltip (hashrate/power summary), only if minimized to tray.
		sb_work->Clear();
		sb_work->Append(STR_CosmicVersion)->Append("\n")->Append(gCudaDevicesStarted)->Append(" CUDA GPUs solving @ ")->Append(lcl_totalhashrate_d.ToString("0.00"))->Append(" MH/s");
		if (gCpuSolverThreads > 0)
			sb_work->Append(" & ")->Append(gCpuSolverThreads)->Append(" CPUs ")->Append(" @ ")->Append(lcl_totalcpuhashrate_d.ToString("0.00"))->Append(" KH/s");
		//notifyIcon1->Text = STR_CosmicVersion + "\n" + gCudaDevicesStarted.ToString() + " CUDA GPUs solving @ " + lbl_totalhashrate->Text;  // show total hashrate this timer Tick
		notifyIcon1->Text = sb_work->ToString();
		//  else  notifyIcon1->Text = STR_CosmicVersion + "\Mining at " + lbl_totalhashrate->Text;  // show total hashrate this timer Tick (verbosity check was here)
	}

	HandleEventsListBox();  // ... adds items to the Events View, removes them from queue, and then scrolls the listbox down.

	if (gCudaSolving && gSoloMiningMode) {
		if (!bgWorker_SoloView->IsBusy && gTxView_WaitForTx == NOT_WAITING_FOR_TX)			/* <--- redundant WaitForTx check? see PopulateTxItem(). */ /* [WIP] */
			GetSolutionForTxView(); // was PassSolutionsToSoloTxWorker().	/* gTxViewItems will change	[WIP]: async stuff */
	}

// === Network Annunciator (NET) ===	[TODO]: icon instead of label <--
//	show or hide NET annunciator, shows bg worker state, color indicates latency. tooltip shows net stats!
	statusbar_anncNET->Enabled = NetworkBGWorker->IsBusy;		// grayed unless Network BGworker active
	statusbar_anncTXN->Enabled = bgWorker_SoloView->IsBusy;		// grayed unless TXHelper BGworker active
	Update_NET_Annunciator();									// Update NET annunciator's tooltip

#ifdef DEBUGGING_PERFORMANCE
	if (profile_timer1) {	//note: the # of milliseconds appears rounded down to the nearest ms
		if (DEBUGMODE)  printf("[prof.] timer1: %" PRIu64" ticks / ~%" PRIu64" ms\n", sw_timer1->ElapsedTicks, sw_timer1->ElapsedMilliseconds);
		sw_timer1->Stop(); }  // profiling every n runs of timer1's Tick() func
#endif
	timer1->Start();
}


// this backgroundWorker thread handles various network tasks, including mining parameters retrieval and submitting solutions.
// also responsible for getting Eth gas price, Tx count, and account balances (in solo mode.)
private: System::Void NetworkBGWorker_DoWork(System::Object^  sender, System::ComponentModel::DoWorkEventArgs^  e)
{
	loguru::set_thread_name("Network Thread");
	LOG_IF_F(INFO, gVerbosity >= V_NORM, "DoWork:  started.");	//<-
	domesg_verb("Network Worker started.", false, V_MORE);		//<-

	const bool compute_target{ menuitem_options_computetarg->Checked };	 // if unchecked, parse the remote target instead.
	const unsigned short maxtarg_exponent{ 234 };	// 0xBTC: 2^234 maxtarget.  [TODO]: make user-configurable <--
	unsigned short the_result{ 0 };
	
	//	miningParameters params {};	 // mining parameters will be updated, if needed, in recurring_network_tasks() each run of the below loop.

	while(gApplicationStatus != APPLICATION_STATUS_CLOSING && !gMiningStopStart)
	{
		try {
			LOG_IF_F(INFO, DEBUGMODE && DEBUG_NET_BGWORKER, "DoWork:	Loop iteration running");
			if (NetworkBGWorker->CancellationPending) {
				domesg_verb("Network BGworker cancelled ", true, V_MORE);
				e->Cancel = true;
				return;
			}

			// Pool Mode: always check for solutions each run.	Solo Mode: only check if no pending Tx on the network already.	[TESTME]
			if (!gSoloMiningMode || (gSoloMiningMode && gTxView_WaitForTx == NOT_WAITING_FOR_TX)) {
				LOG_IF_F(INFO, DEBUGMODE && DEBUG_NET_BGWORKER, "DoWork:	Checking for Solutions");
				if (!Comm_CheckForSolutions(&gMiningParameters)) {	// [FIXME]? <-- run this part immediately, then wait (per user timing settings) to update params
					//err	[WIP] <---
				} // [warning]: thread will not sleep if sol'ns waiting to be sent. make sure Comm_CheckForSolutions() never leaves any sol'ns in queue
			} else
				LOG_IF_F(INFO, HIGHVERBOSITY, "Tx already pending for this Challenge");

			// Perform the frequent network tasks. 0=OK, 1=err, stop. enum? [todo]:
			LOG_IF_F(INFO, DEBUGMODE && DEBUG_NET_BGWORKER, "DoWork:	Checking Mining Parameters...");
			the_result = recurring_network_tasks(compute_target, maxtarg_exponent, gMiningParameters, &gMiningParameters);	// [FIXME?]  Passing the global parameters ("old") by value, 
			if (the_result != 0) {																							// and referencing the global params to store any new params [TESTME].
				LOG_IF_F(WARNING, DEBUGMODE, "recurring_network_tasks():	result %u", the_result);	// [FIXME] <---
			//	return;
			}

			if (!gSoloMiningMode) { //=== pool mode: ===
				LOG_IF_F(INFO, DEBUGMODE && DEBUG_NET_BGWORKER, "Checking Interval:	%u ms	(default: %d)", gNetInterval, MIN_NETINTERVAL_POOL);
				if (gNetInterval < MIN_NETINTERVAL_POOL || gNetInterval > MAX_NETINTERVAL_POOL) { /* [WIP] */
					gNetInterval = DEFAULT_NETINTERVAL_POOL;
					LOG_F(WARNING, "DoWork:	Reset pool-mode network interval to default:	%d ms", DEFAULT_NETINTERVAL_POOL);	//
				}

				if (!GetSolnQueueContents())
					Thread::Sleep(gNetInterval);		// thread should not sleep if solutions waiting to be sent.	[testme]
			} else { //=== solo mode: ===
				LOG_IF_F(INFO, DEBUGMODE && DEBUG_NET_BGWORKER, "Checking Interval:	%u ms (default: %d)", gSoloNetInterval, MIN_NETINTERVAL_SOLO);
				if (gSoloNetInterval < MIN_NETINTERVAL_SOLO || gSoloNetInterval > MAX_NETINTERVAL_SOLO) { /* [WIP] */
					LOG_IF_F(WARNING, NORMALVERBOSITY, "DoWork:	Reset solo-mode network interval to default:	%d ms", DEFAULT_NETINTERVAL_SOLO);
					gSoloNetInterval = DEFAULT_NETINTERVAL_SOLO;
				}

				if (!GetSolnQueueContents())
					Thread::Sleep(gSoloNetInterval);	// thread should not sleep if solutions waiting to be sent.  [new, testme].
			}

			LOG_IF_F(INFO, DEBUGMODE && DEBUG_NET_BGWORKER, "DoWork:	Iteration Finished (took -- ticks/ms)");	//... % " PRIu64 " ticks | % d ms", 0, 0);
			continue;
		}
		catch (...) {
			LOG_F(ERROR, "Caught exception in NetworkBGWorker_DoWork():	N/A");	// [TODO]: log the type of exception.
			// ...
			//if mining, BGworker should re-start automatically.
		}
	} //while

}

private: System::Void NetworkBGWorker_RunWorkerCompleted(System::Object^ sender, System::ComponentModel::RunWorkerCompletedEventArgs^ e)
{  // Runs when the network BGworker completes.
   // TODO/FIXME: libcurl easy-cleanup if ending abnormally? <---
	if (e->Cancelled) {  /* Messages for the events */
#ifdef DEBUG_NET_BGWORKER
		domesg_verb("Network BGWorker cancelled. ", true, V_DEBUG);
		LOG_IF_F(INFO, DEBUGMODE, "Network BGWorker cancelled.");		//return; //<--
#endif
	}

	if (e->Error) {	/* network BGworker thread ended but was _not_ cancelled: */
		msclr::interop::marshal_context marshalctx;
		String^ anyErr_MStr = e->Error->ToString();
		domesg_verb("Network BGworker stop: " + marshalctx.marshal_as<std::string>(anyErr_MStr), true, V_DEBUG);
		domesg_verb("Network thread stopped unexpectedly. Restarting it in " +
			std::to_string(timer_net_worker_restart->Interval / 1000) + "s...", true, V_NORM);
		timer_net_worker_restart->Enabled = true; // restarts worker in ~4s
	} else {
		domesg_verb("Network BGworker ending neatly. ", false, V_DEBUG);
		return;
	}
}

private: System::Void CWind_StopMining( void )
{ // called when start/stop mining button clicked (stopping)
	LOG_IF_F(INFO, gVerbosity>=V_MORE, "Start/Stop Button pressed (stopping).");
	if (gSoloMiningMode) { /* solo mode only: erase skey in memory  */
		Erase_gSK();					// overwrite key
		timer_txview->Stop();			// <-- redundant
	}
// either mode:
	if (NetworkBGWorker->IsBusy)  { NetworkBGWorker->CancelAsync(); }  // stop the network BGworker if running
	gCudaDevicesStarted = 0;
	gMiningStopStart = true;	//<-- [FIXME / TODO].
	//
	// [WIP / TODO]: Call 'Stop()' member function of Solvers?
	//

	// stop mining on any enabled devices, and cleans up
	// (note: CUDA Solver CPU threads will stop when gCudaSolving==false.)
	//for (uint8_t i = 0; i < CUDA_MAX_DEVICES; ++i)
	unsigned short solvers_count = static_cast<unsigned short>(Solvers.size());
	for (unsigned short i = 0; i < solvers_count; ++i)
	{
		if (!Solvers[i]->enabled)	 /* if not enabled			*/ /* was: !gCudaDeviceEnabled[i] */
			continue;				 /* skip to next iteration	*/

		this->DevicesView->Items[i]->SubItems[9]->Text = "Stopping";
	//	StopMining( i );
		if (DEBUGMODE) { domesg_verb("Stopping CUDA GPU #" + std::to_string(i) + "...", true, V_DEBUG); }	//remove
		gSolving[i] = false;  // matching do-while loop should/will stop
	//	gCudaDeviceStatus[i] = DEVICE_STATUS_NULL;
		Solvers[i]->solver_status = SolverStatus::Null;	// [todo]: make sure this works device-generically.
															// solver #s (within COSMiC) are separate from CUDA device #'s.
	//	this->DevicesView->Items[i]->SubItems[9]->Text = "Idle";	//<-- this should be set after solver shutdown. [TODO / FIXME].
	}	
	
	ResetCoreParams( &gMiningParameters );				 // see network.cpp. for now <-- [WIP]
	// ...
	ClearEventsQueue();				 // for events list box
	ClearSolutionsQueue();
	//clearTxViewItemsData();		 // TxView items are cleared when Solo mining is next started, so user can observe their
									 // results after stopping.  [WIP]: clear items' data but leave listview items? <--
	start1->Enabled = false;
	start1->Text = "Stopping...";	 // start/stop button text
	this->Text = STR_CosmicVersion;  // title bar text
	statusbar_elapsedTime->Text = STR_CosmicVersion;		// version # in right corner of status bar when not mining
	statusbar_elapsedTime->DisplayStyle = ToolStripItemDisplayStyle::Text;	// hide clock icon by time
	timer_enableminingbutton->Enabled = true;				// will re-enable the Start/Stop button momentarily
	statusbar_minerstate->Text = "Status: Cleaning up...";
	combobox_modeselect->Enabled = true;					// undim mode select control
	//
	gCudaSolving = false;									// GPU solvers host threads will stop automatically

//	if (gCpuSolverThreads) { button_cpumine_startstop->PerformClick(); }		  // make sure CPU mining stops when main start/stop button is clicked <-
	domesg_verb("NOT clicking button_cpumine_startstop [WIP]. ", true, V_DEBUG);  // <- [WIP / FIXME] <--
//
}


// [TODO]:  this function can be condensed significantly, lots of old code
// [TODO]:  finish option to (neatly) start GPU & CPU mining at the same time.
private: System::Void start1_Click(System::Object^ sender, System::EventArgs^ e)
{ // when the Start/Stop Mining button is pressed:
	tabControl1->SelectTab(0);  // bring Events View tab to front
	tabControl1->Refresh();		// draw it before we work in the main thread.

	unsigned short maxtarg_exponent{ 234 };		// maxtarg = 2^n  (0xbitcoin uses 2^234) <----- Make user-configurable! [TODO]
	unsigned short devicesToStart{ 0 };
	gTxView_WaitForTx = NOT_WAITING_FOR_TX;		// reset pause state.  [moveme]
	gNetPause = false;							// [MOVEME].

	if (!gCudaSolving)	// not currently mining when clicked
	{
		// see which devices we're starting
		for (unsigned short d = 0; d < DevicesView->Items->Count; ++d)
		{	// was: ` gCudaDeviceEnabled[d] `.
			if (Solvers[d]->enabled)  { ++devicesToStart; }		// count devices enabled (that we expect to Start)
			if (gWatchQat_Devices[d].watchqat_enabled) {
				// 
				// ... [WIP].
			}

		}

		if (devicesToStart < 1) { // <--
			LOG_IF_F(INFO, NORMALVERBOSITY, "start1: No devices enabled for mining- not starting.");
			MessageBox::Show("No devices are enabled for mining. Please enable one or more GPUs in the Devices List \n"
				"(right-click and select 'Enable Device'), or use the CPU Mining tab.", "COSMiC", MessageBoxButtons::OK, MessageBoxIcon::Information);
			return;		// don't try to start on 0 devices ;)
		}

		// [todo]: just dim the Start button if no CUDA Devices are enabled?
		if (gStopCondition) {
			statusbar_minerstate->Text = "Status: Error State";
			System::Windows::Forms::DialogResult quitbox_result = MessageBox::Show("Mining stopped unexpectedly. GPU(s) are in an error state.\n"
				"Please close and re-launch the application.", "COSMiC", MessageBoxButtons::OK, MessageBoxIcon::Stop);
			// "If you are overclocking, check your settings."
			return;
		}
	}

	// semi-redundant: set the mining mode. should also be done when mode is changed, 
	// and when the application is launched.				    [todo]: remember mode?
	printf("Mining Mode: ");
	if (combobox_modeselect->SelectedIndex == MODE_SOLO) {
		LOG_IF_F(INFO, gVerbosity>=V_NORM, "Starting in Solo Mode");
		statusbar_minerstate->Text = "Status: Starting in Solo Mode...";	// <--
		gSoloMiningMode = true;
	} else {	// else if (combobox_modeselect->SelectedIndex == MODE_POOL) {
		LOG_IF_F(INFO, gVerbosity>=V_NORM, "Starting in Pool Mode");
		statusbar_minerstate->Text = "Status: Starting in Pool Mode...";	// <--
		gSoloMiningMode = false;
	}

	statusbar_balanceEth->Enabled = gSoloMiningMode;		// dimmed balances in pool mode
	statusbar_balanceTokens->Enabled = gSoloMiningMode;		// 

	// update the difficulty update frequency
	doEveryCalls_Settings[Timings::getDifficulty] = gDiffUpdateFrequency;	// [MOVEME]?

	if (!gCudaSolving)  // not mining, so:
	{	// check if the miner has been configured from default 
		// TODO: load this setting to a variable at form load to avoid accessing disk?
		if (ConfigurationManager::AppSettings["DialogConfigured"] != "true") {
			MessageBox::Show("The miner is not yet configured. Please enter your mining address, "
				"pool address, etc. using the Options->General Setup... menu item. If Solo Mining, "
				"import an Ethereum account in Options->Configure Solo Mining.", 
				"COSMiC", MessageBoxButtons::OK, MessageBoxIcon::Information);
			return;  // don't start mining
		}

		// blank the subitems of the deviceView (TODO: ensure this is done at mining stop too)
		for (unsigned int i = 0; i < (unsigned short)DevicesView->Items->Count; ++i) {
			LOG_IF_F(INFO, DEBUGMODE, "debug:  Check value:	%d.	Devices Started: %d) \n", i, gCudaDevicesStarted);
			ClearDevicesListSubItem(i, 2);  // hash rate 
			ClearDevicesListSubItem(i, 4);  // temp
			ClearDevicesListSubItem(i, 5);  // pwr
			ClearDevicesListSubItem(i, 6);  // rpm's
			ClearDevicesListSubItem(i, 8);  // solve time
		}
		
		if (gSoloMiningMode) {
		// === Solo Mode: ===
			if (gStr_ContractAddress == "") { /* not configured: */
				Console::WriteLine("No Contract Address configured. Not starting.");
				MessageBox::Show("No Contract Address has been specified. Please use the 'Configure Solo Mining...' menu item.\nMining not started.", "COSMiC", MessageBoxButtons::OK, MessageBoxIcon::Asterisk);
				return;	// don't start mining
			}

			if (gStr_SoloEthAddress.empty() || gStr_SoloEthAddress == "0x00") { /* default value - fixme */
				EnterPasswordForm^ formInst = gcnew EnterPasswordForm();				// instance of EnterPassword form
				
				this->Enabled = false;													// dim form before blocking call
				auto/*System::Windows::Forms::DialogResult*/ drslt = formInst->ShowDialog();	// get password from user to load encrypted acct
				this->Enabled = true;													// undim when unblocked.
				if (drslt == System::Windows::Forms::DialogResult::Cancel)				// user clicked Cancel in EnterPassword dialog
					return;	// don't start mining
			}
			

			// is it still default? or did EnterPassword form (above) fill it in?
			if (!IfEthereumAddress(gStr_SoloEthAddress)) {
				domesg_verb("Couldn't load Ethereum account. Mining not started.", true, V_NORM);
				return;	// don't start mining
			}
 
			textbox_ethaddress->Text = gcnew String(gStr_SoloEthAddress.c_str());  // update in HUD
			listbox_events->Items->Add( gcnew String("-- Contacting Ethereum node... --") );
			if (!listbox_events->Focused) { /* don't scroll if user is interacting w/ the events list.  [TESTME] */
				listbox_events->TopIndex = listbox_events->Items->Count - 1;  // scroll to new item
				listbox_events->Refresh(); }								  // ensure redrawn/visible

			std::string str_txcount{ "" };
			GetParmRslt rslt = UpdateTxCount(gStr_SoloEthAddress, str_txcount);
			if (rslt==GetParmRslt::GETPARAM_ERROR) {
				LOG_F(ERROR, "Couldn't get Transaction Count for ethereum account %s.", gStr_SoloEthAddress.c_str());  // function param?
				return;		// don't start mining
			}
			else if (rslt==GetParmRslt::OK_PARAM_CHANGED) {
				LOG_IF_F( INFO, gVerbosity>=V_MORE, "New txn count:  %s", str_txcount.c_str() );
				// ...
			}
			else if (rslt==GetParmRslt::OK_PARAM_UNCHANGED) {
				/* unchanged. */
			}
			else { // default: should never happen.
				LOG_F( ERROR, "Couldn't get Transaction Count for ethereum account %s: unknown error.", gStr_SoloEthAddress.c_str());  // function param?
				return;		// don't start mining
			}
			// ^
			// check for valid gTxCount_hex before starting? return if invalid. (should never happen if execution reaches this point.
		}
		else {
		// === Pool Mode: ===
			statusbar_minerstate->Text = "Status: Contacting Pool...";
			domesg_verb( "Contacting pool at " + gStr_PoolHTTPAddress + "...", true, V_LESS );  // <- always, impt for UX.
//			listbox_events->Items->Add( gcnew String(/*  ^ that string ^  */);
//			printf("Contacting Pool at %s... \n", gStr_PoolHTTPAddress.c_str());  // [redundant?]
			if (!listbox_events->Focused) {
				listbox_events->TopIndex = listbox_events->Items->Count - 1;  // scroll us to the bottom
				listbox_events->Refresh();	// so we see it (TODO: get initial pool params w/ a BGworker like Solo mode).   <- redundant?
			}
			textbox_ethaddress->Text = gcnew String( gMiningParameters.mineraddress_str.c_str() );  // update mining address in GUI  [TESTME] <--
		}

		// === either mode: ===
		if (NetworkBGWorker->IsBusy) {
			domesg_verb("Network Worker busy: mining might not have stopped cleanly. Please try again.", true, V_LESS);  // always show
			NetworkBGWorker->CancelAsync();  // stop the network BGworker
			return; 	// the BGworker will be started after DoMiningSetup() so mining parameters it needs will be available.
		}				// return to give BGworker a moment to re-start.
		
		start1->Text = "Please Wait...";				// Start/Stop button text changes temporarily
		start1->Enabled = false;						// dimmed

		//minerethaddress: for pool mode.  derivedethereumaddress: for solo mode

		if (!gSoloMiningMode) { //pool mode:  use the pool's minting address, except for the relevant arg to Tokenpool's `submitShare` method.
			gMiningParameters.mineraddress_str = gStr_MinerEthAddress;
			gMiningParameters.mintingaddress_str = "";	// will be retrieved.
		} else { //Solo Mode:	uses the miner's address (derived from imported key). COSMiC will call the token contract's `mint` method directly.
			gMiningParameters.mineraddress_str = gStr_SoloEthAddress;
			gMiningParameters.mintingaddress_str = gStr_SoloEthAddress;
		}

		// [WIP]:	don't continue if DoMiningSetup() returns a non-OK (0) result. <---
		if (DoMiningSetup(menuitem_options_computetarg->Checked, maxtarg_exponent, &gMiningParameters) == 0) { /* <-- [FIXME]?  */
			LOG_IF_F(INFO, HIGHVERBOSITY, "Starting net backgroundWorker.");
			domesg_verb("Starting network thread.", true, V_DEBUG);	// message in "events" list
			//NetworkBGWorker->SetParams(gMiningParameters);	//<-- FIXME: phasing out global.
			NetworkBGWorker->RunWorkerAsync();					// start BGworker!
		} else {
			LOG_F(WARNING, "Network error while getting mining parameters.  (redundant?) ");  // <-
			start1->Text = "Start Mining";
			start1->Enabled = true;
			return;
		}

		//gCudaSolving = true;  // if false, CUDA solver CPU threads will stop automatically. <--- [WIP]: don't set until solvers have started?
		
		timer_enableminingbutton->Start();				// we will re-enable for "Stop" once mining starts (todo: more elegant approach?)
		combobox_modeselect->Enabled = false;			// dim mode-select dropdown menu
		start1->Text = "Please wait...";
		start1->Refresh();								// so we see the change right away
		//
		LOG_IF_F(INFO, DEBUGMODE, "Clearing Solutions View items.");
		ClearTxViewItemsData();					 // clears the listview's associated (native) variables
		listview_solutionsview->Items->Clear();  // first clear out any old items from prev. session
			
		// start timer to launch the TXhelper BGworker and start handling of TXview. [todo]: more direct
		if (gSoloMiningMode) {
			domesg_verb("Clearing solutions view items... ", true, V_DEBUG);  // <--
			statusbar_minerstate->Text = "Status: Starting (Solo Mode)...";

			timer_txview->Enabled = true;
			timer_txview->Start();
		}
		 else { statusbar_minerstate->Text = "Status: Starting (Pool Mode)..."; }
		//configureToolStripMenuItem->Enabled = false;	// disable Configure... menu item
		// ...

	// === start solver threads: ===
		LOG_IF_F(INFO, HIGHVERBOSITY, "Starting CUDA Solver host threads:");
		gCudaSolving = true;  // if false, CUDA solver CPU threads will stop automatically. <--- [WIP]: don't set until solvers have started?
	// d: device as numbered in the device list, starting from 0. unrelated to cuda/other api device#
		for (int d = 0; d < DevicesView->Items->Count; ++d)	// iterate thru devices
		{
			if (Solvers[d] == nullptr) { //just in case, by design this should never happen
				LOG_F(ERROR, "Device# %d corresponds to non-existent Solver!" BUGIFHAPPENED, d);
				return;
				break;	//j.i.c.
			}

			if (!Solvers[d]->enabled)		// [FIXME]: was gCudaDeviceEnabled[d] <--- ... 
				continue;
			//<--- solver#, not device#, in the list!	[WIP]: decouple device# in the list from its solver# ?
			this->DevicesView->Items[d]->SubItems[9]->Text = "Starting";
			//SpawnSolverThread( Solvers[d]->device_type, Solvers[d]->api_device_no );
			Solvers[d]->SpawnThread();	// using the device type/# specified by the Solver object. <--
				/*	spawn a CPU thread for the GPU solver	 */
			++gCudaDevicesStarted;
			//...
		}
	// ===

		// if no devices were started by the above loop (or less than # of -enabled- CUDA devices in system)...
		if (!gCudaDevicesStarted || gCudaDevicesStarted < devicesToStart) {
			NetworkBGWorker->CancelAsync(); // stop network thread (TODO: neater: don't start it)
			MessageBox::Show("No CUDA devices were started. Please enable one or more CUDA devices for COSMiC to use.");
			start1->Text = "Start Mining";
			start1->Enabled = true;
			combobox_modeselect->Enabled = true;
			return; // don't continue
		}

		sw_miningTime->Start();  // elapsed mining time (total)
		statusbar_elapsedTime->DisplayStyle = ToolStripItemDisplayStyle::ImageAndText;  // show clock icon

		//DoMessage("Started mining on " + std::to_string(gCudaDevicesStarted) + quantStr + ".", true, false, 0);// event, not err
		listbox_events->Refresh();  // so it's redrawn after any DoMessage()s

		// TODO: check devicesStarted against devicesToStart, consider how many started (and which) when we call WQ to start below.
		SpawnWatchQatThread();  // start hw health monitoring thread 
	}
	else
	{ // user clicked "Stop Mining":
		CWind_StopMining();				// breaking up into functions
		sw_miningTime->Stop();	// stop "mining time" stopwatch
		sw_miningTime->Reset();  // redundant?
		statusbar_elapsedTime->DisplayStyle = ToolStripItemDisplayStyle::Text;  // hide clock icon
		statusbar_elapsedTime->Text = STR_CosmicVersion;  // from timer back to version #.
		gCudaSolving = false;			// redundant?
	}
} //start1_Click()		[TODO]: simplify and break up this monster function.


private: System::Void configureToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e)
{ // General Options... menu item selected
	Cosmic::OptionsForm^ newForm = gcnew Cosmic::OptionsForm ();
	this->Enabled = false;	// dim CosmicWind
	newForm->ShowDialog();	// blocking call
	this->Enabled = true;	// undim

	// update the difficulty update frequency
	doEveryCalls_Settings[Timings::getDifficulty] = gDiffUpdateFrequency;  // <- any other options to update? [TODO] <--
//	set the verbosity (loguru) <-- [wip]
	textbox_ethaddress->Text = gcnew String( gMiningParameters.mineraddress_str.c_str() );	// update label [TESTME] <--
	textbox_poolurl->Text = gcnew String( gStr_PoolHTTPAddress.c_str() );	// update label
}
private: System::Void aboutCOSMiCToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e)
{  // About ... menu item selected
	Cosmic::AboutForm^ aboutForm = gcnew Cosmic::AboutForm(BigInteger::Zero);	//<-- put the "lifetime # of hashes". [TODO]
	this->Enabled = false;		// dim form
	aboutForm->ShowDialog();	// blocking call
	this->Enabled = true;		// undim form
}

private: System::Void groupBox2_Enter(System::Object^ sender, System::EventArgs^ e) { /* focus? */ }

private: System::Void quitToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e)
{ // Quit menu item was selected
	CosmicWind::Close(); // see the Form Closed handler.
	return;
}

// Auto-start is WIP.	TODO: move startup stuff to a separate comnon function for auto-start and manual. <------ [WIP]: improving auto-start.******
private: System::Void timer2_Tick(System::Object^  sender, System::EventArgs^  e)
{	// This timer starts enabled. It checks the Configuration file for "AutoStart" param


	timer2->Enabled = false;
	timer2->Stop();	//redundant?
	unsigned short miningsetup_result{ 0 }, i{ 0 };

	if (ConfigurationManager::AppSettings["AutoStart"] != "true") {
		domesg_verb("AutoStart is not enabled.", false, V_MORE);
		return; }

	// [WIP] / [TODO]: Retain the Mode setting (Pool/Solo) from last launch? <---
	domesg_verb("Starting mining automatically.", true, V_NORM);
	if (gSoloMiningMode) { LOG_IF_F(INFO, NORMALVERBOSITY, "Auto-starting mining in SOLO mode."); }
	  else { LOG_IF_F(INFO, NORMALVERBOSITY, "Auto-starting mining in SOLO mode."); }

	//NetworkBGWorker->RunWorkerAsync();							//<-- start background worker for network tasks

	LOG_F(WARNING, "Auto-Start is WIP");	//<---
	this->start1->PerformClick();
	//const unsigned short maxtarg_exponent{ 234 };  // 2^234: 0xBTC maxtarget. [TODO]: make user-configurable
	//miningsetup_result = DoMiningSetup( menuitem_options_computetarg->Checked, maxtarg_exponent, &gMiningParameters );  // pass 3rd arg by value
	//if (miningsetup_result == 0) { //0: OK
	//	NetworkBGWorker->RunWorkerAsync();		//NetworkBGWorker->CancelAsync();	
	//	return;
	//} else { /* relevant error message and event log entry should already be done. */ }

	//LOG_IF_F(INFO, gVerbosity >= V_MORE, "Starting hardware health monitoring thread... ");
	//SpawnWatchQatThread();  // [todo]: only start wq if some devices have monitoring enabled! <--

	//gCudaSolving = true;	// when later set to false, CUDA solver threads should stop
	//
	//// launch host threads on enabled devices
	//LOG_IF_F(INFO, gVerbosity >= V_MORE, "Starting CUDA GPU mining threads ...");
	//for (int deviceNo = 0; deviceNo < CUDA_MAX_DEVICES; ++deviceNo) {
	//	if (gCudaDeviceEnabled[deviceNo])
	//		SpawnSolverThread(deviceNo);
	//}
	//start1->Text = "Stop Mining";
	statusbar_minerstate->Text = "Status: Mining (auto-started)";
} // auto-start WIP.


// Show Console Output menu item selected
private: System::Void consoleOutputToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e)
{
	consoleOutputToolStripMenuItem->Enabled = false;	 // gray out menu item
	SpawnConsoleOutput();  // open window to view stream 'stdout'
}

private: System::Void CosmicWind_Load(System::Object^  sender, System::EventArgs^  e)
{
	LOG_F(INFO, "Loading CosmicWind form. " /*, someargs */);
	if (gVerbosity == V_DEBUG) { LOG_F(INFO, "Instantiating stopwatch sw_miningTime. "); }  // dbg
	sw_miningTime = gcnew System::Diagnostics::Stopwatch();  // [MOVEME] ?

// -- DevicesView: `SmoothListView` for CosmicWind Form
// (inherits ListView, adds some properties to prevent flickering when updating stats.)
//	SmoothListView^ newLV = gcnew SmoothListView();  // instantiate custom listview		[MOVEME] ?
//	this->DevicesView->View = View::Details;
//	newLV->Visible = false;

// [TESTME]:
	panel_deviceslist->Controls->Add( this->DevicesView );	// lower row, only one column (0) of table panel
//	tablelayoutpanel_top->Controls->Add(newLV, 0, 1); 
// -- done setting up the DevicesView?						// [WIP] !! ^

// [todo]:	consider collapsing the top panel above the Splitter if 0 CUDA devices detected (CPU mining only).
	executionStateToolStripMenuItem->SelectedIndex = DEFAULT_EXECUTIONSTATE_SETTING;  // avoid display powerdown/system suspend
	hUDUpdateSpeedToolStripMenuItem->SelectedIndex = DEFAULT_HUD_UPDATESPEED;		  // 200 ms UI update rate default

// ^ PASTED, CHECK FOR DUPS. ^

//	// moved to CosmicWind form's InitializeComponent() <--- [WIP] [MOVEME] ?
//	DevicesView->LostFocus += gcnew System::EventHandler(this, &CosmicWind::DevicesView_LostFocus);
//	DevicesView->GotFocus += gcnew System::EventHandler(this, &CosmicWind::DevicesView_GotFocus);
//	DevicesView->Click += gcnew System::EventHandler(this, &CosmicWind::DevicesView_Click);

	// CPU Mining panel stuff:
	textbox_cpu_infobox->Text = System::Environment::ProcessorCount.ToString() + " logical processors";  // TODO: auto-select # of threads?

	SmoothListView^ cpuThreadsLV = gcnew SmoothListView();  // instantiate a double buffered listview. [MOVEME]
	cpuThreadsLV->Visible = false;  // NOT visible immediately, so user doesn't see it build
	panel_threads->Controls->Add(cpuThreadsLV);	   // add to pre-placed panel
	cpuThreadsLV->View = View::Details;			   // columned list
	threadsListView = cpuThreadsLV;				   // so we can access it
	threadsListView->Activation = System::Windows::Forms::ItemActivation::OneClick;
	threadsListView->BorderStyle = System::Windows::Forms::BorderStyle::None;  // BorderSingle;
	threadsListView->HeaderStyle = System::Windows::Forms::ColumnHeaderStyle::Nonclickable;
	threadsListView->Scrollable = true;
	threadsListView->AllowColumnReorder = true;  // <---
	threadsListView->FullRowSelect = true;
	threadsListView->HideSelection = false;  // don't show "ghost" when unfocused control
	threadsListView->CheckBoxes = false;     //
	threadsListView->GridLines = true;
//	threadsListView->Margin.Left = 25;

	//just use Dock::Fill?  [todo/wip]
	threadsListView->Width = panel_threads->Width;		// fill the panel (X)
	threadsListView->Height = panel_threads->Height;	// fill the panel (Y)
//	threadsListView->Anchor = AnchorStyles::Top;		// [ref]
	threadsListView->Visible = true;					// can now be seen
	threadsListView->TabStop = true;
	threadsListView->TabIndex = 23;
	threadsListView->View = View::Details;

	// TODO: use %'s instead
	threadsListView->Columns->Add("Thr#");
	threadsListView->Columns[0]->Text = "Thr#";
	threadsListView->Columns[0]->Width = 40;
	threadsListView->Columns->Add("Hash Rate");
	threadsListView->Columns[1]->Width = 90;
	threadsListView->Columns->Add("Hashes Total");
	threadsListView->Columns[2]->Width = 130;
	threadsListView->Columns->Add("Shares (Invalid/Stales)");
	threadsListView->Columns[3]->Width = 90;
	threadsListView->Columns->Add("Reserved");  // todo (was 'best')
	threadsListView->Columns[4]->Width = 90;
	threadsListView->Columns->Add("Solve Time");
	threadsListView->Columns[5]->Width = 100;
	threadsListView->Visible = true;  // can now be seen

#ifdef TEST_BEHAVIOR
	panel_threads->Enabled = true;
	checkbox_useCPU->Enabled = true;  // CPU mining toggle checkbox enabled (in debug builds)
#else
	panel_threads->Enabled = false;
	checkbox_useCPU->Enabled = false;  // CPU mining not enabled yet (in development)
#endif
	// CosmicWind_ResizeStuff();
}

private: System::Void CosmicWind_FormClosed(System::Object^  sender, System::Windows::Forms::FormClosedEventArgs^  e)
{ // Close requested by a method other than the Quit menu item (X, Alt-F4, Taskbar right-click, Close etc.)
	gApplicationStatus = APPLICATION_STATUS_CLOSING;  // will cause auxiliary threads to stop gracefully
	
	// stop mining/clean up for enabled devices:
	const uint8_t num_of_devices = static_cast<uint8_t>(this->DevicesView->Items->Count);
	if (num_of_devices > 0) {
		LOG_IF_F(INFO, HIGHVERBOSITY, "cleaning up behind %u devices. [MOVEME]?", num_of_devices);
		for (uint8_t i = 0; i < num_of_devices; ++i) {	// was: i < CUDA_MAX_DEVICES
			if (Solvers[i]->enabled /* && Solvers[i].solver_status...*/) {	//gCudaDeviceEnabled[i]
				LOG_IF_F(INFO, HIGHVERBOSITY, "Stopping CUDA%d solver...", i);
				domesg_verb("Stopping CUDA Device # " + std::to_string(i) + " solver...", true, V_DEBUG);	// dbg only!
				gSolving[i] = false;	//matching solver (device host) thread will stop (gracefully)
			//	Solvers[i]->solver_status = SolverStatus::Null;					// cleanup elsewhere? (destructor?)
			}
		}
		//ClearEventsQueue();	//for events list box <--- redundant?
	}

	LOG_IF_F(INFO, NORMALVERBOSITY, "\n\nThanks for using COSMiC !\n");
	Application::Exit();  // actually quit
}

private: System::Void timer_enableminingbutton_Tick(System::Object^  sender, System::EventArgs^  e)
{
	timer_enableminingbutton->Enabled = false;
	// this timer, when it ticks, enables the Start Mining! button again. Gives Network Worker enough time to cancel.
	// serves a second purpose: re-enables the mining button (as `Stop Mining`) after GPUs have started. Don't allow
	// user to try and stop mining while devices are still initializing.
	start1->Enabled = true;

	// if mining, the button is re-enabled once mining begins, but the text does not change.
	if (gCudaSolving)
	{
		start1->Text = "Stop Mining";
		//statusbar_minerstate->Text = "Status: Mining  ") ";  // TODO: replace with GetStatus() function using APPLICATION_STATUS_ values.
	}
	else
	{
		start1->Text = "Start Mining!";
		//statusbar_minerstate->Text = "Status: Idle";  // TODO: replace with GetStatus() function using APPLICATION_STATUS_ values.
	}

// UPDATE TITLE BAR
	// mining threads launched, update titlebar text, inform user, spawn WatchQat thread, etc.
	std::string quantStr = " CUDA Device";
	if (gCudaDevicesStarted > 1)  quantStr += "s";  // plural

	if (gCudaSolving) {
		std::string titleBarStr = std::string(STR_CosmicVersion) +
			"  [ Mining on " + std::to_string(gCudaDevicesStarted) + quantStr + " ] ";

	this->Text = gcnew String( titleBarStr.c_str() );  // set title bar text back to just the version (no longer mining)
	// (todo): consider moving this to be updated, so device count is accurate if one stops. ^
	}
	else
	{
		statusbar_elapsedTime->Text = STR_CosmicVersion;

	}
}

private: System::Void timer_resumeafternetfail_Tick(System::Object^  sender, System::EventArgs^  e)
{ // Timer which is enabled when too many network errors have caused mining to pause (to save power.) Ticks once
	timer_resumeafternetfail->Enabled = false;  // ... then not again til re-enabled

	stat_net_consecutive_errors = 0;
	gNetPause = false;  // allows recurring_network_tasks() to run again
	
	// Start Mining Again!
	start1->PerformClick();
}

private: System::Void label5_Click_1(System::Object^  sender, System::EventArgs^  e)
{
	//devicesListView1->Items[1]->UseItemStyleForSubItems = true;

	// list device enable status, intensity setting
	const unsigned short solvers_count = static_cast<unsigned short>(Solvers.size());
	for (uint8_t i = 0; i < solvers_count ; ++i)							/* was: CUDA_MAX_DEVICES, gCudaDeviceEnabled[i] */
		if ( Solvers[i]->enabled )
			printf("device # %u: %u intensity\n", i, Solvers[i]->intensity);
		else
			printf("device # %d disabled\n", i);
	printf("\n");
}

private: System::Void contextMenu_gpu_Opening(System::Object^ sender, System::ComponentModel::CancelEventArgs^ e)
{ // GPU configuration context menu opening:
	const unsigned short howmany = (unsigned short)this->DevicesView->SelectedItems->Count;
	unsigned short d{ 0 }, g{ 0 };
	bool anyDeviceEnabled{ false }, anyDevicePaused{ false };

	if (gCudaSolving)  //mining: unless in max verbosity ("debug mode"), user can't change intensity now
	{
		resetHashrateCalcToolStripMenuItem->Visible = (gVerbosity == V_DEBUG);  // only visible in 'debug mode'
		// if mining, this option is grayed out (unless debug mode)
		if (gVerbosity == V_DEBUG) {
			enableDisableGpuToolStripMenuItem->Enabled = true;   // dbg
			setIntensityToolStripMenuItem->Enabled = true;  }	 // 
		else {
			setIntensityToolStripMenuItem->Enabled = false;
			enableDisableGpuToolStripMenuItem->Enabled = false;
		}
		enableDisableGpuToolStripMenuItem->ToolTipText = "This setting is disabled while mining.";    // 
		setIntensityToolStripMenuItem->ToolTipText = enableDisableGpuToolStripMenuItem->ToolTipText;  // same
		enableDisableGpuToolStripMenuItem->Text = "Enable/Disable";									  // redundant
	}
	else { //not mining, user can change these settings:
		enableDisableGpuToolStripMenuItem->Enabled = true;
		resetHashrateCalcToolStripMenuItem->Visible = false;
		enableDisableGpuToolStripMenuItem->ToolTipText = "Select whether or not to use this device for mining.";
		setIntensityToolStripMenuItem->ToolTipText = "Set intensity for this device (important: adjust for your hardware!)";
		setIntensityToolStripMenuItem->Enabled = true;  // user can change intensity (not mining)
	}

	if (howmany == 1)  // one device selected
	{ //domesg_verb("contextMenu_gpu for CUDA Device # " + std::to_string(cudaIndex) + " opening ", false, V_DEBUG);
		const unsigned short solverNo = static_cast<unsigned short>(DevicesView->SelectedItems[0]->Index);
		gpuNameAndIndexMenuItem->Text = howmany.ToString() + " CUDA Devices Selected:";
		if (gVerbosity == V_DEBUG)
			printf("device selected: %u \n", solverNo); // dbg

		// 'header' with device index# and name
		// [WIP] / [TESTME]: This now uses the Solver / Device# as assigned by COSMiC, not a CUDA device#. Show API device # also? <--
		contextMenu_gpu->Text = solverNo.ToString();  // context menu's title (text) is not displayed, GPU# is stored here!
		gpuNameAndIndexMenuItem->Text = "device #" + contextMenu_gpu->Text + " (" + DevicesView->Items[solverNo]->SubItems[1]->Text + "):";
		//
			if (Solvers[solverNo]->enabled)		// enabled.		was: `gCudaDeviceEnabled[cudaIndex]`.
				enableDisableGpuToolStripMenuItem->Text = "Disable Device";
			else
			{ // disabled, so:
				gpuNameAndIndexMenuItem->Text += " (Disabled)";
				enableDisableGpuToolStripMenuItem->Text = "Enable Device";
			}
			enableDisableGpuToolStripMenuItem->ToolTipText = "Enable or Disable mining on this device.";
		//
		// cuda engine select:
		cUDAEngineToolStripMenuItem->Enabled = true;
		if (gCuda_Engine[solverNo] < 2)  //check for invalid value. [todo]: use enum type
			cUDAEngineToolStripMenuItem->SelectedIndex = gCuda_Engine[solverNo];  // <--- FIXME

		forceUnpauseToolStripMenuItem->Enabled = (Solvers[solverNo]->pause != PauseType::NotPaused);	//enable menu item if paused	[TESTME]. <--

		hWMonitoringAndAlarmsToolStripMenuItem->Enabled = true;    // enable H/W Health item
		gpuSummaryMenuItem->Enabled = true;						   // enable GPU Summary item
	}
	else if (howmany > 1)  // multiple devices selected:
	{
		// header describes >1 GPU:
		gpuSummaryMenuItem->Enabled = false;  // WIP (GPU summary w/ multiple GPUs) <--- fixme
		contextMenu_gpu->Text = "Multiple Devices";
		gpuNameAndIndexMenuItem->Text = howmany.ToString() + " Devices Selected";
		cUDAEngineToolStripMenuItem->Enabled = false;  // engine select disabled w/ multiple GPUs selected
		hWMonitoringAndAlarmsToolStripMenuItem->Enabled = false;  // TODO (multi-devices health config)
		for (g = 0; g < howmany; ++g)
		{
			if (Solvers[g]->enabled)			// so contextmenu items relevant to 	/* was: gCudaDeviceEnabled[g] */
				anyDeviceEnabled = true;		// selected devices can be enabled contextually.
			
			if (Solvers[g]->pause != PauseType::NotPaused)		// any device has been paused by WatchQat
				anyDevicePaused = true;								// (bulk force-unpause available)
			gpusSummarized[d] = g;	// store CUDA device # (indexed from 0 in the Smooth-ListView) for Summary form
			d += 1;
		}
		// unpausing multiple devices individually or as a group:
		if (anyDevicePaused) {
			forceUnpauseToolStripMenuItem->Text = "Force Unpause (" + howmany.ToString() + ") Devices";
			forceUnpauseToolStripMenuItem->Enabled = true;
		} else  forceUnpauseToolStripMenuItem->Enabled = false;
			
		if (!gCudaSolving) {
			enableDisableGpuToolStripMenuItem->ToolTipText = "Enable or Disable mining on these devices.";
			if (anyDeviceEnabled)  enableDisableGpuToolStripMenuItem->Text = "Disable (" + howmany.ToString() + ") Devices";
			else  enableDisableGpuToolStripMenuItem->Text = "Enable (" + howmany.ToString() + ") Devices";
		} else enableDisableGpuToolStripMenuItem->ToolTipText = "This setting is disabled while mining.";
		//
	} else {  // 0 devices selected.	// prevent opening the context menu by right-clicking empty area, 
										// column headers or AltGr key in non-row area
		if (gVerbosity != V_DEBUG) {
			e->Cancel = true;  // abort opening context menu
			return;
		}
	}

}

private: System::Void configureGPUToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e)
{
	// GPU #x Summary... was clicked in the ListView ContextMenuStrip.

	// NEW: 
	const unsigned short selectedDevicesCnt = (unsigned short)DevicesView->SelectedItems->Count;
	unsigned short selectedDevices[CUDA_MAX_DEVICES] = { 0 };

	if (selectedDevicesCnt > 1)
	{
		unsigned int d{ 0 }, d2{ 0 };
		for (d; d < CUDA_MAX_DEVICES; ++d)
			selectedDevices[d] = UINT8_MAX;		// init value

		for (d = 0; d < selectedDevicesCnt; ++d)
		{
			selectedDevices[d2] = d;  // store CUDA device # (indexed from 0 in the Smooth-ListView).
			d2 += 1;				  // iterate element # in selectedDevices[] array for next device id#
		}
	}

	// New instance of gpuDetailsForm, passed the selected Item of DevicesListView1
	Cosmic::GpuSummary^ gpuDetailsFormInst;
	if (selectedDevicesCnt > 1)  // pass array of CUDA device indices, or pass single device# and the init'd array
		 gpuDetailsFormInst = gcnew Cosmic::GpuSummary(-1, selectedDevices);									 
	else gpuDetailsFormInst = gcnew Cosmic::GpuSummary(this->DevicesView->FocusedItem->Index, selectedDevices);

	if (gVerbosity > V_NORM)  printf("Opening GPU Summary form \n");
	gpuDetailsFormInst->Show();	// show the gpuDetailsForm (not modal)
	//this->DevicesView->SelectedItems->Clear();     // <---
	this->DevicesView->FocusedItem->Focused = false;

	//int whereX = this->DesktopBounds.Left;
}

private: System::Void deviceCountersToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e)
{
	// Device Counters command selected from menu
	Cuda_PrintDeviceCounters();
}

private: System::Void timer_net_worker_restart_Tick(System::Object^  sender, System::EventArgs^  e)
{ // old, but potentially useful.
  // in the event of premature stop (which testing indicates shouldn't happen) but the GPUs are
  // still working fine, this timer will re-start it to ensure mining params are up-to-date.
	// Network Worker Restart timer ticked.
	timer_net_worker_restart->Enabled = false;
	timer_net_worker_restart->Stop();

	domesg_verb("Restarting Network Worker after unexpected stop.", true, V_LESS);  // always
	LOG_F( WARNING, "Restarting Network BGworker after unexpected stop" );
	NetworkBGWorker->RunWorkerAsync();	// re-start
}

private: System::Void pauseUnpauseToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e)
{
	// Net Reason Pause menu item selected
	gNetPause == true ? gNetPause = false : gNetPause = true;
	printf("Switched gNetPause status.\n");
}

private: System::Void CosmicWind_FormClosing(System::Object^  sender, System::Windows::Forms::FormClosingEventArgs^  e)
{
	// The form was requested to be closed by some means.
	uint8_t i = 0;	// loop counter

	// If not mining, just stop mining and close instantly. Otherwise:
	// (this Form Closure event might've been invoked by the System rebooting for updates, etc.)
	// This handler attempts to hold up a shutdown and ask the user if they REALLY want to stop mining/quit.
	if (gCudaSolving == true || gCpuSolverThreads > 0)
	{
		System::Windows::Forms::DialogResult quitbox_result = MessageBox::Show("You, or the system, has requested COSMiC to close. \nMining is still in progress. Stop mining and quit?",
			"COSMiC - Shutdown Requested", MessageBoxButtons::YesNo, MessageBoxIcon::Question);
		if (gVerbosity == V_DEBUG) { printf("quitbox_result: %d\n", (int)quitbox_result); }

		if (quitbox_result != System::Windows::Forms::DialogResult::Yes)  /* user clicked `No` */
			e->Cancel = true; // set cancellation of the Form Closure in the Event (don't exit).
	}
}

// [MOVEME] ?
private: System::Void Init_SetTimings( )
{
	//	doEveryCalls_Settings[Timings::null] = 0;		// redundant
	doEveryCalls_Settings[Timings::getTxReceipt] = 2;	// 
	doEveryCalls_Settings[Timings::getDifficulty] = 1;	// [FIXME] ?
	doEveryCalls_Settings[Timings::getGasPrice] = 10;
	doEveryCalls_Settings[Timings::getTxnCount] = 25;
	doEveryCalls_Settings[Timings::getBalances] = 10000;
	doEveryCalls_Settings[Timings::getPoolMintAddr] = 75000;	//<- 
	doEveryCalls_Settings[Timings::profTimer1] = 40;			//<- check/adjust these [TODO].
	LOG_IF_F(INFO, HIGHVERBOSITY, "Set timings (defaults).");	// 
}


private: System::Void CosmicWind_Shown(System::Object^ sender, System::EventArgs^ e)
{ // FIXME: use TryParse() in place of Convert:: class anywhere I haven't already in
  //			this function (most replaced already), prevents exception in the event
  //			of errors parsing a somehow-corrupt Config file.						[TODO] / [DONE?] <--

  // TODO: Detect CUDA devices first and if <0, collapse the DevicesView SmoothListView? (for CPU mining only.)
  //			  Use detected # of devices, don't require "null" values for nonexistant devices in Config.
  //			  Condense this handler. It could also be faster.
	//IFormatProvider^ numberFormat = gcnew CultureInfo("en-US");
	//System::Configuration::Configuration^ configHndl = ConfigurationManager::OpenExeConfiguration(ConfigurationUserLevel::None);
	bool configuration_issue_detected = false;  // spawn a dialog box if true?

	if (!SetUpDevicesView())			//<--- [MOVEME]: do in the SmoothListView's constructor, when it's instantiated as a member of the CosmicWind form class.
		LOG_F(ERROR, "No. of columns is not 10:		%d" BUGIFHAPPENED, this->DevicesView->Columns->Count);	//<-----
	//
	// Verify that the # of rows in Devices List matches the # of allocated Solvers! <----
	//
	
// === [WIP]: get count of devices. Currently only counts CUDA devices (GPUs) ===
	const unsigned short device_count = Form_DetectDevices();					// populate devices listbox. get device count.
	if (!device_count)
		return;	// ERROR <----- [FIXME]! If allocation failed, display messagebox and exit? <----
	
	Init_PopulateDevicesList( device_count );		//<--- [MOVEME] ?

// === Read the Configuration, apply settings. ===

	const bool b_configOK = CosmicWind_ReadConfiguration(device_count);	// with how many cards to expect when reading Config	[WIP] <--
// [TODO]: remember enabled/disabled devices status via Configuration?
//
	const unsigned short devices_in_view = static_cast<unsigned short>(this->DevicesView->Items->Count);
	for (unsigned short i = 0; i < devices_in_view; ++i) {	//<--- LEFT OFF HERE: get intensity of only the solvers that exist! <----
		if (Solvers[i] != nullptr) { /* make sure the solver was allocated [redundant] */
			DevicesView->Items[i]->SubItems[7]->Text = Solvers[i]->intensity.ToString();	//write intensity (from config, or default.)
			// anything else?
		} else { 
			LOG_F(ERROR, "Matching Solver for Device# %u does not exist!" BUGIFHAPPENED, i);	// should never happen, as long as
			//fatal error [todo]																// the solvers were all allocated.
		}
	}

	combobox_modeselect->SelectedIndex = MODE_POOL;  // Pool Mode by default.  [TODO]:  save last mode used

	Init_SetTimings();
}


private: System::Void contextMenu_eventsBox_Opening(System::Object^  sender, System::ComponentModel::CancelEventArgs^  e)
{
	contextMenu_copySelection->Enabled = false;  // WIP/FIXME
	/* if (this->listbox_events->SelectedItems->Count)
		this->contextMenu_copySelection->Enabled = true;
	else  // no items selected
		this->contextMenu_copySelection->Enabled = false;  <-- disabled for debugging */
}

private: System::Void copyEventsToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e)
{
	// prevent potential exception if nothing selected
	printf("selected events: %d \n", listbox_events->SelectedItems->Count);
	if (listbox_events->SelectedIndices->Count < 1) {
		Console::WriteLine("null selecteditem");
		return;
	}

	String^ scratch_mstring = "";
	unsigned int linecount = listbox_events->SelectedItems->Count;	//listbox_events->SelectedIndices->Count
	if (linecount > 20) { // [TODO]: check bounds of clipboard on various Windows versions.
		Console::WriteLine("CosmicWind: more than 20 items selected- only copying 20 to clipboard. ");
		linecount = 20;
	}

	// StringBuilder should be faster [TODO]. <---
	for (unsigned int i = 0; i < linecount; ++i) {
		if (i > 0)
			scratch_mstring += "\n";	//newline following each line after first
		scratch_mstring += listbox_events->SelectedItems[i]->ToString();
	}

	// put text to the clipboard
	if(DEBUGMODE) Console::WriteLine("Copying: " + scratch_mstring);
	if(DEBUGMODE) Console::WriteLine("Clipboard operation (set)...");
	Clipboard::SetText(scratch_mstring);
//Clipboard::SetDataObject(scratch_mstring);		// <-- new
//	this->listbox_events->SelectedItems->Clear();
	listbox_events->ClearSelected();				// <-- new
}

private: System::Void contextMenu_clearEvents_Click(System::Object^  sender, System::EventArgs^  e)
{
	listbox_events->Items->Clear();
	domesg_verb("User cleared events manually. Flushing events queue. ", false, V_DEBUG);  // to stdout
	ClearEventsQueue();
}

private: System::Void helpToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
}

// === unpausing multiple selected devices ===
private: System::Void forceUnpauseSelectedDevices ()
{
	//can't select more rows (devices) in the DevicesView than actually exist
	LOG_IF_F(INFO, DEBUGMODE, "Manually unpausing %d selected devices:", DevicesView->SelectedItems->Count);
	const unsigned short num_selected = static_cast<ushort>(DevicesView->SelectedItems->Count);
	if (DevicesView->SelectedItems->Count > MAX_SOLVERS || !DevicesView->SelectedItems->Count) {	//just in case: there "could" be too many rows.
		LOG_F(ERROR, "No devices are selected (or too many): %d." BUGIFHAPPENED, num_selected);
		return;	//error
	}

	// [WIP] making this device type-generic!
	// [FIXME] !! this assumed the devices selected started at 0. Rework this to handle the selected device #'s. <------- [FIXME] !

	// iterate through the selected items:
	// [todo]: foreach loop instead?
	for (unsigned short i = 0; i < num_selected; ++i)
	{ //get the device (solver) # from the index of each _selected_ row in DevicesView (numbered from zero)
		const unsigned short solverNo = static_cast<unsigned short>(DevicesView->SelectedItems[i]->Index);
		if (solverNo >= MAX_SOLVERS) {
			LOG_F(ERROR, "forceUnpauseSelectedDevices(): bad Solver# %u." BUGIFHAPPENED, solverNo);
			return;	//error
		}
		if (!Solvers[solverNo]) {
			LOG_F(ERROR, "forceUnpauseSelectedDevices(): Solver# %u does not exist!" BUGIFHAPPENED, solverNo);
			return;	//continue;
		}
		if (Solvers[solverNo]->pause == PauseType::NotPaused)
			continue;	//skip device
	//	if (Solvers[i]->enabled == false)
	//		continue;	//"

		LOG_IF_F(INFO, NORMALVERBOSITY, "Unpausing device# %u", solverNo);	//DBG
		Solvers[solverNo]->pause = PauseType::NotPaused;
		if (solverNo <= 18) {
			gDeviceManuallyUnpaused[solverNo] = true;
			Solvers[solverNo]->ResetHashrateCalc();		// [TESTME].	<---
		} else LOG_F(ERROR, "forceUnpauseSelectedDevices(): Solver# %u exceeds max. supported # of solvers!" BUGIFHAPPENED, solverNo);
	} //for

}

private: System::Void forceUnpauseToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e)
{ // Force Unpause context menu item selected
	if (!gCudaSolving)
		return;			// if not mining

// [FIXME] [TODO]: Also ensure that solver/device `cudaDeviceIndex` is already allocated! <---
	LOG_IF_F(INFO, DEBUGMODE, "Force Unpause: %d devices selected", DevicesView->SelectedItems->Count);
	if (DevicesView->SelectedItems->Count && DevicesView->SelectedItems->Count <= MAX_SOLVERS)
		forceUnpauseSelectedDevices();
	else
		LOG_IF_F(WARNING, NORMALVERBOSITY, "No devices selected to unpause.");
}

private: System::Void setIntensityToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e)
{
	Cosmic::ConfigIntensityForm^ formInstance;
	const unsigned short howMany = (unsigned short)DevicesView->SelectedItems->Count;
	unsigned short gpusArray[CUDA_MAX_DEVICES] = { 0 };
	unsigned short d{ 0 }, g{ 0 };
	
	if (howMany > 1) {
		for (d = 0; d < CUDA_MAX_DEVICES; ++d)		// init the byte array
			gpusArray[d] = UINT8_MAX;				// no device # (UINT8_MAX because CUDA devices indexed from 0.)
		for (d = 0; g < howMany; ++g) {
			gpusArray[d] = g;						// store CUDA device # (indexed from 0 in the Smooth-ListView).
			d += 1;
		}
		formInstance = gcnew Cosmic::ConfigIntensityForm( UINT8_MAX, gpusArray );  // pass in the array of selected GPUs, 1st arg is null
	}
	else  formInstance = gcnew Cosmic::ConfigIntensityForm(DevicesView->SelectedItems[0]->Index, gpusArray);  // device index # and null array

	// display the Intensity Select form:
	this->Enabled = false;		 // dim main form
	formInstance->ShowDialog();  // show form (blocking call)
	this->Enabled = true;		 // undim main form (form closed, will be garbage-collected)
	
	// unblocked once dialog is dismissed:
	for (int i = 0; i < DevicesView->Items->Count; ++i)  // `Count` is type int
		DevicesView->Items[i]->SubItems[7]->Text = Solvers[i]->intensity.ToString();	// set text for gpu/item `d`, in Intensity col.
	// just update every device in the list's intensity now. ^
}

private: System::Void hWMonitoringAndAlarmsToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e)
{
	//if (this->DevicesView->SelectedItems->Count > 1) { ... }
	//	return;

	// Should be able to configure hw monitoring settings on multiple selected devices in one go,
	//	if that is not yet implemented only one device should be configured, and selecting this item
	//	with multiple devices hilighted would cancel opening the ConfigHwMon form.	[FIXME] [WIP] <---
	if (this->DevicesView->FocusedItem->Index < 0 || this->DevicesView->FocusedItem->Index < (MAX_SOLVERS - 1)) {
		Cosmic::ConfigHwMon^ formInstance = gcnew Cosmic::ConfigHwMon(this->DevicesView->FocusedItem->Index);
		this->Enabled = false;			// dim the main form
		formInstance->ShowDialog();		// modal
		this->Enabled = true;			// undim the main form.
	} else {
		LOG_F(ERROR, "Bad focused item# %d in DevicesView!", this->DevicesView->FocusedItem->Index);
	}

}

private: System::Void cUDAEngineToolStripMenuItem_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e)
{
	gCuda_Engine[Convert::ToByte(this->DevicesView->FocusedItem->SubItems[0]->Text)] = (uint8_t)cUDAEngineToolStripMenuItem->SelectedIndex;
}

private: System::Void executionStateToolStripMenuItem_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e)
{
	// execution state selected item changed
	EXECUTION_STATE execState = ES_CONTINUOUS;					   // setting will be ongoing
	if (executionStateToolStripMenuItem->SelectedIndex == 1)	   // allow display powerdown, avoid suspend
		execState = execState | ES_DISPLAY_REQUIRED;
	else if (executionStateToolStripMenuItem->SelectedIndex == 2)  // attempt to prevent display powerdown and suspend
		execState = execState | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED;
	// else... just regular windows behavior (ES_CONTINUOUS, no other flags)

	printf("Setting thread execution state flags. \n");
	SetThreadExecutionState(execState);							   // set our flags
}

private: System::Void configureSoloMiningToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e)
{
	// Configure Solo Mining... menu item selected
	System::Windows::Forms::DialogResult result;
	LOG_F(INFO, "Opening ConfigureSoloMining form.");
	
	ConfigSoloMining^ instance = gcnew ConfigSoloMining();    // instantiate form (see ConfigSoloMining.h)
	this->Enabled = false;	// dim main form while displayed
	result = instance->ShowDialog();
	this->Enabled = true;	// undim main form when unblocked.
	if (result != System::Windows::Forms::DialogResult::Cancel)
		domesg_verb("Updated configuration.", true, V_NORM);
	
	// update UI
	// ===
	// if Solo Mode:  (enable that mode for them, they just configured Solo.)
	if (gSoloMiningMode) {
		combobox_modeselect->SelectedIndex = MODE_SOLO;
		textbox_ethaddress->Text = gcnew String(gStr_SoloEthAddress.c_str());
		textbox_poolurl->Text = "Solo Mode";  } // TODO: get the token contract's name and put that here <---
	//else  // Pool Mode
	//{
	//	lbl_ethaccount->Text = gcnew String(gStr_MinerEthAddress.c_str());  // write to label. It's also updated on General Options... dialog closure
	//	lbl_poolurl->Text = gcnew String(gStr_PoolHTTPAddress.c_str());		// write to pool url label
	//}
	// "Stopped" status?
}

private: System::Void hUDUpdateSpeedToolStripMenuItem_SelectedIndexChanged(System::Object^ sender, System::EventArgs^ e)
{ // HUD Update Rate setting was changed in menubar:
	if (hUDUpdateSpeedToolStripMenuItem->SelectedIndex == 0)		{ timer1->Interval = 100; }
	 else if (hUDUpdateSpeedToolStripMenuItem->SelectedIndex == 1)  { timer1->Interval = 150;  }
	 else if (hUDUpdateSpeedToolStripMenuItem->SelectedIndex == 2)  { timer1->Interval = 200;  }
	 else if (hUDUpdateSpeedToolStripMenuItem->SelectedIndex == 3)  { timer1->Interval = 400;  }
	 else if (hUDUpdateSpeedToolStripMenuItem->SelectedIndex == 4)  { timer1->Interval = 1000; }
	 else { printf("Bad case in HUDUpdateSpeed menuitem handler: %d. \n", hUDUpdateSpeedToolStripMenuItem->SelectedIndex); }
}

//
private: System::Void timer_solobalance_Tick(System::Object^ sender, System::EventArgs^ e)
{ // ticked (make interval configurable? 'update balances in solo mode every __ seconds' or similar)  <- [TODO].
  // get the balances from a BGworker? <--- [TODO].
	if (!gSoloMiningMode) { 
		domesg_verb("timer_solobalance for solo mode only", false, V_DEBUG);  // just in case
		return; }

	timer_solobalance->Stop();  // stop timer while we work
	msclr::interop::marshal_context marshalctx;  // for marshalling managed/native types

	// expects 20 bytes (42 hex digits) prepended by `0x`.
	const bool acctLoaded = IfEthereumAddress(gStr_SoloEthAddress);
	statusbar_balanceEth->Enabled = acctLoaded;		// these fields enabled only if
	statusbar_balanceTokens->Enabled = acctLoaded;	// an account is loaded.
	if (!acctLoaded) {
		//Console::WriteLine("No valid Eth account loaded- didn't get balances ");
		timer_solobalance->Start();  // re-start timer
		return;  }

	// gets balances from the network for the Ethereum account address loaded
	std::string str_ethbalance = GetAccountBalance_Ether(gStr_SoloEthAddress);  // <-- checks before putting into Textbox!
	std::string str_tokensbalance = GetAccountBalance_Tokens(gStr_SoloEthAddress); // <-- FIXME: exception happened here once.
	
	// update our Ether balance in the HUD (plus error prevention/handling)
	if (checkErr_a(str_ethbalance) == true)	  // only update if retrieved successfully
		statusbar_balanceEth->Text = gcnew String( str_ethbalance.c_str() );			// update HUD control
	if (checkErr_a(str_tokensbalance) == true)  // "
		statusbar_balanceTokens->Text = gcnew String( str_tokensbalance.c_str() );		// update HUD control

	// piggybacking on this timer:
	Solo_DonationCheck();		 // check for any waiting donation tx's and send (in practice, should happen right after each mint().)
								 // TODO: combine token amounts ( (reward*donation%)*count ) and send as one Tx! :D  <--  [WIP]: implementing minthelper contract instead
	timer_solobalance->Start();  // re-start timer
}

private: System::Void contextMenu_eventsBox_Closing(System::Object^  sender, System::Windows::Forms::ToolStripDropDownClosingEventArgs^  e)
{
	listbox_events->SelectedItems->Clear();							 // deselect any items
}
private: System::Void contextMenu_gpu_Closing(System::Object^  sender, System::Windows::Forms::ToolStripDropDownClosingEventArgs^  e)
{
	//DevicesView->SelectedItems->Clear();
	// TODO: give focus to another control, such as the bgpanel to avoid "ghost" selection?
}

private: System::Void deviceslist_events_Leave_1(System::Object^  sender, System::EventArgs^  e)
{
	// Devices List lost focus: unselect any items
	DevicesView->SelectedItems->Clear();
}

private: System::Void listbox_events_MouseClick(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e)
{
	if (listbox_events->IndexFromPoint(e->Location) == ListBox::NoMatches)
	{
		if (gVerbosity == V_DEBUG)
			Console::WriteLine("Unindexed point clicked: clearing selected items. ");
		listbox_events->SelectedItems->Clear();
	}
}

private: System::Void bg_panel_Click(System::Object^  sender, System::EventArgs^  e)
{
	DevicesView->SelectedItems->Clear();  // unselect any devices in devices smoothlistview
	listbox_events->SelectedItems->Clear();		// unselect any events in events view listbox
}

private: System::Void CosmicWind_Resize(System::Object^ sender, System::EventArgs^ e)
{ // pop up a balloon-tip from the system tray to show user where the miner minimized to.

	// DEBUG STUFF:
	if (gVerbosity == V_DEBUG) {
		printf("wind size: %d (w)  %d (h) \n", this->Size.Width, this->Size.Height);
		printf("splitcontainer: %d (w)  %d (h) \n", splitPanel->Size.Width, splitPanel->Size.Height);  }

	//splitPanel->Size.Width = this->Size.Width - 50 ;
	//splitPanel->Size.Height = this->Size.Height - 150 ;  // <-------
	// ...
	// end dbg

	// MINIMIZING TO TRAY: reopening will be done by event handlers on the notifyIcon itself (double-click, context menu Show...)
	if (this->WindowState == FormWindowState::Minimized && minimizeToTrayToolStripMenuItem->Checked)
	{ // if minimizing...	
		this->notifyIcon1->BalloonTipTitle = STR_CosmicVersion;
		this->notifyIcon1->BalloonTipText = "COSMiC is still running in the background. :)";
		this->notifyIcon1->Visible = true;				// show system tray icon
		Hide();											// hide from taskbar.

		if (!balloonShown && gVerbosity > V_LESS) {
			this->notifyIcon1->ShowBalloonTip(1750);	// show balloon for 1.75s
			balloonShown = true; }						// won't reshow this session
	}

	DevicesView->SelectedItems->Clear();	 // deselect any devices
	listbox_events->SelectedItems->Clear();  // and any mining events
	listbox_events->TopIndex = listbox_events->Items->Count - 1;  // scroll events list to bottom <--
}

private: System::Void QuitToolStripMenuItem1_Click(System::Object^ sender, System::EventArgs^ e)
{
	// close the form (user will be prompted if mining is in progress)
	this->Close();
}

private: System::Void ShowToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e)
{
	Show();										  // re-show in taskbar
	this->WindowState = FormWindowState::Normal;  // restore window
	notifyIcon1->Visible = false;				  // make notification tray icon invisible again
}

private: System::Void NotifyIcon1_DoubleClick(System::Object^ sender, System::EventArgs^ e)
{
	Show();										  // re-show in taskbar
	this->WindowState = FormWindowState::Normal;  // restore window
	notifyIcon1->Visible = false;				  // make notification tray icon invisible again
}

private: System::Void SummaryToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e)
{
	String^ scratchMStr = "";
	this->notifyIcon1->BalloonTipTitle = "COSMiC Miner V4.1.x TEST";
	scratchMStr = "Hash Rate: " + lbl_totalhashrate->Text + "\n";
	scratchMStr += "Valid Sols: " + lbl_totalsols->Text + "\n";
	scratchMStr += "Approx Power: \n";
	this->notifyIcon1->BalloonTipText = scratchMStr;
	this->notifyIcon1->ShowBalloonTip(4100);
}

private: System::Void TxViewPurgeList_Click(System::Object^ sender, System::EventArgs^ e)
{ // [WIP]: clearing the list of items. Just remove inactive ones (failed/stale/extra)?
	LOG_IF_F(INFO, gVerbosity>=V_MORE, "Cleaning up the Solutions View");  // <--

	ClearTxViewItemsData();					 // clears the listview's associated (native) variables
	listview_solutionsview->Items->Clear();  // and the control itself.

//  clear the solution queue too? (if needed)
	domesg_verb("Manually cleared Solutions View.", true, V_MORE);
	return;
}


System::Void OpenURLinDefaultWebBrowser(System::String^ URL)
{
   if (gVerbosity > V_NORM)
	   Console::WriteLine("Opening URL in default Web Browser: " + URL);

   try
   { // open URL in system default browser.
	   System::Diagnostics::Process::Start(URL);  // TODO: make user-selectable (ETC blockchain etc.)
   }
   catch (System::ComponentModel::Win32Exception^ other)
   { // if an exception thrown
	   MessageBox::Show("Caught exception: " + other->Message, "COSMiC - Exception", MessageBoxButtons::OK, MessageBoxIcon::Hand);
	   Console::WriteLine("OpenURLinDefaultWebBrowser(): exception thrown, aborting! ");
	   return;
   }

}



private: System::Void ViewInBlockExplorerToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e)
{  // View Tx option in Events context menu selected
	// [TODO/WIP]: make this a common function, pass it a txview item# (or a number of them) to it.
	//				the func will also be called to display tx's in the web browser from the TxReceipt form.

	if (!listview_solutionsview->SelectedItems) {
		Console::WriteLine("No transactions selected in Solutions View.");
		return; }

	// some items are selected:
	const unsigned int items_count = listview_solutionsview->SelectedItems->Count;
	if (items_count >= MAX_URLS_TO_OPEN) {
		System::String^ mesgbox_text = "Open " + items_count.ToString() + " transactions in the default Web Browser?";
		if (MessageBox::Show(mesgbox_text, "COSMiC", MessageBoxButtons::OKCancel, MessageBoxIcon::Question) == System::Windows::Forms::DialogResult::Cancel)
			return;  // if "cancel" is clicked. don't open any URLs.
	}

	msclr::interop::marshal_context marshalctx;			// for native to managed types
	std::string str_txhash, str_url{};					// URL to open in browser, and transaction hash to look up
	for (unsigned int i = 0; i < items_count; ++i)
	{ // iterate through multiple selected items. [todo/fixme]: enforce a maximum # or display a warning dialog
	  // if many Tx's are selected when choosing this menu item?
		str_txhash = marshalctx.marshal_as<std::string>( listview_solutionsview->SelectedItems[i]->SubItems[6]->Text );  // from txhash column <--- FIXME: get from gTxViewItems[]
		LOG_IF_F(INFO, gVerbosity>V_NORM, "Opening Tx on Etherscan: %s", str_txhash.c_str());
	//	Console::WriteLine("Showing transaction in browser:  {0}", gcnew String(str_txhash.c_str()));	
		if (!checkString(str_txhash, 66, true, true)) {  // must be 66 characters:  0x + 32 bytes as 64 hex digits
			LOG_F(WARNING, "Bad TxHash for item # %d of TxView! Skipping it.", i);
			continue;	//skip, keep displaying any remaining tx's
		}

		// open a block explorer (TODO: hardcoded to Etherscan for now) in system's selected browser w/ the TxHash. use correct URL for common network types.
		if (gSolo_ChainID == CHAINID_ETHEREUM)		{ str_url = "https://www.etherscan.io/tx/" + str_txhash; }
		 else if (gSolo_ChainID == CHAINID_ROPSTEN)	{ str_url = "https://ropsten.etherscan.io/tx/" + str_txhash; }
//		 else if (gSolo_ChainID == ...)				{ str_url = "..." + txhash_str; }	// goerli, ethereum classic, etc.
		 else  { str_url = "https://www.etherscan.io/tx/" + str_txhash; }				// default: ethereum mainnet

		System::String^ mstr_DELME = gcnew System::String(str_url.c_str());
		OpenURLinDefaultWebBrowser(mstr_DELME);
		// log any err here? <--
	} //for
}

private: System::Void BgWorker_SoloView_RunWorkerCompleted(System::Object^ sender, System::ComponentModel::RunWorkerCompletedEventArgs^ e)
{ // the BGworker thread for the TxView has ended. finished?
	if (!e->Error) {
		//if (gVerbosity == V_DEBUG)  printf("TXHelper BGworker ending neatly. \n");  // TODO: time taken <-
		return; }
	String^ anyErr_MStr = e->Error->ToString();
	Console::WriteLine("TXHelper BGworker ending because: " + e->Error->ToString());
	msclr::interop::marshal_context marshalctx;
	domesg_verb("TXHelper-BGwkr stop: " + marshalctx.marshal_as<std::string>(anyErr_MStr), true, V_NORM );
}

private: System::Void ClearToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e)
{
	if (gVerbosity > V_NORM)  printf("Removing selected items from TxView \n");
	unsigned int howmany = 0;
	for (unsigned short i = 0; i < listview_solutionsview->SelectedItems->Count; ++i)
	{
		listview_solutionsview->SelectedItems[i]->Remove();
		++howmany;
	}
	printf("Cleared %d items. \n", howmany);
}

private: System::Void TxViewContextMenu_Opening(System::Object^ sender, System::ComponentModel::CancelEventArgs^ e)
{
	if (gCudaSolving && gSoloMiningMode)  resubmitTransactionToolStripMenuItem->Enabled = true;  // only if mining & in solo mode.
	  else  resubmitTransactionToolStripMenuItem->Enabled = false;  // otherwise disable re-send tx menu item

	// if no items are selected, disable the Clear Item(s) menu item of the TxView context menu.
	if (listview_solutionsview->SelectedItems->Count >= 1)
	{
		viewInBlockExplorerToolStripMenuItem->Enabled = true;
		viewTxReceiptToolStripMenuItem->Enabled = true;
		clearToolStripMenuItem->Enabled = true;
	}
	else
	{
		viewInBlockExplorerToolStripMenuItem->Enabled = false;
		viewTxReceiptToolStripMenuItem->Enabled = false;
		clearToolStripMenuItem->Enabled = false;
	}
	
	// same approach for the purge-all menuitem (but check items in list, not selected items.)
	listview_solutionsview->Items->Count < 1 ? purgeTxViewMenuItem->Enabled = false : purgeTxViewMenuItem->Enabled = true;
	// ...
}


// [TODO]: in the TXView context menu, the option "Show Tx Receipt..." should be dimmed if >1 tx is selected <--
private: System::Void ViewTxReceiptToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e)
{ // View Tx Receipt context menu item clicked (should only be enabled if one (or more?) items in the TxView are selected.) <--
	if (listview_solutionsview->SelectedItems->Count != 1)
		return;

// [TODO]: check if one is already displayed for this item? don't open another window for same tx concurrently.
// [TODO]: store/retrieve the tx hash from the TxViewItem instead. This works though!
	msclr::interop::marshal_context marshalctx;
	int soln_no = listview_solutionsview->SelectedItems[0]->Index;
	String^ mstr_TxHash = listview_solutionsview->SelectedItems[soln_no]->SubItems[6]->Text;

	const std::string str_txhash = marshalctx.marshal_as<std::string>(mstr_TxHash);
	if (!checkString(str_txhash, 66, true, true)) {	/* tx hash must be 0x + 64 hex characters */
		LOG_F(WARNING, "Bad tx hash. Not opening TxReceiptForm.");
		return;
	}

	std::string str_receipt = Eth_GetTransactionReceipt(str_txhash, soln_no);
	if (!checkErr_b(&str_receipt, true)) {  /* returns false on short strings, errors, empty strings. trim "Error: ". */
		const String^ mstr_ErrMesg = "Error occurred while getting Tx receipt: " + gcnew String(str_receipt.c_str());
		LOG_IF_F(WARNING, NORMALVERBOSITY, "Error occurred getting Tx receipt for Solution # %d: %s ", soln_no, str_receipt.c_str());
		return;
	}

	System::String^ mstr_Receipt = gcnew String(str_receipt.c_str());	// condense this [todo]
	TxReceiptForm^ txRecptForm = gcnew TxReceiptForm(soln_no, listview_solutionsview->SelectedItems[soln_no]->SubItems[6]->Text, mstr_Receipt);
	txRecptForm->Show();	//non-modal. [TODO]: don't allow re-display of the same receipt in multiple forms?
}

private: System::Void PictureBox2_Click(System::Object^ sender, System::EventArgs^ e)
{ // Tokens Graphic Clicked
	if (gVerbosity != V_DEBUG)  return;					// nothing, debug only for now. has weird delay
														// opening the browser in some cases.
	printf("Tokens graphic clicked- Account lookup test \n");
	if (gSoloMiningMode)
	{   // quick sanity check:
		if (gStr_SoloEthAddress.length() != 40/* || gStr_SoloEthAddress == "??"*/ ){
			printf("Solo Mining Ethereum Address appears invalid: %s  (derived) \n", gStr_SoloEthAddress.c_str() );
			MessageBox::Show("Couldn't look up account- is one configured/loaded?", "COSMiC", MessageBoxButtons::OK, MessageBoxIcon::Exclamation);
			return;
		}

		// open chosen block explorer in the system's selected browser w/ the TxHash (TODO: hardcoded to Etherscan for now)
		try {
			String^ MStr = "https://www.etherscan.io/address/" + gcnew String(gStr_SoloEthAddress.c_str());
			System::Diagnostics::Process::Start(MStr);  // TODO: make user-selectable (ETC blockchain etc.)
		} catch (System::ComponentModel::Win32Exception^ other) {
			MessageBox::Show(other->Message, "COSMiC - Exception", MessageBoxButtons::OK, MessageBoxIcon::Hand);
			Console::WriteLine("In PictureBox2_Click() handler: exception thrown launching browser, aborting! ");
			return;
		}
		// end solo mode stuff
	} else {
	 // WIP: pool mode
	 // quick sanity check
		if ( gMiningParameters.mineraddress_str.length() < 42/* || gStr_SoloEthAddress == "??"*/)  /* [TESTME] */
		{
			printf("Solo Mining Ethereum Address appears invalid: %s (derived) [should have `0x`] \n", gStr_SoloEthAddress.c_str());
			MessageBox::Show("Couldn't look up account- is one configured?", "COSMiC", MessageBoxButtons::OK, MessageBoxIcon::Exclamation);
			return;
		}

		try {
			std::string url = gStr_PoolHTTPAddress.substr(0, gStr_PoolHTTPAddress.length() - 5);
			url += "/profile/?address=" + gMiningParameters.mineraddress_str;  // FIXME: won't necessarily work for all pool URLs, make configurable
			String^ MStr = gcnew String( url.c_str() );
			//... +gcnew String(gStr_SoloEthAddress.c_str());
			System::Diagnostics::Process::Start(MStr);  // TODO: make user-selectable (ETC blockchain etc.)
		} catch (System::ComponentModel::Win32Exception^ other){
			MessageBox::Show(other->Message, "COSMiC - Exception", MessageBoxButtons::OK, MessageBoxIcon::Hand);
			Console::WriteLine("In PictureBox2_Click() handler: exception thrown launching browser, aborting! ");
			return;
		}
		// end pool mode stuff
	}
}

private: System::Void Cosmic::CosmicWind::BgWorker_SoloView_ProgressChanged(System::Object^ sender, System::ComponentModel::ProgressChangedEventArgs^ e)
{ //  ProgressPercentage property (int) of event `e` could become a bottleneck here: the max # of Solutions View items we can have is limited
  // (in this implementation) by the size of. consider using a larger type via e->UserState

  // the solutions in gTxViewItems[] have been updated for TxViewItem # `e->progressPercentage`, 
  // now update the UI (listview_solutionsview) item #'s subitems to display their state.
	const int txView_itemNo = e->ProgressPercentage;
	LOG_IF_F( INFO, gVerbosity==V_DEBUG, "Bgworker_SoloView_ProgressChanged: Item # %d "
		" with status code: %d", txView_itemNo, gTxViewItems[txView_itemNo].status);
	
	if (txView_itemNo < 0 || txView_itemNo >= DEF_TXVIEW_MAX_ITEMS) { // ensure txview item # is in valid range
	   //Console::WriteLine("Solution #" + txView_itemNo.ToString() + " not being updated in TxView (bad item# in BgWorker_SoloView_ProgressChanged.");
	   domesg_verb("BgWorker_SoloView_ProgressChanged(): item not updated, item #" + std::to_string(txView_itemNo) + " is invalid.", true, V_NORM);
	   return; }

   if (gTxViewItems[txView_itemNo].status == TXITEM_STATUS_EMPTYSLOT) {
	   printf("BgWorker_SoloView_ProgressChanged(): Item # %d  has illegal status TXITEM_STATUS_EMPTYSLOT! Fixing. \n", txView_itemNo);
	   gTxViewItems[txView_itemNo].status = TXITEM_STATUS_SOLVED;  // <-------
	   return; }

   // update retry count and netNonce columns (from here, the main thread which owns the listview)
  // listview_solutionsview->Items[txView_itemNo]->SubItems[5]->Text = gTxViewItems[txView_itemNo].submitAttempts.ToString();

//if(txView_itemNo > (listview_solutionsview->Items->Count-1)	 { /* item does't exist? */ }
	const int txview_solutionitems = listview_solutionsview->Items->Count;
	if (txView_itemNo < listview_solutionsview->Items->Count) { /* <- TESTME */
		if (gTxViewItems[txView_itemNo].networkNonce != UINT64_MAX)
			listview_solutionsview->Items[txView_itemNo]->SubItems[7]->Text = gTxViewItems[txView_itemNo].networkNonce.ToString();
		 else listview_solutionsview->Items[txView_itemNo]->SubItems[7]->Text = "--";
	} else {
		LOG_F(WARNING, "CosmicWind::BgWorker_SoloView_ProgressChanged(): Not updating non-existent TxView item #%d!", txView_itemNo);
		return;
	} // [TESTME] <--

   if (gTxViewItems[txView_itemNo].status == TXITEM_STATUS_FAILED)
   {
	   listview_solutionsview->Items[txView_itemNo]->BackColor = Drawing::Color::LightSalmon;
	   listview_solutionsview->Items[txView_itemNo]->SubItems[1]->Text = "Failed";
	   gTxView_WaitForTx = NOT_WAITING_FOR_TX;  // stop waiting for the tx: mining will resume
	   
//		ClearSolutionsQueueSlot(...);  //<--[TESTME] ensure the sol'n is popped from q_solutions elsewhere.
	   return;
   }

   if (gTxViewItems[txView_itemNo].status == TXITEM_STATUS_CONFIRMED)
   {
	   domesg_verb("Solution # " + std::to_string(txView_itemNo) + " confirmed by network! ", true, V_NORM);
	   listview_solutionsview->Items[txView_itemNo]->BackColor = Drawing::Color::PaleGreen;
	   listview_solutionsview->Items[txView_itemNo]->SubItems[1]->Text = "Confirmed";

	   //ClearSolutionsQueueSlot( /* was here: slot # */); // <-- old
	   if (DEBUGMODE) { domesg_verb("Not clearing sol'n slot ", false, V_DEBUG); }// <-- DBG only, remove
	   LOG_IF_F(INFO, DEBUGMODE, "Not clearing solo solution # %d yet ", txView_itemNo); // <--
	   // ^ [WIP] check this.

				// gTxView_WaitForTx = NOT_WAITING_FOR_TX;  // redundant: challenge will change if successful mint()
	   return;	// don't resume mining until the new challenge comes in.
   }

   // TODO: the time it was solved, or time it was sent?
   if (gTxViewItems[txView_itemNo].status == TXITEM_STATUS_SUBMITTED)
   { // (WIP)
	   listview_solutionsview->Items[txView_itemNo]->BackColor = Drawing::Color::AliceBlue;
	   listview_solutionsview->Items[txView_itemNo]->SubItems[1]->Text = "Accepted by node";
	 // (WIP)

	   if (checkString(gTxViewItems[txView_itemNo].txHash, 66, true, true)) {	/* expect 256-bit number, 64 hex characters + 0x = 66 */		
		   domesg_verb("Will wait for Sol'n #" + std::to_string(txView_itemNo) + " or new Challenge.", true, V_DEBUG);
		   printf("Sol'n #%d: %s \n", txView_itemNo, gTxViewItems[txView_itemNo].txHash.c_str());
		   gTxView_WaitForTx = txView_itemNo;  // so we can wait for it (1 solution max per Challenge)
		   listview_solutionsview->Items[txView_itemNo]->SubItems[6]->Text = gcnew String(gTxViewItems[txView_itemNo].txHash.c_str());
	   } else { /* auto-resend the transaction (don't clear sol'n buffer slot until confirmed or failed): */
		   printf("TxView Item # %d has no/bad txHash (but no errString!) Reporting 'no txhash'. \n", txView_itemNo);  // <--- WIP
		   printf("TxView Item # %d txhash field is: %s, length is: %zu \n", txView_itemNo, gTxViewItems[txView_itemNo].txHash.c_str(), gTxViewItems[txView_itemNo].txHash.length());
		   listview_solutionsview->Items[txView_itemNo]->SubItems[6]->Text = "err: unknown (no tx-hash received from node) ";
	   } // (WIP) empty txhash- shouldn't happen. don't wait on a tx that didn't send successfully ^

	   //PopSolution();  //ClearSolutionsQueueSlot(gTxViewItems[txView_itemNo].fromSlotNum); // <- (TESTME)
	   // ^ [WIP] the q_solutions item has already been cleared when this item was added. anything else to clean up?
	   return;
   }

   if (gTxViewItems[txView_itemNo].status == TXITEM_STATUS_SUBMITWAIT || gTxViewItems[txView_itemNo].status == TXITEM_STATUS_SOLVED /* <-- new */)
   { // solution that has been solved and is waiting to be sent
	   listview_solutionsview->Items[txView_itemNo]->SubItems[7]->Text = "-";  // Convert::ToString(gTxViewItems[txView_itemNo].networkNonce);  // redundant? <-----
	   listview_solutionsview->Items[txView_itemNo]->BackColor = Drawing::Color::LightGoldenrodYellow;

	   listview_solutionsview->Items[txView_itemNo]->SubItems[1]->Text = gTxViewItems[txView_itemNo].status==TXITEM_STATUS_SUBMITWAIT ? "Ready to Send" : "Solved"; // TXITEM_STATUS_SOLVED
	   
	   // moved here
	   if (!gTxViewItems[txView_itemNo].errString.empty()) {	/* show the TxHash or any potential error encountered while submitting. */
		   listview_solutionsview->Items[txView_itemNo]->SubItems[6]->Text = gcnew String(gTxViewItems[txView_itemNo].errString.c_str());
		   return; } // <-- new
	   else
	   { // if there is no error: txhash should be available now, write it in
		   if (!gTxViewItems[txView_itemNo].txHash.empty())
		   {
			   if (checkString(gTxViewItems[txView_itemNo].txHash, 66, true, true)) /* optionally...*/
			   {
				   if (gVerbosity > V_NORM) { printf("TxView Item # %d:  %s \n", txView_itemNo, gTxViewItems[txView_itemNo].txHash.c_str()); }
				   listview_solutionsview->Items[txView_itemNo]->SubItems[6]->Text = gcnew String(gTxViewItems[txView_itemNo].txHash.c_str()); 	// TX hash if any
				   return;	 // <---
			   } else {
				   LOG_IF_F(WARNING, gVerbosity>=V_NORM, "TxHash failed checkString().");
				   return; }
		   }
		   else
		   { // empty txhash (shouldn't happen)
			   domesg_verb("TxView Item #" + std::to_string(txView_itemNo) + " has no txHash (but no error!)", true, V_DEBUG);
			   listview_solutionsview->Items[txView_itemNo]->SubItems[6]->Text = "err: no txhash received from node ";
			   // TODO: retry sending - (sol'n buffer slot should still be populated! working on this) <--
			   return; // <---
		   }
	   }
	   // moved here
	   // ^ Sloppy. Clean this up, condense if possible [TODO]
	   return;
   }

   if (gTxViewItems[txView_itemNo].status == TXITEM_STATUS_SUBMITTING) {
	   listview_solutionsview->Items[txView_itemNo]->BackColor = Drawing::Color::AliceBlue;
	   listview_solutionsview->Items[txView_itemNo]->SubItems[1]->Text = "Submitting";
	   return;
   }


   if (gTxViewItems[txView_itemNo].status == TXITEM_STATUS_STALE) {
	   listview_solutionsview->Items[txView_itemNo]->BackColor = Drawing::Color::LightGray;
	   listview_solutionsview->Items[txView_itemNo]->SubItems[1]->Text = "Stale or extra";

//	   ClearSolutionsQueueSlot(  /* was here: slot # */);  // free the slot. todo: make accept long type
	   LOG_IF_F(INFO, DEBUGMODE, "Not clearing sol'n slot for stale solution # %d", txView_itemNo); // <-- [WIP] [FIXME]?
// ^^ [CHECK]


	   return;
   } // slot already freed elsewhere?

   if (gTxViewItems[txView_itemNo].status == TXITEM_STATUS_TERMINAL) {
	   listview_solutionsview->Items[txView_itemNo]->BackColor = Drawing::Color::LightGray;
	   listview_solutionsview->Items[txView_itemNo]->SubItems[1]->Text = "Submit failed";
	   domesg_verb("Clearing terminal sol'n #" + std::to_string(txView_itemNo) + ": " +
		   gTxViewItems[txView_itemNo].str_solution + " from queue", true, V_DEBUG);

	   //ClearSolutionsQueueSlot( /* was here: slot # */);  // free the slot. todo: make accept long type?
	   domesg_verb("NOT clearing sol'n slot ", false, V_DEBUG); // <--
	   // ^ ^ [CHECK]
	   //




	   return;
   }   // slot already freed elsewhere?
   else
   { // unknown happened
	   printf("Unhandled status type (%d) in BgWorker_SoloView_ProgressChanged() (main thr) \n", gTxViewItems[txView_itemNo].status);
	   return;
   }
   // (WIP)
}


// IMPT NOTE: don't access the TxView listview control from outside the main thread. This BGworker's "ReportProgress"
//			  method is used to update that control from the thread that "owns" it.
private: System::Void BgWorker_SoloView_DoWork(System::Object^ sender, System::ComponentModel::DoWorkEventArgs^ e)
{
	// [WIP]: see line 4471 etc.: MUTEX LOCK																						// [WIP]
	// don't access the gTxViewItems out of turn (while main thread is!!) <----- (and/or): pass in the params (by value, or ref?) <-- [WIP]

	int solBufSlot = -1;  // null
	unsigned short tx_pending_count{ 0 };  // because bgworker cannot directly alter the listview control (owned by Main Thread). [WIP]: Streamlining TXview.
	bool parse_success{ false };		   // ^ for tracking if a sol'n is pending already for that challenge, don't submit another.

	if (gTxView_WaitForTx != NOT_WAITING_FOR_TX) {
		LOG_IF_F(INFO, HIGHVERBOSITY, "Waiting for solution #%d or next challenge...", gTxView_WaitForTx);	//<-- lower verbosity cutoff?
		CheckStatusOfTransaction(static_cast<unsigned int>(gTxView_WaitForTx));
		//gTxView_WaitForTx = NOT_WAITING_FOR_TX;	//reset: no longer waiting. mining will resume.  <---
		return; }	
	//... [WIP] should also time out waiting for a tx and just resume? new challenge should arrive either way


	// Enforce the max # of items in the ListView !!	<---- [FIXME]
	for (unsigned int i = 0; i < DEF_TXVIEW_MAX_ITEMS; ++i)
	{	// valid txview sol'n slot? matching solution in buffer? (WIP)  NOTE: slot 0 is null. valid txitem #s: 0 thru DEF_TXVIEW_MAX_ITEMS-1.
	
		//if (gTxViewItems[i].fromSlotNum >= 0 && gTxViewItems[i].fromSlotNum < DEF_SOLUTIONSBUFFER_SLOTS)
		//	solBufSlot = gTxViewItems[i].fromSlotNum;  // get the sol'ns buffer slot # for this txView item.		
		
		if (!gTxViewItems[i].slot_occupied)  { continue; }  // any further checks to do?

		if (gTxViewItems[i].status == TXITEM_STATUS_STALE) { continue; }
		// WIP: set _STALE status on challenge change, if solo solution's status is pre-submitted successfully
		// CheckForStalesInTxView() ?

		if (gTxViewItems[i].status == TXITEM_STATUS_EMPTYSLOT) {
			// bgWorker_SoloView->ReportProgress(i);  // <--- FIXME (see bgWorker_SoloView_ProgressChanged() handler)
			continue;  }

		// if sol'n is confirmed/failed by network then we're done processing it, go to next (if any)
		if (gTxViewItems[i].status == TXITEM_STATUS_CONFIRMED || gTxViewItems[i].status == TXITEM_STATUS_FAILED)
			continue;

		  /* <-- hax! pass the parameters to this handler [FIXME] */
		if (gTxViewItems[i].str_challenge != gMiningParameters.challenge_str && gTxViewItems[i].status < TXITEM_STATUS_SUBMITTED)
		{ // * stale! *
			if (gTxViewItems[i].str_challenge.length() == 66) {
				domesg_verb("Solution " + gTxViewItems[i].str_solution + " is stale (challenge changed.) ", true, V_MORE);
				domesg_verb("... from " + gTxViewItems[i].str_challenge + " to " + gMiningParameters.challenge_str + ".", true, V_DEBUG);
				gTxViewItems[i].status = TXITEM_STATUS_STALE;
				//ClearSolutionsQueueSlot(solBufSlot);   // cleared in _ProgressChanged handler of the TXview BGworker (main thread).
				bgWorker_SoloView->ReportProgress(i);  // report this item ready for UI updating (see the BGworker's DoWork handler.)
				continue;  // next item 
			} else {
				if (gVerbosity == V_DEBUG) printf("dbg: not a valid challenge in solution #%d : length is %d \n", i, (int)gTxViewItems[i].str_challenge.length());
				domesg_verb("BgWorker_SoloView_DoWork(): sol'n: " + gTxViewItems[i].str_solution + ", chal: " + gTxViewItems[i].str_challenge, true, V_DEBUG);
				continue; /* TESTME: get rid of this */ }
		}
		// fixme
		if (gTxViewItems[i].status == TXITEM_STATUS_SUBMITTED)
		{  // TxView Item representing sol'n we already sent (we checking its status til confirmed). Its buffer slot has been cleared.		
			tx_pending_count += 1;  // this run

			/* if( */ CheckStatusOfTransaction(i);	/*){ ... } 
			/*	else{ ... } */

			// [OLD STUFF - FIXME] Remove this! <---
			//if the TxView item's associated solutions buffer slot is unoccupied & the sol'n has not been sent already (then it should be.)
			// if (solBufSlot == -1 || !gSolutionsBuffer[solBufSlot].occupied)  // <-- WIP: keeping slot contents til confirm/failed/submit-err
			//continue;			// consider leaving the sol'n buffer slot occupied in case it needs to be resubmitted for some reason, clear it after confirm/fail
		}
		else if (gTxViewItems[i].status == TXITEM_STATUS_SOLVED || gTxViewItems[i].status == TXITEM_STATUS_SUBMITWAIT)  // solved, rdy to send
		{ // solved, not sent yet
			if (gTxViewItems[i].submitAttempts < SET_MAX_TXSEND_ATTEMPTS)
			{	// TODO: .. and update the nonce in the column of the listview when it changes !
				// Put the Network Nonce in the last column. MUST BE DONE FROM MAIN THREAD.
				// .. update the network nonce when ReportProgress if it needs iterating (last node response) <<<<
				
				// this function will update the nonce as necessary based on acct's latest-known Transaction Count and
				// the Network Nonce of the newest solution Tx in the TxView.
				// the txn count for the account will be retrieved right before payload is built & submitted. so, commented out (WIP):
				if (gTxViewItems[i].last_node_response == NODERESPONSE_OK_OR_NOTHING)
				{
					//gTxViewItems[i].networkNonce = Eth_GetTransactionCount();
					//gTxViewItems[i].networkNonce = CheckAgainstUpdateNetworkNonce(gTxViewItems[i].networkNonce);   // OK/No Error or first submission attempt (default value):	
				}
				else if (gTxViewItems[i].last_node_response == NODERESPONSE_NONCETOOLOW || gTxViewItems[i].last_node_response == NODERESPONSE_REPLACEMENT_UNDERPRICED)
				{ // update the network nonce for txview transaction `i` as needed:
					//gTxViewItems[i].networkNonce = Eth_GetTransactionCount();
					//gTxViewItems[i].networkNonce = CheckAgainstUpdateNetworkNonce(gTxViewItems[i].networkNonce);
					// ... (WIP)
				}
				// if ( ...out of gas?... ) {	 etc. ... continue; }

				gTxViewItems[i].submitAttempts += 1;  // move to end?
				// NOT continue;
			}
			else
			{ // max # of submission attempts reached, clear this 'terminal solution' (TODO: save to log to preserve proof-of-work.)
				domesg_verb("Solution  " + gTxViewItems[i].str_digest.substr(0,24) + " reached max # of submit attempts. ", true, V_NORM);
				gTxViewItems[i].status = TXITEM_STATUS_TERMINAL;  // submission failed, won't retry
				bgWorker_SoloView->ReportProgress(i);  // report this item is ready for UI updating (see the BGworker's DoWork handler.)
				continue;
			}  // next item
			 
			//gTxViewItems[i]. networkNonce = gTxCount <---

			// max # of submission attempts has not been reached, so:
			LOG_IF_F(INFO, NORMALVERBOSITY, "Submitting solution # %d from TXview (attempt %d of %d):", i, gTxViewItems[i].submitAttempts, SET_MAX_TXSEND_ATTEMPTS);
			//bgWorker_SoloView->ReportProgress(i);  // update?

			// submit solution:
			if (Solo_SendSolution(i) == 0) { //0: Accepted- clear the SQ slot it occupied. Store/show the TxHash <-- use a common enum
				domesg_verb("Solution #" + std::to_string(i) + " accepted by node.", true, V_NORM);
				gTxViewItems[i].status = TXITEM_STATUS_SUBMITTED;

				//ClearSolutionsQueueSlot( (unsigned short)solBufSlot );  // free the slot (CHANGED: not freeing til confirmed/failed).
				//gTxViewItems[i].submitAttempts = 0;  // reset
				bgWorker_SoloView->ReportProgress(i);  // report this item is ready for UI update
				break;
			} else { // if the submission was not successful (non-zero return code):
				gTxViewItems[i].status = TXITEM_STATUS_SUBMITWAIT;
				domesg_verb("Error sending Solution #" + std::to_string(i) + ": " + gTxViewItems[i].errString, true, V_MORE);
				// note: sol'n buffer slot is not freed, still needed
				bgWorker_SoloView->ReportProgress(i);  // report this item is ready for UI updating

				// handle the error string- behave accordingly for retry. this thread or main one? (wip)
				break;
			}
		}

	} // end of for() loop
	// TODO: profile and condense this monster function ^
}

private: System::Void Timer_txview_Tick(System::Object^ sender, System::EventArgs^ e)
{	// todo: pass in the item # to submit/retry submitting to the BGworker (idea: put the loop here,
	// but run the timer with a long interval.)
	timer_txview->Stop();  // stop timer while we work

	// TODO: finish implementing updates to the gApplicationStatus use
	if (!gCudaSolving /* || gApplicationStatus != APPLICATION_STATUS_MINING */) {
		domesg_verb("Stopping timer_txview because no longer mining ", true, V_DEBUG);
		return;  }

//	add any solutions in the sol'ns buffer to the TxView for handling/display/user interaction.
//	prevent async access to this listview :)

//	experiment with where we update the txn count. calling from *main thread*:
//	if (checkDoEveryCalls(DEF_TIMING_DOEVERYCALLS_GETTXNCOUNT)) gSolo_TxnCount = Eth_GetTransactionCount();
	if (!bgWorker_SoloView->IsBusy)
		bgWorker_SoloView->RunWorkerAsync();  // run the TXhelper BGworker- asynchronously
	 else {
		timer_txview->Start();  // re-start this timer.
		return;  }

	// start it again (if still mining. remember: async)
	if (gCudaSolving) /*(gApplicationStatus == APPLICATION_STATUS_MINING)*/
		timer_txview->Start();
}


private: System::Void enableDisableGpuToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e)
{
	bool gpusDisable{ false };  // if we are enabling or disabling the device(s)
	if (enableDisableGpuToolStripMenuItem->Text->Substring(0, 7) == "Disable")  /* set bool based on menu item text set by	*/
		gpusDisable = true;														/* the contextMenu_gpu open handler			*/

	// replace this, condense. [todo] <--
	if (contextMenu_gpu->Text == "Multiple Devices")  // invisible text of the contextMenu
	{ //multiple devices selected:
		for (unsigned short i = 0; i < MAX_SOLVERS; ++i)
		{
			unsigned short solverNo{ UINT8_MAX };  // 0xFF: not a device#
			if (gpusSummarized[i] == UINT8_MAX)
				continue;  // skip empty device# slots
			solverNo = gpusSummarized[i];

			//gCudaDeviceEnabled[cudaDeviceIndex] = gpusDisable ? false : true;  // across all selected devices
			Solvers[i]->enabled = gpusDisable ? false : true;  // across all selected devices
		}

		// done?
		// return;
	}
	else
	{ //just one device:
		//this->DevicesView->SelectedItems[0]				// <---- Simpler
		System::Byte solverNo{ System::Byte::MaxValue };	//0xFF
		// solverNo = Convert::ToByte(contextMenu_gpu->Text);
		bool parse_ok = System::Byte::TryParse(contextMenu_gpu->Text, solverNo);	//<---- parse device# from undisplayed "Text" property of context menu
		if (parse_ok && solverNo >= 0 && solverNo < (MAX_SOLVERS-1) && Solvers[solverNo] != nullptr) {	//don't access a nonexistant solver
			Solvers[solverNo]->enabled = gpusDisable ? false : true;  // set enabled status based on user selection.	/* gCudaDeviceEnabled[cudaDeviceIndex] */
			// anything else?	[wip].
		} else {
			LOG_F(INFO, "Bad single device# %u - not changing Enabled status." BUGIFHAPPENED, static_cast<unsigned short>(solverNo));
			return;
		}
	//	^ kludgey; replace this. Would use a custom control inheriting ContextMenu to store device#'s, but it _might_ break editing it in the MSVS Designer. [todo / fixme]

	}
}


private: System::Void Timer_cputhreadsview_Tick(System::Object^ sender, System::EventArgs^ e)
{
	timer_cputhreadsview->Stop();  // stop while we work

	const unsigned int numThreads = (unsigned int)nud_numthreads->Value;
	const unsigned int numItems = (unsigned int)threadsListView->Items->Count;
	if (numItems != numThreads) {
		printf("CPUMiningWind: timer1_tick(): number of CPU threads does not match # of items in listView. aborting update. \n");
		timer_cputhreadsview->Start();  // start it again
		return;
	}

	// ..

	// update the threads (rows in the listview `threadsListView`
	double lcl_hashrate = 0;  // init
	for (unsigned int th = 0; th < numItems; ++th)
	{	//cout << "CPU thread # " << th << ": hashrate " << cpuThreadHashRates[th] << " or ";
		lcl_hashrate = cpuThreadHashRates[th] / 1000;
		threadsListView->Items[th]->SubItems[1]->Text = lcl_hashrate.ToString("0.00") + " KH/s";
		threadsListView->Items[th]->SubItems[2]->Text = cpuThreadHashCounts[th].ToString("N0");
		threadsListView->Items[th]->SubItems[3]->Text = cpuThreadSolutionsFound[th].ToString("N0");
		threadsListView->Items[th]->SubItems[4]->Text = "-";  // threadsListView->Items[th]->SubItems[4]->Text = cpuThreadBestResults[th].ToString();
		//threadsListView->Items[th]->SubItems[5]->Text = gCpuThreadSolveTimes[th].ToString("N0") + " secs";  // FIXME: proper time
		// ...etc.
	}

	// just in case.  value set in numericUpDown control:
	nud_updatespeed->Value >= 1 ? timer_cputhreadsview->Interval = (int)nud_updatespeed->Value : timer_cputhreadsview->Interval = 175; // the setting, or the default.

	timer_cputhreadsview->Start();  // start it again
}

private: System::Void splitPanel_SplitterMoved(System::Object^ sender, System::Windows::Forms::SplitterEventArgs^ e)
{
	listbox_events->TopIndex = listbox_events->Items->Count - 1;  // scroll down so latest events visible
}


private: System::Void Label6_Click(System::Object^ sender, System::EventArgs^ e)
{	// label under logo clicked: list devices, enabled status and intensity values to stdout. (log?)
	if (gVerbosity <= V_NORM)
		return;

	printf("Main Mining Window (CosmicWind) form size: (W)%d, (H)%d \n", this->Width, this->Height);
	printf("Splitter Distance: %d, Devices Panel Size: %d, Auxiliary Panel Size: %d \n", splitPanel->SplitterDistance,
		splitPanel->Panel1->Size.Height, splitPanel->Panel2->Size.Height );

	//for (unsigned short solverNo = 0; solverNo < CUDA_MAX_DEVICES; ++solverNo)
	const unsigned short solversAllocated = static_cast<unsigned short>(Solvers.size());	// was: for (s = 0; s < CUDA_MAX_DEVICES; ++s)
	for (unsigned short solverNo = 0; solverNo < solversAllocated; ++solverNo)
	{
		if (Solvers[solverNo] == nullptr) {
			LOG_F(INFO, "Solver # %u does not exist!" BUGIFHAPPENED, solverNo);
			return;
		}

		printf("Device# %u: ", solverNo);
		if (Solvers[solverNo]->enabled) {
			printf("Enabled (");
			printf( "Intensity: %u or threads: %" PRIu32 " )\n", Solvers[solverNo]->intensity, Solvers[solverNo]->threads);  // intensity #
			//	^^	[FIXME]: replace Intensity and Threads with member variables of `genericSolver`-type objects ^^
		} else
			printf("Disabled \n");
	}
}


private: System::Void LogToDiskToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e)
{
	// (TODO)
}

private:  System::Void DevicesView_Click(System::Object^ sender, System::EventArgs^ e)
{
	if (gVerbosity==V_DEBUG)  printf("DevicesView Clicked \n");
}

private: System::Void CosmicWind_ResizeEnd(System::Object^ sender, System::EventArgs^ e)
{ // Resizing Column (Widths) Automatically on Window resize
	// sketch:
	// snap to some default column widths when window is resized

	#define DEVICEVIEW_COLUMNS_AUTORESIZE_THRESHOLD 700  // guess. moveme <--
	//textbox_ethaddress->Size.Width = this->Width - 50;
	//textbox_challenge->Size.Width = this->Width - 50;
	DevicesView->Size = tablelayoutpanel_top->Size;
	// ... (WIP/FIXME)

	//if (autoResizeColumnsToolStripMenuItem1->Checked /* || this->Width < DEVICEVIEW_COLUMNS_AUTORESIZE_THRESHOLD*/)
	//CosmicWind_ResizeStuff();
}

private: System::Void CosmicWind_Click(System::Object^ sender, System::EventArgs^ e)
{  // clicked the form itself
	this->Focus();
	this->DevicesView->SelectedItems->Clear();		  // deselect any items
	this->listbox_events->SelectedItems->Clear();	  // "
}

private: System::Void Listbox_events_Leave(System::Object^ sender, System::EventArgs^ e)
{
	listbox_events->ClearSelected();
	//listbox_events->SelectedItems->Clear();
	//listbox_events->SelectedIndices->Clear();
	if (gVerbosity == V_DEBUG)  printf("Clearing listbox_events selected items (new way) \n");
}

private: System::Void DevicesView_FocusLeave(System::Object^ sender, System::EventArgs^ e)
{
	this->DevicesView->SelectedItems->Clear();			  // deselect any items
	this->DevicesView->FocusedItem->Focused = false;  // try to remove selection 'ghost' outline
}

// CPU Mining start/stop button:
private: System::Void button_cpumine_startstop_Click(System::Object^ sender, System::EventArgs^ e)
{
	const unsigned short maxtarg_exponent{ 234 };  // [TODO]: make maxtarg user-configurable. 0xBTC: 2^234. <--
	nud_numthreads->Enabled = false;  // disable threads numericUpDown control
	label_threads->Enabled = false;	  // and adjacent label.
	ClearCpuMiningVars();			  // see CPUSolver.cpp

	domesg_verb("Cleaning up solutions from last session (CPU mining starting)... ", true, V_NORM);
	ClearTxViewItemsData();	 // clears the listview's associated structure-array's members.
	listview_solutionsview->Items->Clear();
	// ^ [MOVEME]? ^

	// if already mining, stop:
	if (gCpuSolverThreads > 0)
	{
		printf("Stopping CPU mining on %d threads. \n", gCpuSolverThreads);
		button_cpumine_startstop->Text = "Start";
		gCpuSolverThreads = 0;

		label_threads->Enabled = true;
		nud_numthreads->Enabled = true;
		// ... TODO
		return;
	}
	// otherwise...

//	miningParameters initialParameters{};  // <-- [WIP / FIXME]. pass as value?

	// start mining on the # of CPU threads the numericUpDown control specifies
	const unsigned short numThreads = (const unsigned short)nud_numthreads->Value;
	if (!numThreads)  // if <1 thread specified, abort.
		return;

	// if already mining on some CUDA devices vs. starting on the CPU only:
	if (gCudaSolving) {
		GenerateCpuWork( &gMiningParameters );  // [WIP]  generate a midstate/target for the CPU threads to work from the existing hash_prefix
		// (wip)...
	}
	else { /* not yet mining, so: */
		if (DoMiningSetup(menuitem_options_computetarg->Checked, maxtarg_exponent, &gMiningParameters) != 0) { /*  [WIP].  0= OK  */
			LOG_F(WARNING, "Network error encountered contacting the pool.");
			domesg_verb("Network error encountered contacting the pool.", true, V_LESS);  // <- remove?
			return; }

		//if (NewMessage( &gMiningParameters ) != 0) {  /* [WIP] */
		//	MessageBox::Show("Trouble while parsing the Challenge. Please report this bug.", "COSMiC", MessageBoxButtons::OK, MessageBoxIcon::Asterisk);
		//	return;	}
		GenerateCpuWork( &gMiningParameters );  // [WIP] <-

		// start the network BGworker:
		domesg_verb("Starting background thread for network tasks. \n", true, V_MORE);
		if (!NetworkBGWorker->IsBusy)  NetworkBGWorker->RunWorkerAsync();  // if BGworker not running, start it <-
		  else  domesg_verb("Network BGworker already running ", true, V_NORM);
	}

	// TODO: disable main Start/Stop button if user started mining by clicking the CPU Start/Stop button? (wip: streamlining modes).
	if (gSoloMiningMode)
	{	// solo mining-specific only:		
		timer_txview->Enabled = true;			// start timer which launches the TXhelper BGworker and...
		timer_txview->Start();					// ...and starts handling of TXview
		domesg_verb("Starting in Solo mining mode. ", true, V_NORM);
	} else  domesg_verb("Starting in Pool mining mode. ", true, V_NORM);

	//
	// either mode:
	gCpuSolverThreads = numThreads;		 // (todo) check that threads all started? so far so good
	threadsListView->Items->Clear();	 // empty threads list
	unsigned int threadsStarted = 0;
	for (unsigned int thrNo = 0; thrNo < numThreads; ++thrNo)							// populate the listview with our threads as the Items.
	{ // make an item (row) for each thread:
		threadsListView->Items->Add(Convert::ToString(thrNo));					// new item, first "cell" is thread # (internal serial #s 0-up)
		for (unsigned short s = 0; s < threadsListView->Columns->Count; ++s)
			threadsListView->Items[threadsListView->Items->Count - 1]->SubItems->Add("-");  // make empty subitems for each col. of new item
		if (SpawnCpuSolverThread(thrNo) == true)  // launch a thread, check if successful
			threadsStarted += 1;
		else  domesg_verb("CPU solver thread # " + std::to_string(thrNo) + " failed to launch. ", true, V_NORM);
	}

	// TODO: check if all our CPU threads have started (put the references in an array and check ->IsAlive ?)
	if (threadsStarted >= numThreads)  { domesg_verb("All threads launched! ", false, V_DEBUG); }
	
	// only proceed further if mining is really starting <------
	button_cpumine_startstop->Text = "Stop";
	timer_cputhreadsview->Start();  // start timer (triggers updating of listview elements with our threads' stats.)
}

private: System::Void label6_Click(System::Object^ sender, System::EventArgs^ e)
{ // debug info printed to stdout	
	CWind_DebugInfo( &gMiningParameters );  // [TESTME]
}

private: System::Void resetHashrateCalcToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e)
{ // shortcut key: F5
	//UpdateDevicesHashrateCalc();  // updates all GPU(s)' hashrate calculation (do this after resuming from a pause!)
	const unsigned short num_solvers = static_cast<unsigned short>(Solvers.size());
	for (unsigned short solver_no = 0; solver_no < num_solvers; ++solver_no)
		Solvers[solver_no]->ResetHashrateCalc();
}

private: System::Void combobox_modeselect_SelectedIndexChanged(System::Object^ sender, System::EventArgs^ e)
{
	for (unsigned short i = 0; i < CUDA_MAX_DEVICES; ++i) {
		gNum_SolutionCount[i] = 0;
		gNum_InvalidSolutionCount[i] = 0;  // <- anything else? [TODO]
	}

	LOG_IF_F(INFO, HIGHVERBOSITY, "Mode switched: clearing solutions queue and txview... ");	// DEBUGMODE?
	ClearSolutionsQueue();
	//ClearTxViewItemsData();  // <-- here?
	//listview_solutionsview->Items->Clear();

	// Solutions tab only enabled in Solo Mode (for now, anyway)
	if (combobox_modeselect->SelectedIndex == MODE_POOL)
	{ // pool mode selected:
		gSoloMiningMode = false;		// pool mode
		tabSolns->Enabled = false;		// the Solutions tab will be disabled (Pool Mode) (for now.)
		textbox_poolurl->Text = gcnew String( gStr_PoolHTTPAddress.c_str() );  // update textbox w/ pool url
		//
		if (timer_solobalance->Enabled)
			timer_solobalance->Stop();  // stop the Balance retrieval timer if it's running
		if (DevicesView->Columns->Count >= 3)
			DevicesView->Columns[3]->Text = "Valid Shares (inval.) <rej.> ";  // new: separated invalid/rejected by pool counts.
		else {
			MessageBox::Show("DevicesView columns not initialized. Please report this bug.", "COSMiC", MessageBoxButtons::OK, MessageBoxIcon::Error);
			LOG_F(ERROR, "DevicesView has no column 3! Please report this bug."); //<--- 
			return;
		}

		// DEBUG
		menuitem_options_computetarg->Checked = true;  // <---
	}
	else if (combobox_modeselect->SelectedIndex == MODE_SOLO)
	{ // solo mode selected:
		gSoloMiningMode = true;
		tabSolns->Enabled = true;		// the Solutions tab will be available (Solo Mode)
		textbox_poolurl->Text = gcnew String(gStr_ContractAddress.c_str()) + " (Solo Mode)";  // TODO: Show the token name here?
		//
		if (!timer_solobalance->Enabled)  // start getting the Ether and Token balance when this timer's
			timer_solobalance->Start();	  // interval is up (see _Tick event), if it's not already running.
		DevicesView->Columns[3]->Text = "Valid Sol'ns (inval.)";  // no useful metric for rejection/acceptance given other factors

		// DEBUG
		menuitem_options_computetarg->Checked = false;  //?
	}
	else
	{ /* ... */ }
}

private: System::Void DevicesView_GotFocus(System::Object^ sender, System::EventArgs^ e) {
	if (gVerbosity == V_DEBUG) { printf("Clearing Events View listbox selected items, if any (DevicesView got focus). \n"); }
	listbox_events->SelectedIndices->Clear();
}

private: System::Void DevicesView_LostFocus(System::Object^ sender, System::EventArgs^ e)
{
	DevicesView->SelectedItems->Clear();
}

private: System::Void minimizeToTrayToolStripMenuItem_CheckStateChanged(System::Object^ sender, System::EventArgs^ e)
{  //ConfigurationManager::AppSettings["minimizeToTray"] ...
}

private: System::Void lowerPanel_Click(System::Object^ sender, System::EventArgs^ e)
{
#ifdef DEBUGGING_COSMICWIND_GUI
	Console::WriteLine("Clearing listbox and devicesview selected items (lower panel clicked)");
#endif
	listbox_events->ClearSelected();	  // deselect any events in listbox
	DevicesView->SelectedItems->Clear();  // deselect any selected device rows
}

private: System::Void resubmitTransactionToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e)
{
	// todo: dim the item if no txview item(s) selected. make multi-selectable.
	if (listview_solutionsview->SelectedItems->Count < 1)
		return;

	int selected_txview_itemno = listview_solutionsview->SelectedItems[0]->Index;
	//
	gTxViewItems[selected_txview_itemno].submitAttempts = 0;  // a fresh start!
	gTxViewItems[selected_txview_itemno].status = TXITEM_STATUS_SOLVED;	  //
}

private: System::Void checkbox_useCPU_CheckedChanged(System::Object^ sender, System::EventArgs^ e) {
	//panel_cpucontrols->Enabled = checkbox_useCPU->Checked;  //CPU mining panel is enabled if checkbox ticked
}

//
//Retrieve reward amt. in tokens from Contract, print to console <-- [DBG use only]
private: System::Void checkMiningRewardToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
	LOG_IF_F(INFO, DEBUGMODE, "Donations waiting: %d ", gDonationsLeft);
//	const BigInteger MaxVal = BigInteger::Parse("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFE",	
//	System::Globalization::NumberStyles::HexNumber | Globalization::NumberStyles::AllowHexSpecifier);	/* max value when parsing reward from contract */

	/*const*/ BigInteger reward = Solo_GetMiningReward(gStr_ContractAddress);  // get the current token reward (see net_solo.cpp)
	Console::WriteLine("Contract's Token reward (with 8 decimals): " + reward.ToString());
}

private: System::Void CheckStatusOfTransaction(unsigned int solution_no)
{ // solution already sent: check on its status:
	LOG_IF_F(INFO, gVerbosity >= V_MORE, "Checking Tx Status of Solution %d... ", solution_no);	//DEBUGMODE only?
	if (checkDoEveryCalls(Timings::getTxReceipt) == true) {	/* has already been sent: */
		std::string recpt = "";
		bool parse_success{ false };	// redundant?

		domesg_verb("- Checking on status of submitted solution #" + std::to_string(solution_no) +
			" (network nonce " + std::to_string(gTxViewItems[solution_no].networkNonce) + ")... ", false, V_DEBUG);
		recpt = Eth_GetTransactionReceipt(gTxViewItems[solution_no].txHash, solution_no);  // request receipt from node w/ Tx Hash
		if (!checkErr_a(recpt)) {  /* returns false on short strings, errors, empty strings */
			domesg_verb("Error occurred getting Tx Receipt for Solution #" + std::to_string(solution_no) + ": " + recpt, true, V_DEBUG);
			return;	//continue;
		}

		std::string txStatus = ParseKeyFromJsonStr(recpt, "status", 0, solution_no, &parse_success);  // no expected length for value str
		if (!parse_success || !checkErr_a(txStatus)) {  /*  don't trim off "Error: ". `parse_success` redundant? splitting this into its own func. [WIP] */
			printf("%s \n", txStatus.c_str());  // print error string
			return;	//continue;
		}

		if (txStatus == "0x00" || txStatus == "0x0") {
			domesg_verb("Solution in TXView Slot # " + std::to_string(solution_no) +
				"is pending. Confirmation status: " + txStatus, true, V_MORE);	 // spammy
			gTxViewItems[solution_no].status = TXITEM_STATUS_FAILED;			 // <----- WIP/FIXME
			bgWorker_SoloView->ReportProgress(solution_no);						// update listview item
			return;	//continue;
		}

		if (txStatus == "0x01" || txStatus == "0x1") {  // <----- Ditto
			domesg_verb("Solution in TXView Slot # " + std::to_string(solution_no) + " confirmed by network :) ", true, V_MORE);
			gTxViewItems[solution_no].status = TXITEM_STATUS_CONFIRMED;					// no more processing should occur for this sol'n/tx
			++gDonationsLeft;															// one donation will be sent (% of the token reward). TODO: consolidate to less Tx's.
			bgWorker_SoloView->ReportProgress(solution_no);								// update listview item
			// WIP: Ensure the solution is discarded now (already sent & confirmed)
			//++gSolo_TxnCount;  // [FIXME] <---

			return;
		}

		LOG_F(ERROR, "Bad TxStatus in TxHelper BGworker progress handler processing TXview item# %u !", solution_no);	//<--- should never happen
		return;	//continue;
	} //else
		//LOG_IF_F(INFO, DEBUGMODE, "[DEBUG]: NOT checking the status of solution# %u in TXview", std::to_string(solution_no));
}


}; //class CosmicWind



} //namespace Cosmic


#else
#pragma message("Not re-including COSMICWIND.")
#endif

