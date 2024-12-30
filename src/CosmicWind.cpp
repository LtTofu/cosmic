// implementation for Cosmic Windows application, GUI.
// see .h file
#include "CosmicWind.h"
bool gMiningStopStart{ false };		// temp kludge. true if mining is starting/stopping.	[FIXME] <--

#include <sstream>
extern int gSolo_ChainID;	// -or-
//#include <coredefs.hpp>

namespace Cosmic
{
	
	bool CosmicWind::SetUpDevicesView(void)
	{

		DevicesView->Items->Clear();	// [MOVEME]? empty list of items/subitems, keep the pre-defined columns.

	// [MOVEME]:
	/* Set up the custom double-buffered ListView instance's properties, columns etc. */
		DevicesView->Activation = System::Windows::Forms::ItemActivation::OneClick;
		//this->DevicesView->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
		DevicesView->HeaderStyle = System::Windows::Forms::ColumnHeaderStyle::Nonclickable;
		DevicesView->Cursor = System::Windows::Forms::Cursors::Arrow;
		DevicesView->Scrollable = true;
		// new:
		DevicesView->ShowItemToolTips = true;
		DevicesView->AllowColumnReorder = true;
		//
		DevicesView->View = View::Details;
		DevicesView->FullRowSelect = true;
		DevicesView->GridLines = false; // <--- no lines (test)
		DevicesView->HideSelection = false;  // don't show "ghost" when unfocused control
		DevicesView->TabStop = true;
		DevicesView->TabIndex = 23;
		DevicesView->AutoSize = false;
		DevicesView->Scrollable = true;
		DevicesView->Visible = true;  // can now be seen

	// handlers:
	//	DevicesView->SelectedIndexChanged += gcnew System::EventHandler(this, &CosmicWind::devicesListView1_SelectedIndexChanged);
	//	DevicesView->Leave += gcnew System::EventHandler(this, &CosmicWind::deviceslist_events_Leave_1);
		DevicesView->ContextMenuStrip = contextMenu_gpu;  // context menu (right-clicked a GPU in the list, AltGr etc.)

	// DEVICEVIEW COLUMNS:
	// - (WIP) improving how this is done on window setup, resize etc.
		for (unsigned short c = 0; c < 11; ++c) {
			LOG_IF_F(INFO, DEBUGMODE, "adding DevicesView column, count is now: %d", this->DevicesView->Columns->Count);
			this->DevicesView->Columns->Add("Column " + c.ToString());
		}//<---
		//
		this->DevicesView->Columns[0]->Text = "#";
		this->DevicesView->Columns[1]->Text = "GPU Type";
		this->DevicesView->Columns[2]->Text = "Hash Rate";
		this->DevicesView->Columns[3]->Text = "Valid Shares (invalid/stale)";
		this->DevicesView->Columns[4]->Text = "Temp";
		this->DevicesView->Columns[5]->Text = "Power";
		this->DevicesView->Columns[6]->Text = "Fan/Pump";
		this->DevicesView->Columns[7]->Text = "Intensity";
		this->DevicesView->Columns[8]->Text = "Solve Time";
		this->DevicesView->Columns[9]->Text = "Status";
		this->DevicesView->Columns[10]->Text = "Use %";

#ifdef TEST_BEHAVIOR
		LOG_IF_F(INFO, DEBUGMODE, "form size: %d (w)  %d (h)", this->Size.Width, this->Size.Height);					// DBG
		LOG_IF_F(INFO, DEBUGMODE, "splitter position: %d (w)  %d (h)", splitPanel->Size.Width, splitPanel->Size.Height);		// DBG
#endif

		DevicesView->Columns[0]->Width = (int)(DevicesView->Width * 0.025);  // #
		DevicesView->Columns[1]->Width = (int)(DevicesView->Width * 0.175);  // gpu name
		DevicesView->Columns[2]->Width = (int)(DevicesView->Width * 0.1);	 // hash rate
		DevicesView->Columns[3]->Width = (int)(DevicesView->Width * 0.125);  // shares (invalid)
		DevicesView->Columns[4]->Width = (int)(DevicesView->Width * 0.1);    // gpu temp
		DevicesView->Columns[5]->Width = (int)(DevicesView->Width * 0.075);  // power draw
		DevicesView->Columns[6]->Width = (int)(DevicesView->Width * 0.075);  // fan speed
		DevicesView->Columns[7]->Width = (int)(DevicesView->Width * 0.075);  // intensity
		DevicesView->Columns[8]->Width = (int)(DevicesView->Width * 0.1);	 // solve time
		DevicesView->Columns[9]->Width = (int)(DevicesView->Width * 0.1);	 // status
		DevicesView->Columns[10]->Width = (int)(DevicesView->Width * 0.05);  // usage%

		return (DevicesView->Columns->Count == 10);	//true=OK false=Err
	}


	unsigned short CosmicWind::Form_DetectDevices(void)
	{	// [WIP / FIXME]: some functional overlap between this and DetectDevices()! <---
		// [WIP / TODO]: support device types other than CUDA in the Devices List.
		LOG_IF_F(INFO, NORMALVERBOSITY, "Detecting supported hardware...");
		domesg_verb("Detecting supported hardware...", true, V_NORM);			// [todo]:	use log callback.

		unsigned short total_devices{ 0 };

		// === cuda devices: ===
		const unsigned short cuda_devices_count = Detect_CUDA_Devices();
		if (cuda_devices_count)
		{
			total_devices += cuda_devices_count;
			domesg_verb("Success! Detected " + std::to_string(cuda_devices_count) + " device(s).", true, V_NORM);  // not a error (event)
			if (gVerbosity >= V_NORM /*&& !gDialogConfigured */)  // [todo]: only show this message on "first run"
				domesg_verb("Right-click any device in the Devices List to configure its Intensity, etc.", true, V_NORM);
		}
		else {
			domesg_verb("No supported devices were detected. Are recent nVidia graphics drivers installed?", true, V_LESS);  // always
			// [TODO] / [WIP]: don't treat like an error if any other device types present
		}

		// [TODO]: detect other device types. Treat CPU threads as a single solving device.
		//

		return total_devices;

	}//<----- functional overlap with DetectDevices! [FIXME] <-----


	// Currently specific to CUDA devices. Make more generic, support other types [TODO].
	unsigned short CosmicWind::Detect_CUDA_Devices()
	{ // detect multiple CUDA devices, spawn solver objects for them, and add them to the Device List.	[WIP] <---
		const ushort num_cudaDevices = static_cast<unsigned short>(Cuda_GetNumberOfDevices());	// number of devices reported by CUDA API
		bool error{ false };

		if (DEBUGMODE)
			SpawnConsoleOutput();	// <--- REMOVE: this could bring about the end of the world otherwise. :D <------
		
		//	if (num_cudaDevices > (CUDA_MAX_DEVICES - 1))	{ /* [FIXME]: accurate error message for too many devices! <--
		if (num_cudaDevices < 1 || num_cudaDevices > (CUDA_MAX_DEVICES - 1)) { // [CHECKME]. [TODO]: don't show if other suitable devices were found.
			MessageBox::Show("No CUDA devices found. The program will not work correctly! "
				"Try installing the latest nVidia graphics drivers.", "COSMiC", MessageBoxButtons::OK, MessageBoxIcon::Information);
			LOG_IF_F(INFO, NORMALVERBOSITY, "No compatible CUDA devices found. Are any present? \n"
				"Latest nVidia graphics drivers installed?");
			return 0;
		}
		else printf("Detected %u CUDA Device(s).\n", num_cudaDevices);

	//	DevicesView->Items->Clear();	// [MOVEME]? empty list of items/subitems, keep the pre-defined columns.

	// [NEW]
		if (!allocate_gpusolvers_cuda(num_cudaDevices)) {	/* Specify type? */
			LOG_F(ERROR, "allocate_gpusolvers_cuda() failed!");		// TODO: exit?
			error = true;
			//return 0;		//err- no devices
		}
		// NEW
		// WAS HERE: setting properties of Devices List, adding the columns, setting their sizes.

		return num_cudaDevices;
	}



	System::Void CosmicWind::CosmicWind_ResizeStuff() /* [WIP] Called by the form's _ResizeEnd() event handler */
	{
		// [todo / fixme]:  also do this for the CPU threads listview <--
		//
		if (DEBUGMODE) { printf("Auto-resizing %d DeviceView columns \n", DevicesView->Columns->Count); }

		for (int col = 0; col < DevicesView->Columns->Count; ++col)
			DevicesView->Columns[col]->AutoResize(ColumnHeaderAutoResizeStyle::HeaderSize);  // or ::ColumnContent;

		//[todo]: do this only if the window width changed
		//DevicesView->Columns[col]->AutoResize(ColumnHeaderAutoResizeStyle::ColumnContent);
		//unsigned int listViewWidth = DevicesView->Width;

		for (int col = 0; col < threadsListView->Columns->Count; ++col) {
			threadsListView->Columns[col]->AutoResize(ColumnHeaderAutoResizeStyle::HeaderSize);
			DevicesView->Columns[0]->Width = (int)(DevicesView->Width * 0.025);  // #
			DevicesView->Columns[1]->Width = (int)(DevicesView->Width * 0.175);  // gpu name
			DevicesView->Columns[2]->Width = (int)(DevicesView->Width * 0.1);	 // hash rate
			DevicesView->Columns[3]->Width = (int)(DevicesView->Width * 0.1);	 // shares (invalid)
			DevicesView->Columns[4]->Width = (int)(DevicesView->Width * 0.075);  // gpu temp
			DevicesView->Columns[5]->Width = (int)(DevicesView->Width * 0.075);  // power draw
			DevicesView->Columns[6]->Width = (int)(DevicesView->Width * 0.075);  // fan speed
			DevicesView->Columns[7]->Width = (int)(DevicesView->Width * 0.075);  // intensity
			DevicesView->Columns[8]->Width = (int)(DevicesView->Width * 0.075);  // solve time
			DevicesView->Columns[9]->Width = (int)(DevicesView->Width * 0.125);  // status
			DevicesView->Columns[10]->Width = (int)(DevicesView->Width * 0.1);   // usage%
		}

		// [todo]: resize the columns of the Cpu Mining listview, too. <--
	}





	System::Void CosmicWind::Init_PopulateDevicesList(const unsigned short num_devices)
	{
		// [WIP] / [TODO]: making this device type-generic (types other than CUDA.) <---
		// iterate through # of detected devices and populate checkedListBox1
		for (uint8_t deviceNo = 0; deviceNo < num_devices; ++deviceNo)
		{
			// scratchStdString = Cuda_GetDeviceNames(static_cast<int>(deviceNo));			// [OLD]
			const std::string str_devicename = Solvers[deviceNo]->gpuName; // device name from solver class object instead [TESTME]. <--
			LOG_IF_F(INFO, DEBUGMODE, "Device #%u: %s", deviceNo, str_devicename.c_str());	// DBG
		//
			ListViewItem^ newRow = this->DevicesView->Items->Add(deviceNo.ToString());		// device # column
			newRow->UseItemStyleForSubItems = false;										// allow cell style customization
		// [TOOD]: color to indicate type?
			newRow->SubItems->Add(gcnew String(str_devicename.c_str()));
			newRow->SubItems->Add("-");				// hashrate column (placeholder dash.)
			newRow->SubItems->Add("-");				// shares/invalids column (placeholder dash.)
			newRow->SubItems->Add("-");				// temperature column (placeholder dash.)
			newRow->SubItems->Add("-");				// power draw column (placeholder dash.)
			newRow->SubItems->Add("-");				// fan speed column (placeholder dash.)
			newRow->SubItems->Add( Solvers[deviceNo]->intensity.ToString() );		// intensity column [FIXME] <----
			newRow->SubItems->Add("-");				// status column
			newRow->SubItems->Add("Idle");			// solve time column (placeholder dash.)
			newRow->SubItems->Add("-");				// GPU utilization % (includes other applications)
		//	newRow->ToolTipText = "Right-click a device for more options.";

			Solvers[deviceNo]->enabled = true;		// enable by default (new approach, no checkboxes).
		}
		// ^^ MOVE THIS UI STUFF TO ANOTHER FUNC., SEPARATE FROM DETECTION, JUST READ THE DATA FROM OBJECT IN `SOLVERS[]`
		//		AND USE THE # OF DEVICES TO ADD (STARTING FROM SOLVERS[0].) <--- [WIP]
	}


	System::Void CosmicWind::HandleEventsListBox()
	{	// get event string(s) from queue and add to events listbox, w/ date/time it was added to the listbox. [todo / wip]: time it was spawned.
	//			[note]:			Note: the time it was spawned may be recorded in the log file.  Max event length:  ... <--- [fixme]
		std::string str_event{};

		while (GetEventStr(&str_event))
		{ // add any event str's returned. false: break out.
			if (str_event.empty()) {
				LOG_F(WARNING, "Empty event string in HandleEventsListBox().");
				break;
			}

			if (str_event.length() > EVENTTEXT_MAXLENGTH)
				str_event = str_event.substr(0, EVENTTEXT_MAXLENGTH);  // enforce max event length	[todo/fixme]: use window width?

			listbox_events->Items->Add(DateTime::Now.ToString("[dd MMM HH:mm:ss] ") + gcnew String(str_event.c_str()));  // [wip]
			if (listbox_events->Items->Count >= DEF_EVENTS_LISTBOX_CAPACITY)		/* check if the listbox has too many items (for performance reasons) */
				listbox_events->Items->RemoveAt(0);									// remove oldest line
	//	-or-	listbox_events->TopIndex = listbox_events->Items->Count - 1;  // and ->Refresh()? scroll listbox down as new items are added
		} //while

		if (!str_event.empty()) { /* if one or more mining events were added to the control, this string will still be populated */
			listbox_events->TopIndex = listbox_events->Items->Count - 1;	// scroll listbox down
			listbox_events->Refresh();										// we want this updated immediately [TESTME].
		}
	}

	// NET annunciator: enabled when NetworkBGWorker is active. text color  red: network error, rust: high latency.
	System::Void CosmicWind::Update_NET_Annunciator()
	{
		unsigned int scratchInt{ 0 };
		if (!stat_net_last_success && !gNotifiedOfSolSubmitFail)
			statusbar_anncNET->ForeColor = System::Drawing::Color::Red;  // only once per net error :)
		else
		{ // rust color to NET annunciator sublty indicates less-than-optimal network latencies
			if ((gSoloMiningMode && stat_net_lastpoolop_totaltime > 600) ||
				(!gSoloMiningMode && stat_net_lastpoolop_totaltime > 400)) /* ms */
				statusbar_anncNET->ForeColor = Drawing::Color::RosyBrown; // High Latency
			else   statusbar_anncNET->ForeColor = Drawing::Color::Black;  // OK
		}

		// Warning: some time measurements for LibCurl operations are only reliable if last operation returned "OK"!
		//			so, does not display invalid values to the user.
		StringBuilder^ sb = gcnew StringBuilder("Previous request: ", 150);  // capacity=150 characters
		if (stat_net_last_success == true)
		{ // last curl operation was successful:
			if (gSoloMiningMode) {  // solo mode:
									//MStr = "Previous request: " + gcnew String( str_net_lastcurlresult.c_str() ) +
									//	"\nAverage operation time: " + stat_net_avgethop_totaltime.ToString("0.0") + " ms";
				sb->Append(str_net_lastcurlresult.c_str())->Append("\nAverage request time: ");		// legal param to Append?
				sb->Append(Math::Round(stat_net_avgethop_totaltime))->Append(" ms");  // <--			// "
			}
			else { // pool mode:
				//MStr = "Previous request: " + gcnew String( str_net_lastcurlresult.c_str() ) +
				//	"\nAverage request time: " + stat_net_avgpoolop_totaltime.ToString("0.0") + " ms";
				sb->Append(str_net_lastcurlresult.c_str())->Append("\nAverage request time: ");
				sb->Append(Math::Round(stat_net_avgpoolop_totaltime))->Append(" ms");  // todo <--  .ToString("0.0") -alike
			}
		}
		else { // last network operation NOT successful:
			if (!gNotifiedOfSolSubmitFail)  /* only once per net error */
				sb->Append(str_net_lastcurlresult.c_str());
		}

		//... TODO (improve usefulness of this tooltip.)
		scratchInt = (unsigned int)(q_solutions.size());  // any solutions in queue (such as due to network/pool outage)?
		if (scratchInt) {  /*  # of solutions queued >0?  */
			//tooltip_NET->SetToolTip(lbl_net, MString + "\n\n" + scratchInt.ToString() + " solutions ready to send." );  }  //MString += "\n\n" + sols_queued.ToString() + " solutions queued to send";
			sb->Append("\n\n");
			sb->Append(scratchInt);
			sb->Append(" solutions queued to send.");
		}
		else {
			sb->Append("\n(No solutions queued to send.)");
			//tooltip_NET->SetToolTip(lbl_net, sb->ToString());  /* + "\n(No solutions queued to send)" */);
		}

		//tooltip_NET->SetToolTip( lbl_net, sb->ToString() );  // old
		statusbar_anncNET->ToolTipText = sb->ToString();	  // set tooltip txt from stringbuilder
	}


	// args: cuda device# and DevicesView subitem #. blanks that info cell with "-"  (.clear()?)
	__inline void CosmicWind::ClearDevicesListSubItem(const short row, const short col)
	{
		if (gVerbosity == V_DEBUG) {
			if (DevicesView->Items[row]->SubItems[col]->Text->Length)
				DevicesView->Items[row]->SubItems[col]->Text = "";
		}
		else {
			if (DevicesView->Items[row]->SubItems[col]->Text != "-")
				DevicesView->Items[row]->SubItems[col]->Text = "-";
		}
	}


	System::Void CosmicWind::GetSolutionForTxView()	/* <- rename, condense w/ calling func timer1_Tick() ?    */
	{
		LOG_IF_F(INFO, DEBUGMODE, "GetSolutionForTxView(): running ");
		if (q_totxview.empty())		//<-- redundant
			return;					//<-- better practice to check after acquiring mtx_totxview?

		unsigned int solution_no{ 0 }/*, howmany{0}*/;	// respectively: solution # indexed from 0 (TXVIEW)
														//and # of sol'ns added this function run <-- remove?

		// [TODO/WIP] make sure gTxViewItems[] cannot be accessed by more than one thread at a time.
		//			  such as: main thread (for directly editing the UI), network-bgworker or txview-bgworker. <--
		txViewItem temp_txn{};
		bool soln_ready = GetTxViewItem(&temp_txn, &solution_no);
		//	while (GetTxViewItem(&temp_txn, &solution_no))		/* while any solution item #'s remain in queue */
		if (!soln_ready)
			return;

		if (solution_no >= DEF_TXVIEW_MAX_ITEMS) {
			LOG_F(ERROR, "item# %u from q_totxview is out of range.", solution_no);
			domesg_verb("item #" + std::to_string(solution_no) + " from q_totxview is out of range ", true, V_NORM);
			return;
		}

		//		if (...) { continue; }   // if txview array slot not populated with a valid solution? [WIP]
		LOG_IF_F(INFO, DEBUGMODE, "adding sol'n nonce %s ", temp_txn.str_solution.c_str());  // <-- 

				   // TODO: use mtx_txview here? don't let main thread & network thread access gTxViewItems[] at once <--- 
		if (temp_txn.slot_occupied && temp_txn.str_solution.length() == HEXSTR_LEN_UINT256)
		{
			++gTxViewItems_Count;  // <--
			if (!gTxViewItems_Count || gTxViewItems_Count >= DEF_TXVIEW_MAX_ITEMS) { /* <-- beware async access */
				domesg_verb("Too many solutions in TxView- not adding solution ", true, V_NORM);
				LOG_F(WARNING, "The Solutions View (txview) is full. Cannot add item #%d.", solution_no);
				// [TODO / FIXME]: clear the txview solutions list? <---
				return;
			}  // "just in case"- the max # to displayed solution tx's is set to a reasonable value (adjust it).
		}

		String^ mstr_solnstring_scratch = gcnew String(gTxViewItems[solution_no].str_solution.c_str());	// beware cross-thread access to gTxViewItems[] and gTxViewItems_Count <--
		ListViewItem^ newItem = listview_solutionsview->Items->Add(mstr_solnstring_scratch);	// sol'n nonce string
		gTxViewItems_Count = listview_solutionsview->Items->Count;						// <--
		newItem->SubItems->Add("Solved");													// status of the sol'n
		newItem->SubItems->Add(Convert::ToString(temp_txn.deviceOrigin));					// device # (or CPU thread, make this clear in UI) [TODO]
		newItem->SubItems->Add(gcnew String(temp_txn.str_challenge.c_str()));				// challenge it was solved for <--
		newItem->SubItems->Add(solution_no.ToString());									// Solution# (gTxViewItems[] element #, not listview item index#.)
		newItem->SubItems->Add("0");														// Attempts: initially `0`
		newItem->SubItems->Add("-");														// Tx Hash (or error) from the node
		newItem->SubItems->Add("-" /*Convert::ToString(temp_txn.networkNonce)*/);			// can't know the Network Nonce we'll use til submit time <--
 // [TODO]: time solved column?
 //		gSolutionsBuffer[slotNo].solution_no = lcl_itemNum;  // and the other way around

 //		const unsigned int itemNo = (unsigned int)(listview_solutionsview->Items->Count - 1);  // "Solution #" in the Solutions View
 //		gTxViewItems[itemNo].txHash.clear();
 //		gTxViewItems[itemNo].errString.clear();
 //		gTxViewItems[itemNo].status = TXITEM_STATUS_SOLVED; <-- do in another func w/ lock for mtx_txview?
 //		++howmany;

 //	Ensure the TXviewItem is initialized. <---

		LOG_IF_F(INFO, DEBUGMODE && soln_ready, "GetSolutionForTxView() finishing, added "/* + std::to_string(howmany) + "%d */ "solution to view"/*, howmany*/);
		// ^ [WIP / FIXME]: not practical to solve multiple solutions to one challenge, because the challenge will change after a solution is minted for it.
		// Consider just writing the extra solutions data to a log file, adding them to the total sol'ns found, and discarding them (not added to the TXview.)
	}


	// bridge out ahead


	// ... [todo]
	// returns true if a config error was encountered

//using namespace System;
//using namespace System::Configuration;

System::Boolean CosmicWind::CosmicWind_ReadConfiguration(const unsigned short cuda_devicecount)
{
	IFormatProvider^ numberFormat = gcnew CultureInfo("en-US");
	System::Configuration::Configuration^ configHndl = ConfigurationManager::OpenExeConfiguration(ConfigurationUserLevel::None);
	IFormatProvider^ iFormatProv = gcnew Globalization::CultureInfo("en-US");
	bool configuration_issue_detected = false;			// will spawn a dialog box if true
	// xml keys and scratch vars:
	String^ theDeviceIntensityKey = "";			// cuda device `i` intensity setting (not thread count)
	String^ keyname_hwmonenable = "";			// cuda device `i` hw health monitor on/off
	String^ keyname_minrpm = "";				// - cuda device `i` minimum fan/pump rpm (value)
	String^ keyname_minrpm_enable = "";			// - cuda device `i` minimum fan/pump rpm alarm on/off
	String^ keyname_maxgputemp = "";			// - cuda device `i` gpu max temp threshold (value)
	String^ keyname_maxgputemp_enable = "";		// - cuda device `i` gpu max temp alarm on/off
	bool parse_ok = false;

	msclr::interop::marshal_context marshalctx;  // for native<->managed marshalling
	std::stringstream sst;		 // new: re-using `sst` (cleared).

	UInt32 scratchUInt{ 0 };
	double scratchDoub{ 0 };
	unsigned int minRpmValue{ 0 };	 // value
	int maxGpuTempValue{ 0 };		 // "

	LOG_IF_F(INFO, HIGHVERBOSITY, "Reading Configuration...");
	//
	if (ConfigurationManager::AppSettings["PoolURL"]) {
		String^ poolurl = ConfigurationManager::AppSettings["PoolURL"];
		gStr_PoolHTTPAddress = marshalctx.marshal_as<std::string>(poolurl);
	}

	// VERBOSITY SETTING
	//
	scratchUInt = 0;
	printf("Reading Verbosity from config... \n");
	if (ConfigurationManager::AppSettings["Verbosity"])
	{ // key found
		if (UInt32::TryParse(ConfigurationManager::AppSettings["Verbosity"], scratchUInt))
		{ // parse successful
			gVerbosity = static_cast<ushort>(scratchUInt);
			LOG_IF_F(INFO, NORMALVERBOSITY, "set Verbosity Level: %u \n", gVerbosity);
			if (gVerbosity == V_DEBUG)
				SpawnConsoleOutput();	// stdout "debug" window
		}
		else { /* parse err- set default: */
			LOG_F(WARNING, "Couldn't parse Verbosity setting from Config. Using default: %d. \n", DEFAULT_VERBOSITY);
			gVerbosity = DEFAULT_VERBOSITY;
		}
	}
	else { gVerbosity = DEFAULT_VERBOSITY; }  // if key not found
 //

	if (ConfigurationManager::AppSettings["MinerEthAddress"]) {
		String^ mineraddr = ConfigurationManager::AppSettings["MinerEthAddress"];
		//params->mineraddress_str = marshalctx.marshal_as<std::string>(mineraddr);
		gStr_MinerEthAddress = marshalctx.marshal_as<std::string>(mineraddr);
		LOG_IF_F(INFO, gVerbosity > V_NORM, "Loaded miner address %s from Config.", gStr_MinerEthAddress.c_str()); //[moveme]?
	}

	// (SOLO) NETWORK ACCESS INTERVAL (in ms). for accessing an API endpoint
	//
	scratchUInt = 0;
	printf("Reading Solo Network Access Interval from config... \n");
	if (ConfigurationManager::AppSettings["soloNetInterval"])
	{ // key found
		if (UInt32::TryParse(ConfigurationManager::AppSettings["soloNetInterval"], scratchUInt))
		{ // parse successful
			if (scratchUInt < 375 || scratchUInt > 2000) /* check for invalid value */
				scratchUInt = 400;						 // apply default
			gSoloNetInterval = scratchUInt;				 // save config's net-interval value
			if (gVerbosity > V_NORM)  printf("Set Network Access Interval to: %d ms \n", scratchUInt);
		}
		else { // parse err, use default
			printf("Couldn't parse Solo Mode Network Access Interval from Config. Using default: %d ms. \n", DEFAULT_NETINTERVAL_SOLO);
			gSoloNetInterval = DEFAULT_NETINTERVAL_SOLO;
		}
	}
	else {
		gSoloNetInterval = DEFAULT_NETINTERVAL_SOLO;
	}  // if key not found

// NETWORK ACCESS INTERVAL (in ms).
//
	scratchUInt = 0;
	printf("Reading Network Access Interval from config... \n");
	if (ConfigurationManager::AppSettings["NetworkInterval"])
	{ // key found
		if (UInt32::TryParse(ConfigurationManager::AppSettings["NetworkInterval"], scratchUInt))
		{ // parse successful
			if (scratchUInt < 375 || scratchUInt > 2000) /* check for invalid value */
				scratchUInt = 500;						 // apply default
			gNetInterval = scratchUInt;					 // save config's net-interval value
			if (gVerbosity > V_NORM)  printf("Set Network Access Interval to: %d ms \n", scratchUInt);
		}
		else { /* parse err- use default: */
			printf("Couldn't parse Network Access Interval from Config. Using default: %d ms. \n", DEFAULT_NETINTERVAL_POOL);
			gNetInterval = DEFAULT_NETINTERVAL_POOL;
		}
	}
	else {
		gNetInterval = DEFAULT_NETINTERVAL_POOL;
	}  // if key not found

// DIFFICULTY UPDATE FREQUENCY (every X network access intervals).
//
	scratchUInt = 0;
	if (gVerbosity > V_NORM) { printf("Reading Difficulty Update Frequency from config... \n"); }
	if (ConfigurationManager::AppSettings["DiffUpdateFreq"])
	{ /* key found: */
		if (UInt32::TryParse(ConfigurationManager::AppSettings["DiffUpdateFreq"], scratchUInt))
		{ /* parse succeeded: */
			if (scratchUInt < 1 || scratchUInt > 2000)		 /* range check */
				scratchUInt = DEFAULT_DIFFUPDATE_FREQUENCY;  // apply default
			gDiffUpdateFrequency = scratchUInt;				 // apply config's diff update rate
			doEveryCalls_Settings[Timings::getDifficulty] = gDiffUpdateFrequency;  // <- any other options to update? [TODO]
			if (gVerbosity > V_NORM) { printf("Set Difficulty Update Frequency to %d. \n", scratchUInt); }
		}
		else { /* parse err: */
			printf("Couldn't parse Difficulty Update Frequency from Config. Using default: %d. \n", DEFAULT_DIFFUPDATE_FREQUENCY);
			gDiffUpdateFrequency = DEFAULT_DIFFUPDATE_FREQUENCY;
		}
	}
	else { gDiffUpdateFrequency = DEFAULT_DIFFUPDATE_FREQUENCY; } // if key not found

// AUTO-DONATE % (default/min: 1.5%)
//
	scratchDoub = 0;
	if (gVerbosity > V_NORM) { printf("Reading Auto-Donation Percent from config. \n"); }
	// read auto-donate % from Configuration (new: uses TryParse expecting US-style decimal.)
	if (ConfigurationManager::AppSettings["AutoDonatePercent"])
	{ // key found:
		if (Double::TryParse(ConfigurationManager::AppSettings["AutoDonatePercent"],
			Globalization::NumberStyles::Number | NumberStyles::AllowDecimalPoint, numberFormat, scratchDoub))
		{ // parse successful:
			printf("Set Auto-Donation to %f \n", scratchDoub);
			gAutoDonationPercent = scratchDoub;  // contains % setting range check
			parse_ok = true;
		}
		else { domesg_verb("error parsing auto-donate %. ", false, 3); }
	}
	if (!parse_ok) {
		printf("Couldn't parse Auto-Donation Percent: defaulting to %f. \n", DEFAULT_AUTODONATE_PERCENT);
		gAutoDonationPercent = DEFAULT_AUTODONATE_PERCENT;
	}
	parse_ok = false;

	//
	// NETWORK NODE (API endpoint) - HTTP(S) address. Note: Infura requires https://
	if (gVerbosity > V_NORM)  printf("Reading Solo Network Node URL from Config... \n");
	if (ConfigurationManager::AppSettings["soloNetworkNode"])  /* load node address from Configuration: */
		gStr_SoloNodeAddress = marshalctx.marshal_as<std::string>(ConfigurationManager::AppSettings["soloNetworkNode"]);

	//
	// CHAIN ID (for solo mining mode)
	if (ConfigurationManager::AppSettings["chainID"])
	{   // found key `chainID` in the Configuration
		int lcl_chainid = CHAINID_ETHEREUM;
		if (int::TryParse(ConfigurationManager::AppSettings["chainID"], lcl_chainid)) {
			if (gVerbosity > V_NORM)  printf("Read ChainID %d from Configuration. \n", lcl_chainid);
			gSolo_ChainID = lcl_chainid;  // parse to int successful: set chainID globally
			parse_ok = true;
		}
	}
	if (!parse_ok) {
		printf("Couldn't parse chainID from Config. Using default chainID: %d (Ethereum MainNet). \n", CHAINID_ETHEREUM);
		gSolo_ChainID = CHAINID_ETHEREUM;
	}
	parse_ok = false; // reset

//
// GAS LIMIT AND GAS PRICE
// (get hex representations that node/network will expect (RLP encoding done at submit-time with rest of payload.)
	printf("Reading User Gas Price from Config. \n");
	if (double::TryParse(ConfigurationManager::AppSettings["userGasPrice"],
		Globalization::NumberStyles::Number | NumberStyles::AllowDecimalPoint, iFormatProv, scratchDoub))
	{ // parse of gasprice key to double (scratchvar) successful :)
		gU64_GasPrice = (uint64_t)(scratchDoub * 1000000000);  // gwei to wei  (todo: consider just storing it as wei in Config.)
		if (gVerbosity == V_DEBUG)  printf("Dbg: parsed gasprice from Config: %" PRIu64 " \n", gU64_GasPrice);
	}
	else { // parse err:
		gU64_GasPrice = DEFAULT_GASPRICE;
		printf("Error parsing userGasPrice from Config: setting default.  %" PRIu64 " wei \n", gU64_GasPrice);
	}
	//
	// integer gaslimit to hexstring of bytes:
	domesg_verb("Reading User Gas Limit from Config.", false, V_DEBUG);  //<--
	if (uint64_t::TryParse(ConfigurationManager::AppSettings["userGasLimit"], Globalization::NumberStyles::Integer, iFormatProv, gU64_GasLimit))
		domesg_verb("Dbg: parsed gaslimit from Config: " + std::to_string(gU64_GasLimit), false, V_DEBUG);  // <- parse successful :)
	else { // err:
		gU64_GasLimit = DEFAULT_GASLIMIT;
		domesg_verb("Error parsing userGasLimit from Config.  Setting default:  " + std::to_string(gU64_GasLimit) + " units", true, V_NORM);
	}
	// gasprice and gaslimit settings loaded, now populate the hex string equivalents (this saves work when encoding Tx's).
	// gas price (to hexstr):
	sst << std::setfill('0') << std::setw(2) << std::hex << gU64_GasPrice;		// <-- [FIXME] this could be improved
	gStr_GasPrice_hex = sst.str();
	if ((gStr_GasPrice_hex.length() % 2) != 0)
		gStr_GasPrice_hex = "0" + gStr_GasPrice_hex;  // if uneven length, pad w/ 0
	domesg_verb("Hex gasprice: " + gStr_GasPrice_hex, true, V_DEBUG);  // DBG

	sst.clear();
	// gas limit (to hexstr):
	std::stringstream sst2;
	sst2 << /*std::setfill('0') << std::setw(2) << */std::hex << gU64_GasLimit;		// <-- [FIXME] ^
	gStr_GasLimit_hex = sst2.str();
	if ((gStr_GasLimit_hex.length() % 2) != 0)
		gStr_GasLimit_hex = "0" + gStr_GasLimit_hex;  // if uneven length, pad w/ 0
	domesg_verb("Hex gaslimit: " + gStr_GasLimit_hex, true, V_DEBUG);  // DBG

//
// CONTRACT ADDRESS (for solo mode. default to 0xBTC 0xb6... contract)
	if (ConfigurationManager::AppSettings["soloContractAddress"])  // if key present in Config:
		gStr_ContractAddress = marshalctx.marshal_as<std::string>(ConfigurationManager::AppSettings["soloContractAddress"]);
	else {
		gStr_ContractAddress = "0xB6eD7644C69416d67B522e20bC294A9a9B405B31";  // redundant
		printf("Couldn't find Contract address in Config. Using default contract: %s (0xBitcoin). \n", gStr_ContractAddress.c_str());
	}

	// populate devices listbox
	//	Form_DetectCudaDevices();

	//
	// PER-DEVICE SETTINGS FROM CONFIGURATION:
	// acting as if all the Solvers are CUDA devices. The #s indexed from 0 should coincide. <--- [WIP / FIXME].
	scratchUInt = 0;
	if (gVerbosity > V_NORM)  printf("Reading per-device settings from Configuration. \n");

	for (uint8_t i = 0; i < CUDA_MAX_DEVICES; ++i)							/* FIXME: MAX_SOLVERS instead. */
	{ // retrieve cuda device `i`'s intensity from config file, w/ default:
		if (Solvers[i] == nullptr) {
			LOG_F(ERROR, "Solver# %u does not exist !", i);
			continue;
		}
		if (!gCudaDevicesDetected[i]) { //don't look for configuration items for device # `i`.
			if (gVerbosity > V_NORM) {
				LOG_F(INFO, " CUDA Device  ID # %d  [ not installed ] ", i);
				continue;	// skip.  [idea]: devices indexed from 0, ordered by heuristic. so if device ID# `i` not installed, 
			}				//					we could `break;` here as no further devices should be installed.
		}
		scratchUInt = 0;  // cleared scratch var before parse
		LOG_IF_F(INFO, HIGHVERBOSITY, "Reading settings for CUDA Device# %u.", i);
		//
		theDeviceIntensityKey = "cudaDevice" + Convert::ToString(i) + "Intensity"; // key to get, ex. "cudaDevice2Intensity"
		if (ConfigurationManager::AppSettings[theDeviceIntensityKey]) { //key found
			if (UInt32::TryParse(ConfigurationManager::AppSettings[theDeviceIntensityKey], scratchUInt))
				Solvers[i]->intensity = (unsigned int)scratchUInt;  //successful
			else {
				printf("Couldn't parse Device# %u's Intensity setting from Config. Using default: %u. \n", i, DEFAULT_CUDA_INTENSITY);
				Solvers[i]->intensity = DEFAULT_CUDA_INTENSITY;	//redundant
			}
		}
		else { //new device? add the intensity key for it (default value).
			Solvers[i]->intensity = DEFAULT_CUDA_INTENSITY;
			configHndl->AppSettings->Settings->Add(theDeviceIntensityKey, Convert::ToString(DEFAULT_CUDA_INTENSITY));  // add key to Configuration
		}
		
		Solvers[i]->SetIntensity();	//Cuda_UpdateDeviceIntensity(i);

		// retrieve the HW Monitoring settings from config file, or use defaults.
		// build key strings for setting and device # `i`.
		keyname_hwmonenable = "cudaDevice" + Convert::ToString(i) + "HWmon";					// setting: device i hw health monitoring on/off
		keyname_minrpm = "cudaDevice" + Convert::ToString(i) + "MinRPM";						// ...device i's minimum fan/pump rpm threshold  (value)
		keyname_minrpm_enable = "cudaDevice" + Convert::ToString(i) + "MinRPMEnable";			// ...device minimum fan/pump rpm alarm          (toggle)
		keyname_maxgputemp = "cudaDevice" + Convert::ToString(i) + "MaxGPUTemp";				// ...device gpu max temp threshold		(value)
		keyname_maxgputemp_enable = "cudaDevice" + Convert::ToString(i) + "MaxGPUTemp_Enable";  // ...device gpu max temp alarm			(toggle)
		bool bScratch{ false };

		// [TODO] condense this with a common function <--
		// read per-Device HW Monitoring Toggle setting:
		if (ConfigurationManager::AppSettings[keyname_hwmonenable])
		{ // key found! read it out
			if (bool::TryParse(ConfigurationManager::AppSettings[keyname_hwmonenable], bScratch))
				gWatchQat_Devices[i].watchqat_enabled = bScratch;  // use Configuration value
			else {
				printf("parse error\n");
				gWatchQat_Devices[i].watchqat_enabled = true;    // parse err? default: ON for this device.
				configuration_issue_detected = true;
			}
		}
		else { /* didn't find this key: */
			LOG_IF_F(WARNING, NORMALVERBOSITY, "Couldn't parse Device# %u's HW Monitoring toggle from Config. Using default: ON.", i);
			gWatchQat_Devices[i].watchqat_enabled = true;
			configHndl->AppSettings->Settings->Add(keyname_hwmonenable, "True");  // add the key to Configuration (on)
			configuration_issue_detected = true;
		}

		// Per-Device Max GPU temp alarm (en/disable) setting:
		if (ConfigurationManager::AppSettings[keyname_maxgputemp_enable])
		{ // key found!
			if (bScratch.TryParse(ConfigurationManager::AppSettings[keyname_maxgputemp_enable], bScratch))  // key exists, get its value
				gWatchQat_Devices[i].health_maxgputemp_enable = bScratch;
			else {
				printf("parse error\n");
				gWatchQat_Devices[i].health_maxgputemp_enable = true;  // default: ON
				configuration_issue_detected = true;
			}
		}
		else { // didn't find key
			LOG_IF_F(WARNING, NORMALVERBOSITY, "Couldn't parse Device# %u's Max GPU Temp Alarm Toggle from Config. Using default: ON.", i);
			gWatchQat_Devices[i].health_maxgputemp_enable = true;
			configHndl->AppSettings->Settings->Add(keyname_maxgputemp_enable, "True");  // add the key to Configuration (on)
			configuration_issue_detected = true;
		}

		// Per-Device Minimum RPM alarm (Toggle) setting:
		if (ConfigurationManager::AppSettings[keyname_minrpm_enable])
		{ // key found!
			if (bScratch.TryParse(ConfigurationManager::AppSettings[keyname_minrpm_enable], bScratch))  // key exists, get its value
				gWatchQat_Devices[i].health_minrpm_enable = (bool)bScratch;  // .NET bool% to bool
			else {
				gWatchQat_Devices[i].health_minrpm_enable = false;  // default: OFF <---
				configuration_issue_detected = true;
			}
		}
		else { /* didn't find key: */
			LOG_IF_F(WARNING, NORMALVERBOSITY, "Couldn't parse device# %u's Minimum RPM Alarm Toggle from Config. Using default: OFF.", i);
			gWatchQat_Devices[i].health_minrpm_enable = false;
			configHndl->AppSettings->Settings->Add(keyname_minrpm_enable, "False");  // add the key to Configuration (off)
			configuration_issue_detected = true;
		}

	// Minimum Fan/Pump RPM (Value) setting loaded
		if (ConfigurationManager::AppSettings[keyname_minrpm]) { //key found
			if (minRpmValue.TryParse(ConfigurationManager::AppSettings[keyname_minrpm], minRpmValue) == true)	/* parse key to uint	*/
				gWatchQat_Devices[i].health_minrpm = minRpmValue;												/* OK: store globally	*/
			else {
				printf("Error while parsing Min. Fan/Pump RPM for CUDA Device # %d. Defaulting to %d. ", i, DEFAULT_HWMON_MINIMUMRPM);
				configuration_issue_detected = true;
			}
		}
		else { /* didn't find key: */
			printf("Couldn't find Configuration key for CUDA Device # %d's Minimum RPM. Defaulting to %d. \n", i, DEFAULT_HWMON_MINIMUMRPM);
			gWatchQat_Devices[i].health_minrpm = DEFAULT_HWMON_MINIMUMRPM;
			configHndl->AppSettings->Settings->Add(keyname_minrpm, Convert::ToString(DEFAULT_HWMON_MINIMUMRPM));  //add the key to Configuration (800 RPM)
			configuration_issue_detected = true;
		}

		// Max GPU Temperature setting (Value) loaded:
		if (ConfigurationManager::AppSettings[keyname_maxgputemp])
		{ // key found!
			if (int::TryParse(ConfigurationManager::AppSettings[keyname_maxgputemp], maxGpuTempValue) == true)
				gWatchQat_Devices[i].health_maxgputemp = maxGpuTempValue;
			else {
				printf("parse error\n");
				gWatchQat_Devices[i].health_maxgputemp = DEFAULT_HWMON_MAXGPUTEMP;
				configuration_issue_detected = true;
			}
		}
		else {  // didn't find this key
			printf("Couldn't find Configuration key for CUDA Device # %d's Maximum GPU Temperature. Defaulting to %d. \n", i, DEFAULT_HWMON_MAXGPUTEMP);
			gWatchQat_Devices[i].health_maxgputemp = DEFAULT_HWMON_MAXGPUTEMP;
			configHndl->AppSettings->Settings->Add(keyname_maxgputemp, Convert::ToString(DEFAULT_HWMON_MAXGPUTEMP));  // add the key to the Configuration (89 C)
			configuration_issue_detected = true;
		}

	} // end of per-device config `for` loop.

	// all per-device configuration settings retrieved (or defaults loaded), reports to user via a dialog box if any errors occurred.
	// test build dialog box, or "configuration issue detected" dialog (should not appear in release builds, unless the .Config file
	// was manually edited.) The application will automatically use defaults if any value's setting is absent or can't be parsed.
	if (configuration_issue_detected) {
#ifdef TEST_BEHAVIOR
		String^ testbuildstr = "This is a TEST build of COSMiC V4, a preview of \n"
			"features in upcoming releases. Certain functionality \n"
			"may be incomplete or not work as expected. Please \n"
			"read the provided documentation and obtain the latest\n"
			"stable build available. Thank you! \n";
		MessageBox::Show(testbuildstr, "COSMiC", MessageBoxButtons::OK, MessageBoxIcon::Information);	// <-- For Test Builds Only! (FIXME)
#else
		String^ conferrorstr = "Error(s) occurred while reading the Configuration. \n\n"
			"Usually, this means the .Config file is corrupted, \n\n"
			"or is not in the expected format.\n\n"
			"Please restore COSMiC from the original archive.\n\n"
			"If you continue to receive this message, please contact the developer.\n";
		MessageBox::Show(conferrorstr, "COSMiC", MessageBoxButtons::OK, MessageBoxIcon::Information);
#endif
	}

	// save out any updates to the Configuration:
	LOG_F(INFO, "Saving Configuration. ");
	configHndl->Save(ConfigurationSaveMode::Modified);
	ConfigurationManager::RefreshSection("appSettings");

	LOG_F(INFO, "Saved the Configuration. ");
	return configuration_issue_detected;
}
// [TODO]: slim down and break up this old monster function.
//




} //namespace Cosmic
