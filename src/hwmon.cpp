#pragma once
// hwmon.cpp : Hardware Monitoring for CUDA devices ("Watchqat") thread
// 2020 LtTofu

//		Hardware monitoring thread for COSMiC!  uses NVML and NVAPI to provide useful hardware feedback
//		such as GPU temperatures, power usage, fan tach. readings and utilization. Enables automatic pausing
//		if device readings exceed user-configurable temp/fan alarm thresholds (with reasonable defaults).

// [TODO]: Needs general cleanup/revision and logging calls in some places.

#include <stdio.h>
#include <string>
//#include <mutex>
#include <inttypes.h>	//format specifiers for uintXX_t types used in InitHardwareMonitoring()

//#include <nvml.h>
//#include <nvapi.h>
#define HWMON
#include "hwmon.h"

#include <loguru/loguru.hpp>
#include "defs.hpp"
#include "util.hpp"

#define SILENCE_NVML_NVAPI_ERRORS

extern ushort gVerbosity;
extern ushort gApplicationStatus;
extern bool gCudaSolving;

#include "generic_solver.hpp"
// -or- 
// class genericSolver;
// #include <vector>
// extern std::vector<genericSolver*> Solvers;

// includes class definition of cudaDevice and std::vector `CudaDevices`:
// #include "cuda_device.hpp"						// [WIP]. Redundant?

//
//extern unsigned int gWatchQat_DeviceFanSpeedsInPercent[CUDA_MAX_DEVICES];
//extern unsigned int gWatchQat_DeviceFanSpeedsInPercent[CUDA_MAX_DEVICES];

//using namespace System::Threading;
//using namespace System::Text;
//using namespace System;
//using namespace System::Windows::Forms;

// see WQdevice definition in `defs.h`. <--
#include "coredefs.hpp"
struct WQdevice gWatchQat_Devices[CUDA_MAX_DEVICES];  // the element # = the CUDA device index#.
//extern std::vector<cudaDevice*> CudaDevices;		//coredefs.cpp

// globals
bool gDeviceManuallyUnpaused[CUDA_MAX_DEVICES]{ 0 };		   // device has been manually unpaused
int gCudaDevicePciBusIDs[CUDA_MAX_DEVICES] = { 0 };

// prototypes and forward declarations
void Cuda_StoreDevicePciBusIDs(); // hashburner.cu
void WatchQat_Loop(void);
std::string Cuda_GetDeviceBusInfo(int devIndex); // forward declaration. could rework this to return a C-style string instead. std::string/String^ simpler for UI use
void AddEventToQueue(const std::string theText);	// see CosmicWind.cpp
bool NVML_DoCall(/*const*/ nvmlReturn_t nvmlResult, const std::string theTask, const unsigned short deviceNo);  // WatchQat.cpp



// [REF]:	https://docs.microsoft.com/en-us/cpp/preprocessor/managed-unmanaged?view=msvc-160
// When applying these pragmas:
//
//	- Add the pragma preceding a function, but not within a function body.
//	- Add the pragma after #include statements.Don't use these pragmas before #include statements.
// compile with: /clr

//#pragma managed(push, off)
void WatchQat_CleanUp_Device(const unsigned short deviceNo, const bool clearCallErrs, const bool resetToDefaults)
{	// ...
	// proper initialization for nvapi_handle, nvml_handle and nvapi_thermals? [todo] <--

//	LOG_IF_F(INFO, DEBUGMODE, "Cleaning up WatchQat device %d.", deviceNo);

	// if TRUE, watchqat will retry these calls & re-enable display of the readings when available
	if (clearCallErrs) {
		gWatchQat_Devices[deviceNo].errcnt_getfanpct = 0;
		gWatchQat_Devices[deviceNo].errcnt_getfanrpm = 0;
		gWatchQat_Devices[deviceNo].errcnt_getpower = 0;
		gWatchQat_Devices[deviceNo].errcnt_gettemp = 0;
		gWatchQat_Devices[deviceNo].errcnt_getutilization = 0;
	}

	// clear the "current" readings:
	gWatchQat_Devices[deviceNo].utilization = -1;  // int (-1=null)
	gWatchQat_Devices[deviceNo].fanspeed_rpm = 0;
	gWatchQat_Devices[deviceNo].fanspeed_percent = 0;
	gWatchQat_Devices[deviceNo].powerdraw_w = 0;
	gWatchQat_Devices[deviceNo].gputemp = -99;	   // int (-99=null)
	
	// reset all the alarm en/disable and thresholds to defaults:
	if (resetToDefaults) {
		gWatchQat_Devices[deviceNo].health_minrpm = DEFAULT_HWMON_MINIMUMRPM;	   // the minimum fan/pump speed alarm threshold (RPMs)
		gWatchQat_Devices[deviceNo].health_maxgputemp = DEFAULT_HWMON_MAXGPUTEMP;  // the GPU temp alarm threshold (degrees C)
		gWatchQat_Devices[deviceNo].health_maxgputemp_enable = true;			   // TRUE if the maximum GPU temp alarm is on
		gWatchQat_Devices[deviceNo].health_minrpm_enable = false;				   // TRUE if min fanspeed alarm is on (FIXME: off by default)

		gWatchQat_Devices[deviceNo].watchqat_enabled = true;  // default ON
		gWatchQat_Devices[deviceNo].doevery = 0;  // reset
	}

	// gWatchQat_Devices[i].___ = ___;  <---- any others (todo/wip)
}

//#pragma managed(push, off)
void WatchQat_CleanUp_All_Devices(const bool clearCallErrs, const bool resetToDefaults)
{
	LOG_IF_F(INFO, DEBUGMODE, "Cleaning up hardware monitor stuff: \n");
	const unsigned short solvers_count = static_cast<const unsigned short>(Solvers.size());
//	for (unsigned short i = 0; i < CUDA_MAX_DEVICES; ++i)
	for (unsigned short i = 0; i < solvers_count; ++i)
	{	/* was: gCudaDeviceEnabled, and MAX_CUDA_DEVICES. */
		if (Solvers[i]->enabled && gWatchQat_Devices[i].watchqat_enabled/*?*/) {	/* Check this. Hwmon device #s should match Solver #s. [TODO] */
			LOG_IF_F(INFO, DEBUGMODE, "cuda device #%d", i);
			WatchQat_CleanUp_Device(i, clearCallErrs, resetToDefaults);				/* [FIXME]: Clean up the WQ device even if it's been disabled since starting. */
		}
	}
}

// INITHARDWAREMONITORING: Gets handles and sets up HW monitoring thru NVML and NVAPI
//#pragma managed(push, off)
short InitHardwareMonitoring()
{
	// nvapi handles local to THIS FUNCTION only
	NvPhysicalGpuHandle local_nvapiDeviceHandles[NVAPI_MAX_PHYSICAL_GPUS];
	NvAPI_Status theStatus = NVAPI_OK; // local result code. inited.
	nvmlReturn_t nvmlret = NVML_SUCCESS; // local result code. inited.
	
	// tally up any devices we couldn't get
	unsigned short associatedDevices{ 0 }, devices_enabled{ 0 },
		nvml_handles_not_retrieved{ 0 }, enabled_cuda_devices{ 0 };
	NvU32 local_devicesfound{ 0 };		   // local scratch for # of devices found
	NvU32 theBusID = 0;					   // local scratch for pci bus id
	uint8_t i = 0;						   // loop counter for NVAPI devices
	uint8_t i2 = 0;						   // loop counter for CUDA devices (association w/ CUDA device indices)

	const unsigned short num_of_solvers = Solvers.size();
	for (i = 0; i < num_of_solvers; ++i)
		if (Solvers[i]->enabled) { ++enabled_cuda_devices; }	// count enabled devices

	Cuda_StoreDevicePciBusIDs();  // CUDA-side function, retrieves them for gCudaDevicePciBusIDs[]

	// initialize NVML, if error end this function early
	LOG_IF_F(INFO, HIGHVERBOSITY, "Initializing NVML...");
	nvmlret = nvmlInit ();
	if (nvmlret == NVML_SUCCESS)
		LOG_IF_F(INFO, HIGHVERBOSITY, "NVML Initialized!");
	else {
		LOG_IF_F(WARNING, "NVML Error (%s). Ending WatchQat setup.", nvmlErrorString(nvmlret));
		return -1;
	}

	// initialize NvAPI, if error end this function early.
	printf("Initializing NVAPI...\n");
	if (NvAPI_Initialize() == NVAPI_OK)
		LOG_IF_F(INFO, /*HIGH*/NORMALVERBOSITY, "NVAPI initialized!\n");
	else {
		printf("Error initializing NVAPI. Ending WatchQat setup.\n"); // TODO: show error string
		return -1;
	}

	printf("Enumerating NVAPI devices (getting handles)... ");
	theStatus = NvAPI_EnumPhysicalGPUs(local_nvapiDeviceHandles, &local_devicesfound);
	if (theStatus != NVAPI_OK)
		printf("Error enumerating NVAPI devices.\n");
	else
		printf("successful. Found %d physical NVAPI devices.\n", (int)local_devicesfound);

	if (local_devicesfound > CUDA_MAX_DEVICES) {
		LOG_F(WARNING, "NVAPI detected more devices than supported (%d). Hardware Monitoring functionality will be affected.", CUDA_MAX_DEVICES);	//FIXME?
		local_devicesfound = CUDA_MAX_DEVICES; // not all devices' NVAPI calls will work. TODO: improve this - currently 19 devices max
		// [TODO]: report this to user in UI !!
	}

	// redundant?
	//WatchQat_CleanUp_All_Devices(true, false);	//clear call err counters, don't reset devices to default settings.

	// retrieve all the NVML device handles using PCI Bus IDs from CUDA device props
	//const unsigned short num_of_solvers = static_cast<const unsigned short>( Solvers.size() );
	for (i = 0; i < num_of_solvers; ++i)	/* was	CUDA_MAX_DEVICES */
	{
		if (!Solvers[i]->enabled)			/* was	if (!gCudaDeviceEnabled[i])	continue; */
			continue; // skip disabled device

		 // get string as C-style as NVML API expects
		if ( NVML_DoCall( nvmlDeviceGetHandleByPciBusId_v2( Cuda_GetDeviceBusInfo(i).c_str(), &gWatchQat_Devices[i].nvml_handle),
			"getting device nvml handle", i ))  // call successful:
			 printf("WatchQat: retrieved NVML handle for device with PCI Bus ID %s. \n", Cuda_GetDeviceBusInfo(i).c_str() );
		 else
			nvml_handles_not_retrieved += 1;

		// set up the NV_GPU_THERMAL_SETTINGS structures (NVAPI)
		gWatchQat_Devices[i].nvapi_thermals.version = NV_GPU_THERMAL_SETTINGS_VER_2;							// difference?
		gWatchQat_Devices[i].nvapi_thermals.sensor[0].controller = NVAPI_THERMAL_CONTROLLER_GPU_INTERNAL;
		gWatchQat_Devices[i].nvapi_thermals.sensor[0].target = NVAPI_THERMAL_TARGET_GPU;
	}
	
	if (nvml_handles_not_retrieved == 0) LOG_IF_F(INFO, HIGHVERBOSITY, "Got NVML handles.");
	 else LOG_IF_F(WARNING, NORMALVERBOSITY, "Couldn't retrieve %u NVML handles. Hardware monitoring will be affected.", nvml_handles_not_retrieved);

	// loop through the detected NVAPI physical devices. Match them up to CUDA devices.
	for (i = 0; i < num_of_solvers; ++i)		/* was: `i < local_devicesfound`	*/
	{
		if (!Solvers[i]->enabled)				/* was: if (!gCudaDeviceEnabled[i]) */
			continue; // skip disabled device

		theStatus = NvAPI_GPU_GetBusId(local_nvapiDeviceHandles[i], &theBusID);
		if (theStatus != NVAPI_OK)
			LOG_IF_F(WARNING, NORMALVERBOSITY, "NVAPI Device # %u: couldn't get PCI BusID!", i);
		else
			LOG_IF_F(INFO, HIGHVERBOSITY, "NVAPI Device # %u with PCI BusID: 0x%" PRIx32 " ", i, static_cast<uint32_t>(theBusID));	//DEBUGMODE

		// 'i' is the NVAPI index. 'i2' is the CUDA device index # it's tested against
		//for (i2 = 0; i2 < CUDA_MAX_DEVICES /*gdevicesstarted*/; ++i2)
		for (i2 = 0; i2 < num_of_solvers; ++i2)
		{
		//	if (DEBUGMODE) { printf("Associating: NVAPI Device %d PCI Bus ID %d == "
		//							"CUDA Device# %u's PCI Bus ID %x ? \n", i, theBusID, i2, gCudaDevicePciBusIDs[i2] );
			if (gCudaDevicePciBusIDs[i2] == theBusID) {
				gWatchQat_Devices[i2].nvapi_handle = local_nvapiDeviceHandles[i];		// store handle globally... for now [WIP].
				printf("Associated with CUDA Device index %u.\n", i2);
				associatedDevices += 1;
				break; // end loop upon associating solving device with NVAPI device#.	// [WIP]: make HW monitoring code support non-CUDA GPUs. <--
			}
		}
	}

	// did we get them all?
	if (associatedDevices < enabled_cuda_devices) // cast from NvU32 to short uint
	{
		AddEventToQueue("Warning: Didn't associate some NVML/NVAPI devices. Hardware monitoring might be affected.");
		printf("Not all devices could be associated. NVAPI functionality may not work.\n");
	}
	else
	{
		if (gVerbosity > V_NORM)  AddEventToQueue("NVML/NVAPI association successful.");
		printf("%d NVAPI devices associated.\n", (int)associatedDevices);
	}

	// ...
	// TODO: return nonzero on error (# of devices not associated or -1 for NVAPI general failure)
	return 0; // no error
}


// NvAPI_DoCall and NVML_DoCall(): Space-saving functions. Write errors to stdout automatically. Optionally in Events listbox (see gVerbosity).
//									2nd arg (string) is a description of the task/call, to make log messages more useful.
// - won't keep performing a call if it fails repeatedly e.g. "not supported" (MAX_ nvml or nvapi errors enforced).
// - disable repeating the malfunctioning calls, on a per-device basis. (example: "Unsupported")					[WIP]
//#pragma managed(push, off)
bool NvAPI_DoCall(/*const*/ NvAPI_Status nvapiResult, const std::string theTask, const unsigned short deviceNo)
{
	if (nvapiResult == NVAPI_OK)
		return true;   // OK!

// otherwise
	char nvapiResultString[80] = "";
	NvAPI_GetErrorMessage(nvapiResult, nvapiResultString);	 // get relevant error, if any
	domesg_verb("NvAPI Error while " + theTask + ": " + std::string(nvapiResultString), true, V_MORE); // could be spammy
	LOG_F(WARNING, "NvAPI error while %s: %s", theTask.c_str(), nvapiResultString);
	return false;  // some error
}


// FIXME: applies to both these functions: gWQ_SuppressErrors_... will silence errors if
//		  a call fails, but keep trying it next time (more elegant handling TODO)
//#pragma managed(push, off)
bool NVML_DoCall(/*const*/ nvmlReturn_t nvmlResult, const std::string theTask, const unsigned short deviceNo)
{
	if (nvmlResult == NVML_SUCCESS)
		return true;  // OK
	else { //to C-style char array:
		const char *nvmlResultString = nvmlErrorString(nvmlResult);		// char nvmlResultString[80]
		if (DEBUGMODE) {
#ifndef SILENCE_NVML_NVAPI_ERRORS
			//const std::string cppStr = std::string(nvmlResultString);
			domesg_verb("NVML Error while " + theTask + " (CUDA device #" + std::to_string(deviceNo) + "): " + 
				std::string(nvmlResultString), true, V_MORE); // potentially spammy otherwise
#endif
		}
		LOG_IF_F(WARNING, HIGHVERBOSITY, "NVML Error while %s (CUDA Device # %u): %s ", theTask.c_str(), deviceNo, nvmlResultString);
		//else { ... }
	}

	return false;  // some error
}

//
// WATCHQAT_LOOPFUNCTION: Runs in its own CPU thread (ThreadW). Performs continual HW Monitoring tasks.
//#pragma managed(push, off)

using namespace System::Threading;
//#pragma managed
void WatchQat_Loop(void)
{
	nvmlUtilization_t deviceUtilization;					// for  nvmlDeviceGetUtilizationRates()
	unsigned int l_power_mw{ 0 }, l_fanspeed_percent{ 0 };  // board power (mW), fan setting in %.
	NvU32 l_fanspeed_rpm_nvu32 = 0;							// tachometer reading in rpm (an NvAPI Uint32)
	unsigned int l_fanspeed_rpm = 0;
	unsigned short i = 0;									// regular loop counter

	// start of H/W Health thread  ("Watchqat")
	InitHardwareMonitoring();

	// stop hardware monitoring thread if the application is closing.
	while (gCudaSolving == true && gApplicationStatus != APPLICATION_STATUS_CLOSING)
	{
		//NvAPI_GPU_GetVbiosVersionString(gWatchQat_Devices[i].nvapi_handle ... )
		const unsigned short num_of_solvers = static_cast<const unsigned short>( Solvers.size() );
		for (i = 0; i < num_of_solvers; ++i)	/* was: `i < CUDA_MAX_DEVICES` */
		{
			if (Solvers[i]->enabled == false || Solvers[i]->pause != PauseType::NotPaused)
				continue; // skip inactive or paused device

		// [WIP]: ignore excessive errors in the event of not supported, outdated runtimes etc. (see the NVML_DoCall() and NvAPI_DoCall() functions.)
		// TODO/WIP: condense devices whose health WQ monitors into an array of structures CUDA_MAX_DEVICES wide. <-- soon

		// === A. GETTING GPU TEMPERATURE ===
		// TODO: look into reading VRM temperatures on cards which expose those sensors thru NVML/NVAPI. Implement with NVML?
		// TODO: consider REDUNDANT temperature reading using NVML and NvAPI.
			if (gWatchQat_Devices[i].errcnt_gettemp < MAX_NVML_NVAPI_CONSECUTIVE_CALL_ERRS)  // check GPU temperature every run
			{
				// REF: 2nd arg could also be: NVAPI_THERMAL_TARGET_NONE, _MEMORY, _POWER_SUPPLY, _BOARD, _VCD_* ...
				if (NvAPI_DoCall(NvAPI_GPU_GetThermalSettings(gWatchQat_Devices[i].nvapi_handle, NVAPI_THERMAL_TARGET_NONE,
					&gWatchQat_Devices[i].nvapi_thermals), "getting device " + std::to_string(i) + " thermals", i)) /* SensorIndex:0 */
				{ // if GPU is over the set temperature max, and device not manually force-unpaused by user after an alarm
					gWatchQat_Devices[i].gputemp = static_cast<int>(gWatchQat_Devices[i].nvapi_thermals.sensor[0].currentTemp);  /* NvS32 to int */
					// REF: other members include: .sensor[*], .Controller, .defaultMaxTemp, .defaultMinTemp, .Target.
				} else
					gWatchQat_Devices[i].errcnt_gettemp += 1;	// an error occurred
			}

		// CHECK FOR OVERHEAT (unless GPU temp alarm disabled, manually unpaused after an alarm, etc.)
		// (...) don't do the next check if there's not an up-to-date GPU temp reading.		// [WIP] <---
			if (gWatchQat_Devices[i].gputemp >= gWatchQat_Devices[i].health_maxgputemp &&
				gWatchQat_Devices[i].health_maxgputemp_enable && !gDeviceManuallyUnpaused[i])
			{
				// [WIP]: support device types other than CUDA.
				Solvers[i]->pause = PauseType::GPUTempAlarm;	// kernel launching will stop, user informed
				domesg_verb("WatchQat: CUDA GPU # " + std::to_string(i) + " exceeded temperature limit of " +
					std::to_string(gWatchQat_Devices[i].health_maxgputemp) + " C - Pausing Device !", true, V_LESS); // always show
			}
		//else { /* no max temp setting */ }
		//}

		// === B. GETTING FAN SETTING IN % ===  (temporarily disabled)
		// We do this first so we can check goal speed in % when checking for fan failure in tach reading code below.
		// This requires NVML because nVidia apparently doesn't expose fan speed in % ("goal speed") via NVAPI
		 
			/*	if (NVML_DoCall(nvmlDeviceGetFanSpeed(gWatchQat_DeviceHandles_NVML[i], &l_fanspeed_percent),
					"getting cuda device #" + std::to_string(i) + " fanspeed %", i))
					  gWatchQat_DeviceFanSpeedsInPercent[i] = l_fanspeed_percent;  // success:
				 else gWatchQat_DeviceFanSpeedsInPercent[i] = -999;  // failed, set null value
			} else  gWatchQat_DeviceFanSpeedsInPercent[i] = 0; */

		// === C. GETTING TACHOMETER READINGS ===
		// TODO: check the fan% and verify any suspected fan fail, in which case just inform user w/ a color change or similar
		//       but don't pause the device. Account for fans that gradually ramp up as heat increases by default.
			if ((gWatchQat_Devices[i].doevery % 2 != 0) && gWatchQat_Devices[i].errcnt_getfanrpm < MAX_NVML_NVAPI_CONSECUTIVE_CALL_ERRS)  // half the time
			{
				if (NvAPI_DoCall(NvAPI_GPU_GetTachReading(gWatchQat_Devices[i].nvapi_handle, &l_fanspeed_rpm_nvu32),
					"getting CUDA device # " + std::to_string(i) + " tachometer reading", i))
				{ // call successful
					l_fanspeed_rpm = (unsigned int)l_fanspeed_rpm_nvu32;  // NvU32 to uint
					gWatchQat_Devices[i].fanspeed_rpm = l_fanspeed_rpm;
				}
				 else
				 {
					 gWatchQat_Devices[i].fanspeed_rpm = 0;
					 gWatchQat_Devices[i].errcnt_getfanrpm += 1;
				 }
			}

			// CHECK FOR ANY SUSPECTED FAN FAILURE:
			// if tachometer reading (in RPMs) is below user-specified/default minimum, & device not manually unpaused by user:
			// if a fan minimum speed threshold is set (0=no fan failure detect). Don't do this check if problem reading Fan RPMs.
			if (gWatchQat_Devices[i].errcnt_getfanrpm < MAX_NVML_NVAPI_CONSECUTIVE_CALL_ERRS)
			{
				if (l_fanspeed_rpm < gWatchQat_Devices[i].health_minrpm && gWatchQat_Devices[i].health_minrpm_enable && !gDeviceManuallyUnpaused[i])
				{
					// [WIP]: support device types other than CUDA
					//if (gWatchQat_DeviceFanSpeedsInPercent[i] > 50){  .. check if fan speed % is set low (FIXME)
					Solvers[i]->pause = PauseType::FanAlarm;	// kernel launching will stop, user informed
					LOG_F(WARNING, "Device# %u fan below alarm threshold (%u RPM) - Pausing Device ! \n", i, gWatchQat_Devices[i].health_minrpm);
				}
			}

		// === D. GETTING POWER DRAW ===
		// read the GPU power draw (using NVML). According to nVidia, the exposed power (in milliwatts) is for the entire graphics board,
		// not just the GPU. nV also lists accuracy as +/- 5% on FERMI devices. For newer? When known, I will update this comment accordingly
		// as it needs to be revealed to the user somewhere obvious but unobtrusive, particularly for those calculating ROI.

			//if (gCuda_Pause) ...  <--- stray?

			// idea: get pwr reading for device `i` every other run
			//if ( (gWatchQat_Devices[i].doevery % 2 != 0) && gWatchQat_Devices[i].errcnt_getpower < MAX_NVML_NVAPI_CONSECUTIVE_CALL_ERRS)  // other half
			if ((gWatchQat_Devices[i].doevery % 2 == 0 /* || gWatchQat_Devices[i].doevery == 7*/) && gWatchQat_Devices[i].errcnt_getpower < MAX_NVML_NVAPI_CONSECUTIVE_CALL_ERRS)
			{
				if (NVML_DoCall(nvmlDeviceGetPowerUsage(gWatchQat_Devices[i].nvml_handle, &l_power_mw),
					"getting cuda device #" + std::to_string(i) + " power usage", i))
					gWatchQat_Devices[i].powerdraw_w = (double)l_power_mw / (double)1000;  // mW->Watts (decimal)
				else
				{ // err:
					gWatchQat_Devices[i].powerdraw_w = -1; // <---
					gWatchQat_Devices[i].errcnt_getpower += 1;
				}
			}
			// ...

			// TODO: This is an NVML call but is outside the lower check if HW monitoring's enabled for testing reasons
			// TODO: consider getting the utilization via NVAPI instead for lower CPU utilization. Bad NVML! bad!
			// update GPU utilization % column ... about 2x a second (at 100ms WatchQat thread sleep time after each loop).
			if ((gWatchQat_Devices[i].doevery == 3 || gWatchQat_Devices[i].doevery == 7) &&
				gWatchQat_Devices[i].errcnt_getutilization < MAX_NVML_NVAPI_CONSECUTIVE_CALL_ERRS)
			{
				if ( NVML_DoCall(nvmlDeviceGetUtilizationRates(gWatchQat_Devices[i].nvml_handle, &deviceUtilization),
					"getting device utilization", i ) )
					gWatchQat_Devices[i].utilization = (int)deviceUtilization.gpu;
				else {
					gWatchQat_Devices[i].utilization = -1;
					gWatchQat_Devices[i].errcnt_getutilization += 1;  }
			}

			// timing stuff:
			gWatchQat_Devices[i].doevery += 1;  // 1 cycle complete
			if (gWatchQat_Devices[i].doevery >= 10)
				gWatchQat_Devices[i].doevery = 0;  // reset
			//if (gVerbosity == V_DEBUG)  printf("device # %d .doevery: %d \n", i, gWatchQat_Devices[i].doevery);
		} // for() loop end

		// done- WatchQat's CPU thread sleeps (and count * of 5 cycles, reset .doevery if 4+)
		Thread::Sleep(DEF_WATCHQAT_THREAD_SLEEP_MS);  // was 90ms (TODO: configurable).	//[TODO]: cross-platform sleep function and/or
																						//		  ... an alternate threading lib.
	}  // while() loop end

	// Thread ending:
	if (gVerbosity > V_NORM)  printf("# WatchQat: Shutting down NVML... ");
	if ( NVML_DoCall(nvmlShutdown(), "shutting down NVML", -1) )  // not device-specific
		domesg_verb("NVML shutdown successful.", true, V_MORE);
	
	// Clean up (done elsewhere)
	// WatchQat_CleanUp_All_Devices(true, false);

	// unload NvAPI:
	if ( NvAPI_DoCall(NvAPI_Unload(), "unloading NvAPI library", -1) )   // not device-specific
		printf("Unloaded NvAPI successfully. \n");
	return;
}


//using namespace System::Threading;
using namespace System::Text;
using namespace System;
using namespace System::Windows::Forms;

// ThreadW: for WatchQat

public ref class ThreadW
{
public:
	ThreadW(void) // function parameters would go here
	{ // Constructor
	}

	void WatchQatThreadEntryPoint()
	{
		//String^ threadName = Thread::CurrentThread->Name;
		loguru::set_thread_name("Hardware Monitor");
		LOG_F(INFO, "Hardware monitoring thread starting.");

		WatchQat_Loop();  // device health/safety monitoring will begin in the new CPU thread

		WatchQat_CleanUp_All_Devices(true, false);  // clear call err counts, don't reset defaults
		LOG_F(INFO, "Hardware monitoring thread ending.");
	}

	~ThreadW(void)
	{ // Destructor
		// WATCHQAT SHUTDOWN HERE <----
	}
};


//
// WATCHQAT:  Hardware Health Thread
//#pragma managed
int SpawnWatchQatThread()
{
	ThreadW^ o1 = gcnew ThreadW(); // ThreadW for WatchQat, defined above, this file
	Thread^ t1 = gcnew Thread(gcnew ThreadStart(o1, &ThreadW::WatchQatThreadEntryPoint));
	t1->Name = "Hardware Monitor";
	t1->Start();

	// TODO: return 0 if OK, nonzero if error
	return 0;
}
