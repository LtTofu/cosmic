// hwmon.h : header for hw monitoring thread (watchqat)
// 2020 LtTofu
#pragma once

#define MAX_NVML_NVAPI_CONSECUTIVE_CALL_ERRS	2		// WatchQat: error counts to stop a call that fails repeatedly
#define DEF_WATCHQAT_THREAD_SLEEP_MS			175		// ms

// default values
#define DEFAULT_HWMON_MAXGPUTEMP		89		// Reasonable default
#define DEFAULT_HWMON_MINIMUMRPM		800		// Reasonable fan/pump minimum expectation (only used if bad config value was retrieved)
//#define CUDA_MAX_DEVICES				19		// consolidate w/ Defs <--

//#ifdef HWMON
#include "defs.hpp"

#include <nvml.h>
#include <nvapi.h>
//
int SpawnWatchQatThread();
short InitHardwareMonitoring();

void WatchQat_CleanUp_Device(const unsigned short deviceNo, const bool clearCallErrs, const bool resetToDefaults);
void WatchQat_CleanUp_All_Devices(const bool clearCallErrs, const bool resetToDefaults);
//
struct WQdevice
{ // sketching...
	bool					watchqat_enabled;			// will save CPU time if off

	// nvml stuff:
	nvmlDevice_t			nvml_handle;				// <-

	// nvapi stuff:
	NvPhysicalGpuHandle		nvapi_handle;				// NvAPI device handle
	NV_GPU_THERMAL_SETTINGS nvapi_thermals;				// for reading NvAPI temp sensors

	// most recent readings
	int						utilization;				// inited to -1
	unsigned int			fanspeed_rpm;				// actual rpm's (tachometer reading)
	unsigned short			fanspeed_percent;			// setting in %, not actual speed
	double					powerdraw_w;				//
	int						gputemp;					// GPU temperature in degrees C

	// safety features (true=ON, false=OFF)
	bool					health_maxgputemp_enable;
	bool					health_minrpm_enable;

	// thresholds
	unsigned int			health_minrpm;				// can't be <0 RPM. 0=no min
	int						health_maxgputemp;			// signed, can be <0C/32F

	// allows an error-prone call to be skipped:
	unsigned int			errcnt_gettemp;				// errcount: getting gpu temp
	unsigned int			errcnt_getpower;			// ...power (mw)
	unsigned int			errcnt_getfanrpm;			// ...fan tach in rpm
	unsigned int			errcnt_getfanpct;			// ...fan setting in %.
	unsigned int			errcnt_getutilization;		// ...CUDA utilization

	unsigned short			doevery;					// for timings
	// ...
};

extern struct WQdevice gWatchQat_Devices[CUDA_MAX_DEVICES];  // the element # = the CUDA device index#.
//#endif

extern bool gDeviceManuallyUnpaused[CUDA_MAX_DEVICES];
//extern int gCudaDevicePciBusIDs[CUDA_MAX_DEVICES];

