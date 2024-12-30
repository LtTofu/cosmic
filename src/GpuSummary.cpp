#pragma once

// used by CosmicWind.h, GpuSummary.h
constexpr auto CUDA_MAX_DEVICES = 19;	// keep consistent. (or include defs.h)
unsigned short gpusSummarized[CUDA_MAX_DEVICES] = { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 
													0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 
													0xFF, 0xFF, 0xFF };		// CUDA device indices (numbered from 0).	[old]
unsigned short gpuSummary_solverNo{ 0xFF };			// a single device