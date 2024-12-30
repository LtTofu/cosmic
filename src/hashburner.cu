#pragma once

// hashburner(.cu/.cuh)
// keccak256 solver for COSMiC V4.  2020 LtTofu
//
// Thx: Infernal Toast, Zegordo, Mike Seiler (Snissn), Azlehria,
//		Brian Bowden (Dunhili) et al?

//#include <iostream>
//#include <cstdint>  // [TESTME/TODO]: standardize headers in use
//#include <cinttypes>	// inttypes.h. Includes stdint/cstdint? <---
#include <inttypes.h>	// Includes stdint.h
#include <loguru/loguru.hpp>

//#include "types.hpp"
#include "hashburner.cuh"
// #include "cuda_device.hpp"

#include "defs.hpp"	//<-- for the below macros. Condensing...
#define TEST_BEHAVIOR
#define THREAD_SLEEPS_DURING_NETWORK_PAUSE

// THX: Azlehria
#if defined __INTELLISENSE__
/* reduce vstudio warnings (__byteperm, blockIdx...) */
#  define __CUDA_ARCH__ 1
#  include <device_launch_parameters.h>
#  undef __CUDA_ARCH__
#  define __CUDACC__ 1
#  include <device_atomic_functions.hpp>
#  undef __CUDACC__
//#  include <cstring>
#endif //__INTELLISENSE__

#include <cuda.h>
#include <cuda_runtime.h>
//#include <cuda_runtime_api.h>
#include <driver_types.h>
//#include <device_launch_parameters.h>
//#include <device_functions.h>


#ifdef THREAD_SLEEPS_DURING_NETWORK_PAUSE
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>			// for sleep() in host code. [TODO]: replace or move
#endif

__device__ uint64_t solns_data[MAX_SOLUTIONS];
__device__ uint32_t solns_count[1];				//<-- not really an array

// these are precomputed now (20190815: streamlining old 0xbitcoin-miner/gpuminer stuff)
dim3 gCuda_Grid[CUDA_MAX_DEVICES] = { 0 };	// set at mining start
dim3 gCuda_Block[CUDA_MAX_DEVICES] = { 0 };	// 


bool gSolving[CUDA_MAX_DEVICES] = { 0 };	 // if set to false, solving stops, CUDA solver cleanup/shutdown
bool gNetPause{ false };

// __device__ uint64_t*	h_message;
// __host__ uint64_t*	h_solutions_count;
// __device__ uint64_t	d_solutions[MAX_SOLUTIONS];
// __device__ uint8_t*	d_solutionCount;

// uint64_t cudaDeviceClockSpeed[CUDA_MAX_DEVICES] = {0};
// uint8_t cudaDeviceComputeCapability[CUDA_MAX_DEVICES] = {0};
// struct timeb tStart[CUDA_MAX_DEVICES], tEnd[CUDA_MAX_DEVICES];
uint64_t gCuda_HashCounts[CUDA_MAX_DEVICES] = { 0 };
uint64_t gNum_SolutionCount[CUDA_MAX_DEVICES] = { 0 };
uint64_t gNum_InvalidSolutionCount[CUDA_MAX_DEVICES] = { 0 };

uint64_t cnt[CUDA_MAX_DEVICES] = { 0 };  // For striding across search space
										 // Multiple devices now implemented. decide on independent or unified counter [todo].

// these should only be used if mid/target are NOT passed in as kernel parameters.
// should be "aggressively cached" on all the devices.
//__constant__ /*__align__(32)*/ uint64_t d_midstate[25];	// kernel input (200-byte/1600-bit state). Align this! [TODO]<----
//__constant__ /*__align__(32)*/ uint64_t d_target;			// device target. most-significant 64 bits

//=== ===

// round constants for keccak256:
__constant__ /* __align__(32) */ uint64_t rc[24] =
{	/* Element     (elements with 32 contiguous zero bits: 1, 4-5, 8, 9-12, 18, 22). */
	/* ------- */
	/* 00..02  */  0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
	/* 03..05  */  0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
	/* 06..08  */  0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
	/* 09..11  */  0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
	/* 12..14  */  0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
	/* 15..17  */  0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
	/* 18..20  */  0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
	/* 21..23  */  0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};

//#ifndef __INTELLISENSE__
__device__ __forceinline__
uint64_t bswap_64(uint64_t input, uint32_t scratch)
{ // uses PRMT instruction for permutation. w/ control chars. recommended by nV docs
	asm("{									\n\t"
		//		"  .reg .u32 scr;					\n\t"
		//		"  prmt.b32 scr, %0, 0, 0x0123;		\n\t"
		//		"  prmt.b32 %0, %1, 0, 0x0123;		\n\t"
		//		"  mov.b32 %1, scr;					\n\t"
		"	prmt.b32 %2, %1, 0, 0x0123;		\n\t"
		"	prmt.b32 %1, %0, 0, 0x0123;		\n\t"
		"	mov.b32 %0, %2;					\n\t"
		"}" : "+r"(reinterpret_cast<uint2&>(input).x), "+r"(reinterpret_cast<uint2&>(input).y), "+r"(scratch));
	// thx to Azlehria for idea to use reinterpret_cast to get at the 32-bit halves of the `input` param!
	return input;
}
//#endif

//
__device__ __forceinline__
uint64_t ROTL64funl_b(uint64_t input, uint32_t magnitude, uint32_t scratch)
{ // funnel shift
	asm("{									\n\t"
		//		".reg .b32 scr;						\n\t"
		"shf.l.wrap.b32 %2, %1, %0, %3;		\n\t"
		"shf.l.wrap.b32 %1, %0, %1, %3;		\n\t"
		"mov.b32 %0, %2;					\n\t"  //<--- replace? write to output pointer instead?
		"}" : "+r"(reinterpret_cast<uint2&>(input).x), "+r"(reinterpret_cast<uint2&>(input).y), "+r"(scratch) : "r"(magnitude));
	return input;
}

//
__device__ __forceinline__
uint64_t ROTR64funl_b(uint64_t input, uint32_t magnitude, uint32_t scratch)
{  // WIP: this approach appears noticeably faster (reduced register pressure?)
	asm("{									\n\t"
		//".reg .b32 scr;					\n\t"  // 32-bit scratch register
		"shf.r.wrap.b32 %2, %0, %1, %3;		\n\t"  // 32-bit funnel shift right with wrap
		"shf.r.wrap.b32 %1, %1, %0, %3;		\n\t"  // with flipped halfword input operands
		"mov.b32 %0, %2;					\n\t"  // copy half from scratch reg
		"}" : "+r"(reinterpret_cast<uint2&>(input).x), "+r"(reinterpret_cast<uint2&>(input).y), "+r"(scratch) : "r"(magnitude));
	return input;
}

// ROTL64funl (funnel shift LEFT)
__device__ __forceinline__
uint64_t ROTL64funl(uint64_t input, const uint32_t offs, uint64_t scratch)  // <-- use this as template
{  // rotate uint64 input Left by `offs` positions using pair of 32-bit SHF instructions (funnel shift).
	asm("{									\n\t"  // testing this approach to reduce register pressure:
		"shf.l.wrap.b32 %0, %3, %2, %4;		\n\t"  // funnel shift (larger input than output), with wrap
		"shf.l.wrap.b32 %1, %2, %3, %4;		\n\t"  // 
		"}" : "=r"(reinterpret_cast<uint2&>(scratch).x), "=r"(reinterpret_cast<uint2&>(scratch).y) :
		"r"(reinterpret_cast<uint2&>(input).x), "r"(reinterpret_cast<uint2&>(input).y), "r"(offs));
	return scratch; //<---
}

// ROTR64funl (funnel shift RIGHT)
__device__ __forceinline__
uint64_t ROTR64funl(uint64_t input, uint32_t magnitude, uint64_t scratch)
{  // rotate uint64 input Right by `offs` positions using pair of 32-bit SHF instructions (funnel shift)
	asm("{									\n\t"
		"shf.r.wrap.b32 %0, %2, %3, %4;		\n\t"  // 32-bit funnel shift right with wrap
		"shf.r.wrap.b32 %1, %3, %2, %4;		\n\t"  // with flipped halfword input operands
		"}" : "=r"(reinterpret_cast<uint2&>(scratch).x), "=r"(reinterpret_cast<uint2&>(scratch).y) :
		"r"(reinterpret_cast<uint2&>(input).x), "r"(reinterpret_cast<uint2&>(input).y), "r"(magnitude));
	return scratch;
}

//
__device__ __forceinline__
uint64_t xor5_lop3(uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e)
{
	asm("{										\n\t"
		//		"  .reg .b32 scr;"						\n\t	// uint64 `a` is used as a scratch var instead
		"  lop3.b32 %0, %0, %2, %4, 0x96;		\n\t"	// 0x96 = XOR-XOR-XOR (immLut)
		"  lop3.b32 %0, %0, %6, %8, 0x96;		\n\t"	//
		"  lop3.b32 %1, %1, %3, %5, 0x96;		\n\t"	//
		"  lop3.b32 %1, %1, %7, %9, 0x96;		\n\t"	//
		"}" : "+r"(reinterpret_cast<uint2&>(a).x), "+r"(reinterpret_cast<uint2&>(a).y) :
		"r"(reinterpret_cast<uint2&>(b).x), "r"(reinterpret_cast<uint2&>(b).y),
		"r"(reinterpret_cast<uint2&>(c).x), "r"(reinterpret_cast<uint2&>(c).y),
		"r"(reinterpret_cast<uint2&>(d).x), "r"(reinterpret_cast<uint2&>(d).y),
		"r"(reinterpret_cast<uint2&>(e).x), "r"(reinterpret_cast<uint2&>(e).y));
	return a;
}

//
__device__ __forceinline__
uint64_t xor5(uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e)
{
	asm("{"
		//"	 .reg .b64 scr;			"  // scratch register.
		"	 xor.b64 %0, %0, %1;			\n\t"
		"	 xor.b64 %0, %0, %2;			\n\t"
		"	 xor.b64 %0, %0, %3;			\n\t"
		"	 xor.b64 %0, %0, %4;			\n\t"
		"}" : "+l"(a) : "l"(b), "l"(c), "l"(d), "l"(e));  // `a` is read from _and_ written to, so "+l"
	return a;
}

// XOR5 with 32-bit instructions/operands. Outputs a^b^c^d^e.
__device__ __forceinline__
uint64_t test_xor5_32bit(uint64_t a, const uint64_t b, const uint64_t c, const uint64_t d, const uint64_t e)
{
	asm("{								\n\t"
		"	 xor.b32 %0, %0, %2;		\n\t"
		"	 xor.b32 %0, %0, %4;		\n\t"
		"	 xor.b32 %0, %0, %6;		\n\t"
		"	 xor.b32 %0, %0, %8;		\n\t"
		"	 xor.b32 %1, %1, %3;		\n\t"
		"	 xor.b32 %1, %1, %5;		\n\t"
		"	 xor.b32 %1, %1, %7;		\n\t"
		"	 xor.b32 %1, %1, %9;		\n\t"
		"}" : "+r"(reinterpret_cast<uint2&>(a).x), "+r"(reinterpret_cast<uint2&>(a).y) :		  /*  written to  */
		"r"(reinterpret_cast<const uint2&>(b).x), "r"(reinterpret_cast<const uint2&>(b).y), /*  read only:  */
		"r"(reinterpret_cast<const uint2&>(c).x), "r"(reinterpret_cast<const uint2&>(c).y),
		"r"(reinterpret_cast<const uint2&>(d).x), "r"(reinterpret_cast<const uint2&>(d).y),
		"r"(reinterpret_cast<const uint2&>(e).x), "r"(reinterpret_cast<const uint2&>(e).y));
	return a;
}

//
__device__ __forceinline__
void test_xor3_32bit(uint64_t* a, const uint64_t b, const uint64_t c)
{  // XOR3 with 32-bit instructions/operands. Outputs a^b^c
	asm("{									\n\t"
		"	 xor.b32 %0, %0, %2;			\n\t"
		"	 xor.b32 %0, %0, %4;			\n\t"
		"	 xor.b32 %1, %1, %3;			\n\t"
		"	 xor.b32 %1, %1, %5;			\n\t"
		"}" : "+r"(reinterpret_cast<uint2&>(*a).x), "+r"(reinterpret_cast<uint2&>(*a).y) :
		"r"(reinterpret_cast<const uint2&>(b).x), "r"(reinterpret_cast<const uint2&>(b).y),
		"r"(reinterpret_cast<const uint2&>(c).x), "r"(reinterpret_cast<const uint2&>(c).y));
	return;
}

//
__device__ __forceinline__
uint64_t xor3_lop3(uint64_t a, uint64_t b, uint64_t c)
{
	// TODO: retry lop3 implementation at some point, see if it is any faster (same for xor5)
	asm("{										\n\t"
		"  lop3.b32 %0, %0, %2, %4, 0x96;		\n\t"  // 0x96 = XOR-XOR-XOR (immLut)
		"  lop3.b32 %1, %1, %3, %5, 0x96;		\n\t"  // 
		"}" : "+r"(reinterpret_cast<uint2&>(a).x), "+r"(reinterpret_cast<uint2&>(a).y) :
		"r"(reinterpret_cast<uint2&>(b).x), "r"(reinterpret_cast<uint2&>(b).y),
		"r"(reinterpret_cast<uint2&>(c).x), "r"(reinterpret_cast<uint2&>(c).y));
	return a;
}

// XOR3: this handy function still used by Compatibility kernel
__device__ __forceinline__
uint64_t xor3(uint64_t a, const uint64_t b, const uint64_t c)
{
	asm("{										\n\t"
		"  xor.b64 %0, %0, %1;					\n\t"
		"  xor.b64 %0, %0, %2;					\n\t"
		"}" : "+l"(a) : "l"(b), "l"(c));
	return a;
}

__device__ __forceinline__
uint64_t ROTLfrom32(uint64_t input, uint32_t offs, uint64_t scratch)
{  // shortcut to rotation by 32 (transpose halves), rotate left by amount `offs`
	asm("{											\n\t"
		"    shf.l.wrap.b32 %0, %2, %3, %4;			\n\t"
		"    shf.l.wrap.b32 %1, %3, %2, %4;			\n\t"
		//		"    mov.b32 %0, %2;						\n\t"
		"}" : "=r"(reinterpret_cast<uint2&>(scratch).x), "=r"(reinterpret_cast<uint2&>(scratch).y) :
		"r"(reinterpret_cast<uint2&>(input).x), "r"(reinterpret_cast<uint2&>(input).y), "r"(offs));
	return scratch;  // was input;
}

//
__device__ __forceinline__
uint64_t ROTRfrom32(const uint64_t input, const uint32_t offs, uint64_t scratch) //<-- scratch needs to be 64-bit
{  // shortcut to rotation by 32 (transpose halves), rotate right by amount `offs`
	asm("{											\n\t"
		"    shf.r.wrap.b32 %0, %3, %2, %4;			\n\t"  // funnel shift (SHF, 32-bit), result to scratch.x
		"    shf.r.wrap.b32 %1, %2, %3, %4;			\n\t"  // ...and to scratch.y.
//		"    mov.b32 %0, %2;						\n\t"  // 20191231: eliminated this MOV
"}" : "=r"(reinterpret_cast<uint2&>(scratch).x), "=r"(reinterpret_cast<uint2&>(scratch).y) :
	"r"(reinterpret_cast<const uint2&>(input).x), "r"(reinterpret_cast<const uint2&>(input).y), "r"(offs));
	return scratch;  // was input;
}

// _alt functions for experimentation, only used in "performance" kernel
__device__ __forceinline__
void ROTRfrom32_alt(const uint64_t input, uint64_t* output, const uint32_t offs)
{  // "shortcut" to the result of a rotation by 32 positions (transpose halves), rotate right by amount `offs` after transposing
   // the 32-bit halves. Rotation using 2 32-bit SHF (funnel shift)s, equiv. output to rotation (right) of uint64 `input` by
   // offset of 32+offs (replaces old code w/ rotations of >32 positions)
	asm("{"
		//"	 .reg .b32 scr;						\n\t"	// 32 bit reg
		"    shf.r.wrap.b32 %0, %3, %2, %4;		\n\t"	// 32-bit funnel shift, result to output's .x (was: result to reg `scr`)
		"    shf.r.wrap.b32 %1, %2, %3, %4;		\n\t"	// same, result to `output` param's y.
		//"    mov.b32 %0, scr;					\n\t"	// mov 32 bits from reg `scr` to `output` param's x
		"}											" :
	"+r"(reinterpret_cast<uint2&>(output).x), "+r"(reinterpret_cast<uint2&>(output).y) :  /* =r because only written to, not read from? */
		"r"(reinterpret_cast<const uint2&>(input).x), "r"(reinterpret_cast<const uint2&>(input).y), "r"(offs));  /*  only read from */
}

// 
// PERMUTATION HELPERS (for rotation by fixed amounts)
// 
__device__ __forceinline__
void ROTLby16_b(uint64_t input, uint64_t* output)		// TODO: ROTRby16 PTX Function
{
	asm("{										\n\t"
		"   prmt.b32 %1, %2, %3, 0x5432;		\n\t"
		"   prmt.b32 %0, %2, %3, 0x1076;		\n\t"
		" }" : "=r"(reinterpret_cast<uint2&>(*output).x), "=r"(reinterpret_cast<uint2&>(*output).y) :	 /* writable  */
		"r"(reinterpret_cast<uint2&>(input).x), "r"(reinterpret_cast<uint2&>(input).y));				 /* only read */
}

// ROTLby8b: Use PRMT instructions to assemble output uint64 with required roation.
//			 Instead of returning value, it's written directly to output address.
__device__ __forceinline__
void ROTLby8_b(uint64_t input, uint64_t* output)
{
	asm("{									\n\t"
		"prmt.b32 %0, %2, %3, 0x2107;		\n\t"
		"prmt.b32 %1, %2, %3, 0x6543;		\n\t"
		"}": "=r"(reinterpret_cast<uint2&>(*output).x), "=r"(reinterpret_cast<uint2&>(*output).y) :  /* writable  */
		"r"(reinterpret_cast<uint2&>(input).x), "r"(reinterpret_cast<uint2&>(input).y));		 /* only read */
}

// ROTLby8: Use PRMT instructions to assemble output uint64 with required rotation.
//			Regular-return version with one function parameter.
__device__ __forceinline__
uint64_t ROTLby8(const uint64_t input)
{
	uint64_t scratch;	//intentionally uninitialized.
	asm("{											\n\t"
		"		prmt.b32 %0, %2, %3, 0x2107;		\n\t"
		"		prmt.b32 %1, %2, %3, 0x6543;		\n\t"
		"}" :
	"=r"(reinterpret_cast<uint2&>(scratch).x), "=r"(reinterpret_cast<uint2&>(scratch).y) :		  /* written to */
		"r"(reinterpret_cast<const uint2&>(input).x), "r"(reinterpret_cast<const uint2&>(input).y)); /* only read  */
	return scratch;
}

// ROTRby8b: Use PRMT instructions to assemble output uint64 with required rotation.
//			 Instead of returning value, it's written directly to output address.
__device__ __forceinline__
void ROTRby8_b(const uint64_t input, uint64_t* output)
{
	asm("{											\n\t"
		"prmt.b32 %0, %3, %2, 0x0765;				\n\t"
		"prmt.b32 %1, %3, %2, 0x4321;				\n\t"
		"}" :
	"=r"(reinterpret_cast<uint2&>(*output).x), "=r"(reinterpret_cast<uint2&>(*output).y) :		   /*  writable   */
		"r"(reinterpret_cast<const uint2&>(input).x), "r"(reinterpret_cast<const uint2&>(input).y));  /*  only read  */
}

// ROTRby8: Use PRMT instructions to assemble output uint64 with required rotation.
//				   Regular-return version with one function parameter.
__device__ __forceinline__
uint64_t ROTRby8(uint64_t input)
{
	uint64_t scratch;
	asm("{"
		"prmt.b32 %0, %3, %2, 0x0765;	\n\t"
		"prmt.b32 %1, %3, %2, 0x4321;	\n\t"
		"}" :  "=r"(reinterpret_cast<uint2&>(scratch).x), "=r"(reinterpret_cast<uint2&>(scratch).y) :
		"r"(reinterpret_cast<uint2&>(input).x), "r"(reinterpret_cast<uint2&>(input).y));
	return scratch;
}

__device__ __forceinline__
void ROTRby24(uint64_t input, uint64_t* output)
{  // Alternate (in testing) version w/ void return-type, output to a pointer-to-uint64 arg. and intrinsic functions in lieu of PTX.
   // TODO: Speed comparison versus PTX implementation. Speed is bound to vary arch-to-arch vs. shift/xor-based rotation or funnel shift.
	reinterpret_cast<uint2&>(*output).x = __byte_perm(reinterpret_cast<uint2&>(input).y, reinterpret_cast<uint2&>(input).x, 0x2107);  // permutation
	reinterpret_cast<uint2&>(*output).y = __byte_perm(reinterpret_cast<uint2&>(input).y, reinterpret_cast<uint2&>(input).x, 0x6543);  // other half
}

// FIXME/TODO: make SURE that the correct version is running on Maxwell Gen2, Pascal!
__device__ __forceinline__
uint64_t chi_lop3(uint64_t a, uint64_t b, uint64_t c)
{   // chi_lop3(): keccak "Chi" using nVidia LOP3.LUT instruction to merge 3 logical operations into one.
	//	note: because lop3 is a 32-bit instruction we do two to process the 64-bit uint inputs
#if __CUDA_ARCH__ >= 500
	asm("{										\n\t"
		"  lop3.b32 %0, %0, %2, %4, 0xD2;		\n\t"
		"  lop3.b32 %1, %1, %3, %5, 0xD2;		\n\t"
		"}" : "+r"(reinterpret_cast<uint2&>(a).x), "+r"(reinterpret_cast<uint2&>(a).y) :	 /* `a` is read & written	  */
		"r"(reinterpret_cast<uint2&>(b).x), "r"(reinterpret_cast<uint2&>(b).y),			 /* `b` and `c` are only read */
		"r"(reinterpret_cast<uint2&>(c).x), "r"(reinterpret_cast<uint2&>(c).y));
	return a;																												 // note: +l is .u64, "r" is .u32
#else
	return a ^ ((~b) & c);
#endif
}

__device__ __forceinline__
void chi_lop3_alt(const uint64_t a, const uint64_t b, const uint64_t c, uint64_t* output)
{
	//#if __CUDA_ARCH__ >= 500
	asm("{										 \n\t"
		"  lop3.b32 %0, %2, %4, %6, 0xD2;		 \n\t"
		"  lop3.b32 %1, %3, %5, %7, 0xD2;		 \n\t"
		"}": "=r"(reinterpret_cast<uint2&>(*output).x), "+r"(reinterpret_cast<uint2&>(*output).y) :	 /* output: written only	   */
		"r"(reinterpret_cast<const uint2&>(a).x), "r"(reinterpret_cast<const uint2&>(a).y),	 /* `a` is read & written	   */
		"r"(reinterpret_cast<const uint2&>(b).x), "r"(reinterpret_cast<const uint2&>(b).y),	 /* `b` and `c` are only read. */
		"r"(reinterpret_cast<const uint2&>(c).x), "r"(reinterpret_cast<const uint2&>(c).y));
	// note: +l is .u64, "r" is .u32
	return;
	//#else ... return a ^ ((~b) & c); ... #endif (old)
}

__device__ __forceinline__
void ROTLby1_64(uint64_t input, uint64_t* output)
{ // rotates 64-bit uint `input` by 1 (using SHF, funnel shift instruction)
	asm("{									\n\t"
		"shf.l.wrap.b32 %0, %3, %2, 1;		\n\t"  // funnel shift Left with Wrap by immediate `1`
		"shf.l.wrap.b32 %1, %2, %3, 1;		\n\t"  // 32-bit .x and .y inputs reversed
		"}" : "=r"(reinterpret_cast<uint2&>(*output).x), "=r"(reinterpret_cast<uint2&>(*output).y) :
		"r"(reinterpret_cast<uint2&>(input).x), "r"(reinterpret_cast<uint2&>(input).y));
}

__device__ __forceinline__
uint64_t chi_compat(const uint64_t a, const uint64_t b, const uint64_t c)
{
	return a ^ ((~b) & c);
}

//new!
__device__ __forceinline__
void ROTL64funl_c(uint64_t input, uint64_t* output, uint32_t offs)
{  // testing this approach to reduce register pressure
	asm("{									\n\t"
		"shf.l.wrap.b32 %0, %3, %2, %4;		\n\t"
		"shf.l.wrap.b32 %1, %2, %3, %4;		\n\t"
		"}" : "=r"(reinterpret_cast<uint2&>(*output).x), "=r"(reinterpret_cast<uint2&>(*output).y) :
		"r"(reinterpret_cast<uint2&>(input).x), "r"(reinterpret_cast<uint2&>(input).y), "r"(offs));
}

//new!
__device__ __forceinline__
void ROTR64funl_c(uint64_t input, uint64_t* output, uint32_t offs)
{  // testing this approach to reduce register pressure
	asm("{									\n\t"
		"shf.r.wrap.b32 %0, %2, %3, %4;		\n\t"
		"shf.r.wrap.b32 %1, %3, %2, %4;		\n\t"
		"}" : "=r"(reinterpret_cast<uint2&>(*output).x), "=r"(reinterpret_cast<uint2&>(*output).y) :
		"r"(reinterpret_cast<uint2&>(input).x), "r"(reinterpret_cast<uint2&>(input).y), "r"(offs));
}

__device__ __forceinline__
void ApplyRoundConstant(uint64_t* input, const uint8_t round_num)
{ // XOR `input` with round constant for `round_num`. Appears slightly faster (at least on Pascal arch.)
  // RC elements where input.x is all leading zeroes: 1, 4-5, 8-12, 18, 22.
  //REF: (round_num==0)  input ^= 0x0000000000000001;
	if (round_num == 1)  reinterpret_cast<uint2&>(*input).x ^= 0x00008082;		// XOR 32 bits of input w/ a 32-bit constant -or-
	else if (round_num == 2)  *input ^= 0x800000000000808a;						// XOR 64-bit input w/ a 64-bit constant
	else if (round_num == 3)  *input ^= 0x8000000080008000;
	else if (round_num == 4)  reinterpret_cast<uint2&>(*input).x ^= 0x0000808b;
	else if (round_num == 5)  reinterpret_cast<uint2&>(*input).x ^= 0x80000001;
	else if (round_num == 6)  *input ^= 0x8000000080008081;
	else if (round_num == 7)  *input ^= 0x8000000000008009;
	else if (round_num == 8)  reinterpret_cast<uint2&>(*input).x ^= 0x0000008a;  //
	else if (round_num == 9)  reinterpret_cast<uint2&>(*input).x ^= 0x00000088;  // 
	else if (round_num == 10) reinterpret_cast<uint2&>(*input).x ^= 0x80008009;  // 
	else if (round_num == 11) reinterpret_cast<uint2&>(*input).x ^= 0x8000000a;  // 
	else if (round_num == 12) reinterpret_cast<uint2&>(*input).x ^= 0x8000808b;  // 
	else if (round_num == 13) *input ^= 0x800000000000008b;
	else if (round_num == 14) *input ^= 0x8000000000008089;
	else if (round_num == 15) *input ^= 0x8000000000008003;
	else if (round_num == 16) *input ^= 0x8000000000008002;
	else if (round_num == 17) *input ^= 0x8000000000000080;
	else if (round_num == 18) reinterpret_cast<uint2&>(*input).x ^= 0x0000800a;  // TODO: first half is 0s, do '32 bit' XOR
	else if (round_num == 19) *input ^= 0x800000008000000a;
	else if (round_num == 20) *input ^= 0x8000000080008081;  // TODO: Note RC:20 and RC:6 are the same.
	else if (round_num == 21) *input ^= 0x8000000000008080;  // TODO: Note RC:20/21 differ by just one bit.
	else if (round_num == 22) reinterpret_cast<uint2&>(*input).x ^= 0x80000001;  // TODO: first half is 0s, do '32 bit' XOR
}


//
// === "HashBurner" (performance hashing function) ===
__device__
uint8_t keccak(const uint8_t deviceID, const uint64_t nonce, const uint64_t midstate[25], const uint64_t target)
{
	/*__align__(32)*/ uint64_t state[25];	// intentionally uninitialized. written to before read from.
	/*__align__(32)*/ uint64_t C[5];		// , D[5];
	//uint8_t i, x;

#if __CUDA_ARCH__ >= 700
	C[0] = midstate[2] ^ ROTR64funl(nonce, 20, 0);
	C[1] = midstate[4] ^ ROTL64funl(nonce, 14, 0);
#else
	C[0] = midstate[2] ^ ROTR64funl_b(nonce, 20, 0);  //0=scratch
	C[1] = midstate[4] ^ ROTL64funl_b(nonce, 14, 0);  //
#endif

	state[0] = chi_lop3(midstate[0], midstate[1], C[0]);
	//state[0] = state[0] ^ 0x0000000000000001;
	asm("xor.b32 %0, %0, 0x00000001;" : "+r"(reinterpret_cast<uint2&>(state[0]).x));	//XOR relevant 32 bits against Round 0 constant. the remaining 32 bits are zero.
	
	state[1] = chi_lop3(midstate[1], C[0], midstate[3]);
	state[2] = chi_lop3(C[0], midstate[3], C[1]);
	state[3] = chi_lop3(midstate[3], C[1], midstate[0]);
	state[4] = chi_lop3(C[1], midstate[0], midstate[1]);

#if __CUDA_ARCH__ >= 700
	C[0] = midstate[6] ^ ROTL64(nonce, 20);
#else
	C[0] = midstate[6] ^ ROTL64funl_b(nonce, 20, 0);  //0=scratch
#endif

	C[1] = midstate[9] ^ ROTR64(nonce, 2);
	state[5] = chi_lop3(midstate[5], C[0], midstate[7]);
	state[6] = chi_lop3(C[0], midstate[7], midstate[8]);
	state[7] = chi_lop3(midstate[7], midstate[8], C[1]);
	state[8] = chi_lop3(midstate[8], C[1], midstate[5]);
	state[9] = chi_lop3(C[1], midstate[5], C[0]);

#if __CUDA_ARCH__ < 600
	C[0] = midstate[11] ^ ROTL64(nonce, 7);
	ROTLby8_b(nonce, &C[1]);
	C[1] ^= midstate[13];
#else
	C[0] = midstate[11] ^ ROTL64(nonce, 7);
	C[1] = midstate[13] ^ ROTL64(nonce, 8);
#endif

	state[10] = chi_lop3(midstate[10], C[0], midstate[12]);
	state[11] = chi_lop3(C[0], midstate[12], C[1]);
	state[12] = chi_lop3(midstate[12], C[1], midstate[14]);
	state[13] = chi_lop3(C[1], midstate[14], midstate[10]);
	state[14] = chi_lop3(midstate[14], midstate[10], C[0]);
	//
	C[0] = midstate[15] ^ ROTRfrom32(nonce, 5, 0);		// ref: C[0] = midstate[15] ^ ROTL64(nonce, 27);
	//C[0] = midstate[15] ^ ROTL64funl(nonce, 27);		// alt
	//C[0] = midstate[15] ^ ROTLfrom32(nonce, 5, 0);	//
#if __CUDA_ARCH__ >= 700
	C[1] = ROTL64(nonce, 16) ^ midstate[18];
#elif __CUDA_ARCH__ < 600
	ROTLby16_b(nonce, &C[1]);	//C[1] = midstate[18] ^ ROTLby16(nonce);
	C[1] ^= midstate[18];
#else
	ROTLby16_b(nonce, &C[1]);	//C[1] = midstate[18] ^ ROTL64(nonce, 16);
	C[1] ^= midstate[18];
#endif
	//
	state[15] = chi_lop3(C[0], midstate[16], midstate[17]);
	state[16] = chi_lop3(midstate[16], midstate[17], C[1]);
	state[17] = chi_lop3(midstate[17], C[1], midstate[19]);
	state[18] = chi_lop3(C[1], midstate[19], C[0]);
	state[19] = chi_lop3(midstate[19], C[0], midstate[16]);
	//
	C[0] = midstate[20] ^ ROTR64(nonce, 1);	// <-- try ROTRby1_64()
	C[1] = midstate[21] ^ ROTR64(nonce, 9);	//		...and ROTRby8().
#if __CUDA_ARCH__ < 600
	C[2] = midstate[22] ^ ROTLfrom32(nonce, 7, 0);
#else
	C[2] = midstate[22] ^ ROTR64funl(nonce, 25, 0);
#endif
	//
	state[20] = chi_lop3(C[0], C[1], midstate[22] ^ ROTLfrom32(nonce, 7, 0)); // was C[2] 
	state[21] = chi_lop3(C[1], midstate[22] ^ ROTLfrom32(nonce, 7, 0), midstate[23]);  // was C[2]
	state[22] = chi_lop3(midstate[22] ^ ROTLfrom32(nonce, 7, 0), midstate[23], midstate[24]);  // was C[2]
	state[23] = chi_lop3(midstate[23], midstate[24], C[0]);
	state[24] = chi_lop3(midstate[24], C[0], C[1]);
	// --

	// 20181224: unrolled loops w/ precomputed modulus
	// /* __align__(32) */ uint64_t D[5];
	// uint64_t C3, C4;
#pragma unroll 23
	for (uint8_t i = 1; i < 23; ++i)
	{
		// Theta
		// for (x = 0; x < 5; ++x)
		//	   C[(x + 6) % 5] = xor5(state[x], state[x + 5], state[x + 10], state[x + 15], state[x + 20], 0   );
#if __CUDA_ARCH__ >= 600 && __CUDA_ARCH__ < 700
		C[0] = test_xor5_32bit(state[4], state[9], state[14], state[19], state[24]);
		C[1] = test_xor5_32bit(state[0], state[5], state[10], state[15], state[20]);
		C[2] = test_xor5_32bit(state[1], state[6], state[11], state[16], state[21]);
		C[3] = test_xor5_32bit(state[2], state[7], state[12], state[17], state[22]);
		C[4] = test_xor5_32bit(state[3], state[8], state[13], state[18], state[23]);
#else  // on Maxwell Gen2 and back (e.g. GTX9xx), use LOP3-accelerated XOR5 sub.
	   // 20190604: Regular compiler-optimized XOR for Volta+Turing and up
		C[0] = test_xor5_32bit(state[4], state[9], state[14], state[19], state[24]);
		C[1] = test_xor5_32bit(state[0], state[5], state[10], state[15], state[20]);
		C[2] = test_xor5_32bit(state[1], state[6], state[11], state[16], state[21]);
		C[3] = test_xor5_32bit(state[2], state[7], state[12], state[17], state[22]);
		C[4] = test_xor5_32bit(state[3], state[8], state[13], state[18], state[23]);
#endif
		state[0] = xor3_lop3(state[0], ROTL64(C[2], 1), C[0]);
		state[1] = xor3_lop3(state[1], ROTL64(C[3], 1), C[1]);
		state[2] = xor3_lop3(state[2], ROTL64(C[4], 1), C[2]);
		state[3] = xor3_lop3(state[3], ROTL64(C[0], 1), C[3]);
		state[4] = xor3_lop3(state[4], ROTL64(C[1], 1), C[4]);
		state[5] = xor3_lop3(state[5], ROTL64(C[2], 1), C[0]);
		state[6] = xor3_lop3(state[6], ROTL64(C[3], 1), C[1]);
		state[7] = xor3_lop3(state[7], ROTL64(C[4], 1), C[2]);
		state[8] = xor3_lop3(state[8], ROTL64(C[0], 1), C[3]);
		state[9] = xor3_lop3(state[9], ROTL64(C[1], 1), C[4]);
		state[10] = xor3_lop3(state[10], ROTL64(C[2], 1), C[0]);
		state[11] = xor3_lop3(state[11], ROTL64(C[3], 1), C[1]);
		state[12] = xor3_lop3(state[12], ROTL64(C[4], 1), C[2]);
		state[13] = xor3_lop3(state[13], ROTL64(C[0], 1), C[3]);
		state[14] = xor3_lop3(state[14], ROTL64(C[1], 1), C[4]);
		state[15] = xor3_lop3(state[15], ROTL64(C[2], 1), C[0]);
		state[16] = xor3_lop3(state[16], ROTL64(C[3], 1), C[1]);
		state[17] = xor3_lop3(state[17], ROTL64(C[4], 1), C[2]);
		state[18] = xor3_lop3(state[18], ROTL64(C[0], 1), C[3]);
		state[19] = xor3_lop3(state[19], ROTL64(C[1], 1), C[4]);
		state[20] = xor3_lop3(state[20], ROTL64(C[2], 1), C[0]);
		state[21] = xor3_lop3(state[21], ROTL64(C[3], 1), C[1]);
		state[22] = xor3_lop3(state[22], ROTL64(C[4], 1), C[2]);
		state[23] = xor3_lop3(state[23], ROTL64(C[0], 1), C[3]);
		state[24] = xor3_lop3(state[24], ROTL64(C[1], 1), C[4]);

		// Rho Pi
		C[0] = state[1];
#if __CUDA_ARCH__ < 600
		state[1] = ROTR64funl_b(state[6], 20, 0);
#else
		state[1] = ROTR64(state[6], 20);
#endif
		state[6] = ROTL64funl_b(state[9], 20, 0);  // <-
		state[9] = ROTR64(state[22], 3);  // <-
#if __CUDA_ARCH__ >= 700
		state[22] = ROTR64(state[14], 25);  // 20190604: for improved performance on Turing
#elif __CUDA_ARCH__ < 600
		state[22] = ROTLfrom32(state[14], 7, 0);  // 0 dummy val for scratch param
#else
		state[22] = ROTR64funl_b(state[14], 25, 0);
#endif
		// note to self: this is faster on Maxwell, slower on Pascal:		state[14] = ROTL64funl(ROTLby16(state[20]), 2);
		state[14] = ROTL64funl_b(state[20], 18, 0);						//	state[14] = ROTL64(state[20], 18);
		state[20] = ROTR64(state[2], 2);
		//#if __CUDA_ARCH__ >= 600
		//		state[2] = ROTR64(state[12], 21);
		//#else
		//		state[2] = ROTR64funl_b(state[12], 21, 0);
#if __CUDA_ARCH__ >= 700
		state[2] = ROTR64funl(state[12], 21, 0);
#else
		state[2] = ROTLfrom32(state[12], 11, 0);
#endif

#if __CUDA_ARCH__ >= 600
		state[12] = ROTRfrom32(state[13], 7, 0);  // state[12] = ROTL64(state[13], 25);
#else
		state[12] = ROTL64funl(state[13], 25, 0);
		// (todo) try ROTL64(ROTLby24(state[13]), 1)?  or  state[12] = ROTRfrom32(state[13], 7, 0)
#endif	
// removed #if __CUDA_ARCH__ >= 600 ...  (20191227)
		ROTLby8_b(state[19], &state[13]);  //state[13] = ROTL64(state[19], 8);
		ROTRby8_b(state[23], &state[19]);  //state[19] = ROTR64(state[23], 8);
//
#if __CUDA_ARCH__ >= 600
		state[23] = ROTR64funl(state[15], 23, 0);  // 0=scratch
#else
		ROTRby24(state[15], &state[23]);  // ROTR64(state[15], 23)
		state[23] = ROTL64(state[23], 1);
#endif

		//#if __CUDA_ARCH__ >= 600 ...
		state[15] = ROTRfrom32(state[4], 5, 0);  // state[15] = ROTL64(state[4], 27);

#if __CUDA_ARCH__ < 600
		state[4] = ROTL64funl_b(state[24], 14, 0);
#else
		state[4] = ROTL64(state[24], 14);
#endif
		state[24] = ROTL64(state[21], 2);
		state[21] = ROTR64(state[8], 9);
#if __CUDA_ARCH__ < 600
		state[8] = ROTR64funl_b(state[16], 19, 0);
#else
		state[8] = ROTR64(state[16], 19);
#endif
		state[16] = ROTLfrom32(state[5], 4, 0);
		state[5] = ROTRfrom32(state[3], 4, 0);
#if __CUDA_ARCH__ >= 600
		state[3] = ROTL64funl_b(state[18], 21, 0);
#else
		state[3] = ROTL64funl(state[18], 21, 0);
#endif
		state[18] = ROTL64(state[17], 15);
		state[17] = ROTL64(state[11], 10);
		state[11] = ROTL64(state[7], 6);
		state[7] = ROTL64(state[10], 3);
		state[10] = ROTL64(C[0], 1);

		uint8_t x;  // TODO: replace w/ uint32_t ?
#pragma unroll 5
		for (x = 0; x < 25; x += 5)
		{  // or:  memcpy(&C[0], &state[x], 40);
			C[0] = state[x];
			C[1] = state[x + 1];
			C[2] = state[x + 2];
			C[3] = state[x + 3];
			C[4] = state[x + 4];
			//
			state[x] = chi_lop3(C[0], C[1], C[2]);
			state[x + 1] = chi_lop3(C[1], C[2], C[3]);
			state[x + 2] = chi_lop3(C[2], C[3], C[4]);
			state[x + 3] = chi_lop3(C[3], C[4], C[0]);
			state[x + 4] = chi_lop3(C[4], C[0], C[1]);
		}

		// Iota
		ApplyRoundConstant(&state[0], i);	// appears slightly faster
		//state[0] = state[0] ^ rclocal[i]; // XOR with round `i` constant
	}

	//[ref]:	for (/*int32_t or uint32_t*/ x = 0; x < 5; ++x)
	//				C[(x + 6) % 5] = xor5(state[x], state[x + 5], state[x + 10], state[x + 15], state[x + 20], 0 );
	// unrolled:
#if __CUDA_ARCH__ >= 600 && __CUDA_ARCH__ < 700  // Pascal
	C[1] = xor5_lop3(state[0], state[5], state[10], state[15], state[20]);
	C[2] = xor5_lop3(state[1], state[6], state[11], state[16], state[21]);
	C[3] = xor5_lop3(state[2], state[7], state[12], state[17], state[22]);
	C[4] = xor5_lop3(state[3], state[8], state[13], state[18], state[23]);
	C[0] = xor5_lop3(state[4], state[9], state[14], state[19], state[24]);
#else  // Volta, Turing, Maxwell... etc.
	C[1] = test_xor5_32bit(state[0], state[5], state[10], state[15], state[20]);  // <-- new!
	C[2] = test_xor5_32bit(state[1], state[6], state[11], state[16], state[21]);  //
	C[3] = test_xor5_32bit(state[2], state[7], state[12], state[17], state[22]);  //
	C[4] = test_xor5_32bit(state[3], state[8], state[13], state[18], state[23]);  //
	C[0] = test_xor5_32bit(state[4], state[9], state[14], state[19], state[24]);  //
#endif

#define d0 ROTL64(C[2], 1)  // recompute these as needed instead of more load/store
#define d1 ROTL64(C[3], 1)
#define d2 ROTL64(C[4], 1)

	// extra-condensed final check against 64-bit difficulty target
	return (bswap_64((chi_lop3(xor3_lop3(state[0], d0, C[0]),
		ROTR64funl(xor3_lop3(state[6], d1, C[1]), 20, 0),
		ROTR64funl(xor3_lop3(state[12], d2, C[2]), 21, 0)
	) ^ 0x8000000080008008), 0) <= target);  // round 23 constant
}



//
// === Compatibility hashing function ===
__device__
ushort keccak_compat(const uint8_t deviceID, const uint64_t nonce, const uint64_t midstate[25], const uint64_t target)		//<---- regular keccak() too!
{
	uint64_t state[25]{ 0 }, C[5]{ 0 }, D[5]{ 0 };
	//uint2* stateVec = (uint2*)&state;
	unsigned short i{ 0 }, x{ 0 };

	C[0] = midstate[2] ^ ROTR64(nonce, 20);
	C[1] = midstate[4] ^ ROTL64(nonce, 14);

	state[0] = chi_compat(midstate[0], midstate[1], C[0]);   //^ 0x0000000000000001;

	// experimental/disabled. 
	// stateVeC[0].x = stateVeC[0].x ^ 0x00000001;   // because the rest is leading zeroes :)
	// shouldn't it be .y? only XOR low end of RC 0
	// asm("xor.b32 %0, %0, 0x00000001;" : "+r"(stateVeC[0].x));

	state[0] = state[0] ^ 0x0000000000000001;		  // Round Constant. was RC[0].

	state[1] = chi_compat(midstate[1], C[0], midstate[3]);
	state[2] = chi_compat(C[0], midstate[3], C[1]);
	state[3] = chi_compat(midstate[3], C[1], midstate[0]);
	state[4] = chi_compat(C[1], midstate[0], midstate[1]);

	C[0] = midstate[6] ^ ROTL64(nonce, 20);
	C[1] = midstate[9] ^ ROTR64(nonce, 2);
	state[5] = chi_compat(midstate[5], C[0], midstate[7]);
	state[6] = chi_compat(C[0], midstate[7], midstate[8]);
	state[7] = chi_compat(midstate[7], midstate[8], C[1]);
	state[8] = chi_compat(midstate[8], C[1], midstate[5]);
	state[9] = chi_compat(C[1], midstate[5], C[0]);

	C[0] = midstate[11] ^ ROTL64(nonce, 7);
	C[1] = midstate[13] ^ ROTL64(nonce, 8);
	state[10] = chi_compat(midstate[10], C[0], midstate[12]);
	state[11] = chi_compat(C[0], midstate[12], C[1]);
	state[12] = chi_compat(midstate[12], C[1], midstate[14]);
	state[13] = chi_compat(C[1], midstate[14], midstate[10]);
	state[14] = chi_compat(midstate[14], midstate[10], C[0]);

	C[0] = midstate[15] ^ ROTL64(nonce, 27);
	C[1] = midstate[18] ^ ROTL64(nonce, 16);

	state[15] = chi_compat(C[0], midstate[16], midstate[17]);
	state[16] = chi_compat(midstate[16], midstate[17], C[1]);
	state[17] = chi_compat(midstate[17], C[1], midstate[19]);
	state[18] = chi_compat(C[1], midstate[19], C[0]);
	state[19] = chi_compat(midstate[19], C[0], midstate[16]);

	C[0] = midstate[20] ^ ROTR64(nonce, 1);
	C[1] = midstate[21] ^ ROTR64(nonce, 9);
	C[2] = midstate[22] ^ ROTR64(nonce, 25);

	state[20] = chi_compat(C[0], C[1], C[2]);
	state[21] = chi_compat(C[1], C[2], midstate[23]);
	state[22] = chi_compat(C[2], midstate[23], midstate[24]);
	state[23] = chi_compat(midstate[23], midstate[24], C[0]);
	state[24] = chi_compat(midstate[24], C[0], C[1]);

#if __CUDA_ARCH__ >= 350
#  pragma unroll
#endif
	for (i = 1; i < 23; ++i)
	{
		// Theta
		//for (x = 0; x < 5; ++x)
		//{
		//	C[(x + 6) % 5] = xor5(state[x], state[x + 5], state[x + 10], state[x + 15], state[x + 20], 0   );
		//}
#if __CUDA_ARCH__ >= 600
		C[0] = xor5_lop3(state[4], state[9], state[14], state[19], state[24]);
		C[1] = xor5_lop3(state[0], state[5], state[10], state[15], state[20]);
		C[2] = xor5_lop3(state[1], state[6], state[11], state[16], state[21]);
		C[3] = xor5_lop3(state[2], state[7], state[12], state[17], state[22]);
		C[4] = xor5_lop3(state[3], state[8], state[13], state[18], state[23]);
#else
		C[0] = xor5(state[4], state[9], state[14], state[19], state[24]);
		C[1] = xor5(state[0], state[5], state[10], state[15], state[20]);
		C[2] = xor5(state[1], state[6], state[11], state[16], state[21]);
		C[3] = xor5(state[2], state[7], state[12], state[17], state[22]);
		C[4] = xor5(state[3], state[8], state[13], state[18], state[23]);
#endif
		// test

		// Trusty Old "TEST" (20181224): Manually unrolled loop with the modulus op precomputed from loop iterator.
		D[0] = ROTL64(C[2], 1);
		D[1] = ROTL64(C[3], 1);
		D[2] = ROTL64(C[4], 1);
		D[3] = ROTL64(C[0], 1);
		D[4] = ROTL64(C[1], 1);
		// note: high register usage at this point
		for (x = 0; x < 5; ++x)
		{
			D[x] = ROTL64(C[(x + 2) % 5], 1) ^ C[x];  // FIXME: replace possibly slow modulus.
			state[x] = state[x] ^ D[x];
			state[x + 5] = state[x + 5] ^ D[x];
			state[x + 10] = state[x + 10] ^ D[x];
			state[x + 15] = state[x + 15] ^ D[x];
			state[x + 20] = state[x + 20] ^ D[x];
		}

		// Rho Pi
		C[0] = state[1];

		state[1] = ROTR64(state[6], 20);
		state[6] = ROTL64(state[9], 20);
		state[9] = ROTR64(state[22], 3);
		state[22] = ROTR64(state[14], 25);
		state[14] = ROTL64(state[20], 18);
		state[20] = ROTR64(state[2], 2);
		state[2] = ROTR64(state[12], 21);
		state[12] = ROTL64(state[13], 25);
		state[13] = ROTL64(state[19], 8);

		//state[19] = ROTRby8(state[23]);
		ROTRby8_b(state[23], &state[19]);

		state[23] = ROTR64(state[15], 23);
		state[15] = ROTL64(state[4], 27);
		state[4] = ROTL64(state[24], 14);
		state[24] = ROTL64(state[21], 2);
		state[21] = ROTR64(state[8], 9);            // R9
		state[8] = ROTR64(state[16], 19); // orig
		state[16] = ROTR64(state[5], 28);
		state[5] = ROTL64(state[3], 28);
		state[3] = ROTL64(state[18], 21); // test
		state[18] = ROTL64(state[17], 15);
		state[17] = ROTL64(state[11], 10);
		state[11] = ROTL64(state[7], 6);
		state[7] = ROTL64(state[10], 3);
		state[10] = ROTL64(C[0], 1);

		// chi_compat
		for (/*int32_t or uint32_t*/ x = 0; x < 25; x += 5)
		{
			//C[0] = state[x];			// test, 20180928
			//C[1] = state[x + 1];
			//C[2] = state[x + 2];
			//C[3] = state[x + 3];
			//C[4] = state[x + 4];
			memcpy(&C[0], &state[x], 40);

			state[x] = chi_compat(C[0], C[1], C[2]);
			state[x + 1] = chi_compat(C[1], C[2], C[3]);
			state[x + 2] = chi_compat(C[2], C[3], C[4]);
			state[x + 3] = chi_compat(C[3], C[4], C[0]);
			state[x + 4] = chi_compat(C[4], C[0], C[1]);
		}

		// Iota
		//state[0] = state[0] ^ RClocal[i]; // XOR with round constant for loop iteration i
//		state[0] = ApplyRoundConstant(state[0], i);
		ApplyRoundConstant(&state[0], i);
		//state[0] ^= rc[i];  // apply constant for round # `i`
	}
	// end of for loop

	// UNROLL TEST:
	C[1] = xor5(state[0], state[5], state[10], state[15], state[20]);
	C[2] = xor5(state[1], state[6], state[11], state[16], state[21]);
	C[3] = xor5(state[2], state[7], state[12], state[17], state[22]);
	C[4] = xor5(state[3], state[8], state[13], state[18], state[23]);
	C[0] = xor5(state[4], state[9], state[14], state[19], state[24]);
	/* if (input < 5) return input;
	/*   else return input-5		*/

	D[0] = ROTL64(C[2], 1);
	D[1] = ROTL64(C[3], 1);
	D[2] = ROTL64(C[4], 1);
	state[0] = xor3(state[0], D[0], C[0]);
	state[6] = ROTR64(xor3(state[6], D[1], C[1]), 20);
	state[12] = ROTR64(xor3(state[12], D[2], C[2]), 21);
	state[0] = chi_compat(state[0], state[6], state[12]) ^ 0x8000000080008008;    // was RC[23];

	return bswap_64(state[0], 0/*dummy*/) <= target;  //return bswap_64(state[0]) <= d_target;
}




// [REF]:	*solNonce = theCnt + (static_cast<unsigned long long>(blockDim.x) * blockIdx.x + threadIdx.x);	// cast to the larger type first
//			*solNonce = theCnt + (blockDim.x * blockIdx.x + threadIdx.x);


// the add/multiply here is probably faster than storing the individual thread# of each kernel
#define THREAD_NO (blockDim.x * blockIdx.x + threadIdx.x)

// [MOVEME]?
#ifndef __INTELLISENSE__
__global__ __launch_bounds__(TPB50, 1)
#endif
//void cuda_kernel (const unsigned short deviceID, const uint64_t stride_cnt, const uint8_t engineSelect)
void cuda_kernel (const ushort deviceID, uint64_t* device_solns, uint32_t* device_solns_count, 
	const uint64_t stride_cnt, const uint64_t midstate[25], const uint64_t target, const uint8_t engine_select)
{ // === Hash the input (keccak256) and compare output to difficulty target. ===
	if (engine_select == CUDA_ENGINE_COMPATIBILITY) {
		if (!keccak_compat(deviceID, stride_cnt + static_cast<uint64_t>(THREAD_NO), midstate, target))
			return;
	}
	else if (engine_select == CUDA_ENGINE_HASHBURNER) {
		if (!keccak(deviceID, stride_cnt + static_cast<uint64_t>(THREAD_NO), midstate, target))
			return;
	}

// store counter value which makes the message's hash <= the target, into device's solutions array:
	uint32_t pos = atomicAdd(device_solns_count, 1);	//should return old value (before it was incremented)
	if (pos < MAX_SOLUTIONS)
		device_solns[pos] = static_cast<uint64_t>(stride_cnt + THREAD_NO);	//# of sol'ns found this launch (if any)
	//	memcpy(&device_solns[pos], &soln64, 8);		//addressof(&) seemingly necessary here, if device_solns[pos] implies that the pointer is dereferenced
	//	(*device_solns)[pos] = static_cast<uint64_t>(stride_cnt + THREAD_NO);	//# of sol'ns found this launch (if any)

} //cuda_kernel




int Cuda_GetNumberOfDevices(void)
{ // CUDA_GETNUMBEROFDEVICES: Returns number of CUDA devices detected in the system. If user has obsolete graphics drivers,
  //							could return 0 despite supported cards being present. UI code handles such eventualities.
	int dev_count{ 0 };
	if (!Cuda_Call(cudaGetDeviceCount(&dev_count), "getting device count", 0))
		return 0; // err

	return dev_count;  // # of CUDA devices
}

//
// CUDA_GETDEVICENAMES: Takes a CUDA device index # and returns the device's name string.
std::string Cuda_GetDeviceNames(int devIndex)
{
	cudaDeviceProp myProps;
	cudaGetDeviceProperties(&myProps, devIndex);

	std::string myStr(myProps.name);	 // convert to std::string from C-style string in CDP `myProps`
	return myStr;
}

//
// Cuda_GetDeviceBusInfo: Returns, for now, just the Bus ID, as a C++ std::string
std::string Cuda_GetDeviceBusInfo(int devIndex)
{
	char thePciBusID[14] = "";  // C style char array expected by API function.
	cudaError_t cudaResult = cudaDeviceGetPCIBusId(thePciBusID, 13, devIndex);	// write PCI bus ID into `thePciBusID` array

	if (cudaResult != CUDA_SUCCESS) { /* check for error */
		printf("Error getting PCI Bus ID for CUDA Device # %d! \n", devIndex);
		return "Error";
	}

	// API call successful
	std::string myStr(thePciBusID);
	return myStr;
}



#include "cuda_device.hpp"
#include "generic_solver.hpp"

// [FIXME] / [WIP]:		this should be done during the device init (of the Solver object's cuda_Device or w/e) <---
void Cuda_StoreDevicePciBusIDs(void)
{ /* WIP: make this a member function of class cudaDevice or genericSolver */
	const unsigned short num_of_devices = static_cast<unsigned short>( CudaDevices.size() );
	LOG_IF_F(INFO, HIGHVERBOSITY, "Getting PCI bus IDs for %u devices...", num_of_devices);

	for (uint8_t i = 0; i < num_of_devices; ++i) {	/* was:		i < CUDA_MAX_DEVICES */
		if (CudaDevices[i]->solver->enabled != true) {
//			LOG_F(WARNING, "Not getting PCI bus ID for disabled device %d.", i);
			continue;  // skip disabled devices
		}

// this should still work because CUDA devices are inited and numbered from 0 before
// solvers of other types. <--- [wip] / [fixme]
		gCudaDevicePciBusIDs[i] = Solvers[i]->cuda_device->deviceProperties.pciBusID;	// <--- [FIXME].
//		gCudaDevicePciBusIDs[i] = this->cuda_device->deviceProperties.pciBusID;			// <--- OOP approach
		LOG_IF_F(INFO, HIGHVERBOSITY, "Got PCI Bus ID for CUDA device # %d :  %d \n", i, gCudaDevicePciBusIDs[i]);
	}
}


void Cuda_PrintDeviceCounters() {	/* [TODO]: make this a member function of genericSolver class */
	printf("\n\n -- Device Counters (cnt): -- \n");
	const size_t number_of_devices = CudaDevices.size();
	for (unsigned short dev = 0; dev < CUDA_MAX_DEVICES; ++dev) {
		// ...

		if (CudaDevices[dev]->solver->enabled) {	/* was:		gCudaDeviceEnabled */
			printf("CUDA dev #%d: %" PRIu64 " \n", dev, cnt[dev]);
			cnt[dev] > CNT_OFFSET ? printf("   (greater than CNT_OFFSET %" PRIu64 " \n", CNT_OFFSET) : printf("\n");
		}
	}
	printf("\n\n");
}


//
// === Functions for class cudaDevice: ===
//

bool cudaDevice::Check(void)
{ //if not initialized, return false?
	if(!h_solutions || !h_solutions_count || !d_solutions || !d_solutions_count) {
		this->SetStatus(DeviceStatus::Fault);
		solver->solver_status = SolverStatus::DeviceError;
		return false;
	}
	return true;	//no null pointers
}


bool cudaDevice::find_solutions (unsigned short* solns_found)	// [MOVEME] ?
{ // launch mining kernels and pass back solutions count (if any) to genericSolver::Solve()
  // [WIP]:	the count should be initialized to 0 before launching, and good idea to initialize the device-side `dev_solutions` buffer also. <--- [WIP].

	if (!this->Check()) return false;	//check host<->device memory pointers

#ifndef __INTELLISENSE__
	cuda_kernel <<< gCuda_Grid[dev_no], gCuda_Block[dev_no] >>>
		(dev_no, d_solutions, d_solutions_count, cnt[dev_no], solver->midstate, solver->target, gCuda_Engine[dev_no]);
#endif
	
// synchronize host/device (blocking call- see device flags.)
	if (!Cuda_Call(cudaDeviceSynchronize(), "synchronizing w/ device", dev_no)) {
		this->SetStatus(DeviceStatus::Fault);
		ftime(&solver->tEnd);
		return false;	//error
	}

	if (!this->Check()) return false;	//check the pointers

// copy back # of solutions found this launch, if any.
	if (!Cuda_Call(cudaMemcpy(/*&*/h_solutions_count, /*&*/d_solutions_count, sizeof(uint32_t), cudaMemcpyDeviceToHost), "reading sol'n count", this->dev_no)) {		/*<---- d_solution cannot be read in host function */
		this->SetStatus(DeviceStatus::MemoryFault);
		ftime(&solver->tEnd);
		return false;	//error
	}
	ftime(&solver->tEnd);	// [todo]: improve "old" hashrate calculation code
	
// [MOVEME] ?	timing & hashrate stuff. <---
	cnt[this->dev_no] += solver->threads;		// search space covered by this device		<--- make `cnt` a member of genericSolver [TODO] <--
	solver->hash_count += solver->threads;		// hashrate, stats display
	solver->solve_time = (double)(((solver->tEnd.time * 1000) + solver->tEnd.millitm) - ((solver->tStart.time * 1000) + solver->tStart.millitm)) / 1000;
	solver->hash_rate = solver->solve_time > 0 ? (static_cast<double>(solver->hash_count) / solver->solve_time) / 1000000 : 0;	//to megahashes/sec. (never div. by 0!)
	if (DEBUGMODE) {//REMOVE<--
		printf("\n\n-- cuda dev.# %d solve time: ~%f sec., hash rate: ~%f MH/s --\n", dev_no, solver->solve_time, solver->hash_rate);			//REMOVE<--
		printf("stride cnt: %" PRIu64 " | %" PRIx64 "(hex)	|	hashes computed: %" PRIu64 " ", cnt[dev_no], cnt[dev_no], solver->hash_count); }//REMOVE<--
// [MOVEME] ?	timing & hashrate stuff. <---

//	if(h_solutions_count != nullptr && h_solutions != nullptr)...	<--- [WIP] <---
	if (*h_solutions_count == 0)
		return true;	//OK, no solutions found this run.
	else if (*h_solutions_count > 0 && *h_solutions_count <= MAX_SOLUTIONS)		// was < MAX_SOLUTIONS
	{ //solution(s) found by the device, copy to host:
		//ftime(&solver->tEnd);		// <- hashrate calc, but including time taken copying memory. [MOVEME] ?
		*solns_found = static_cast<unsigned short>(*h_solutions_count);	//<--- for calling func.
		LOG_IF_F(INFO, DEBUGMODE, "Solver# %u: Copying back %u solutions from cuda device# %d", solver->solver_no, *solns_found, dev_no);
		if (!Cuda_Call(cudaMemcpyFromSymbol(h_solutions, d_solutions, (*solns_found) * sizeof(uint64_t), 0, cudaMemcpyDeviceToHost), "reading solutions", dev_no)) {
			status = DeviceStatus::MemoryFault;						//
			solver->device_status = DeviceStatus::MemoryFault;		//<---
			return false;		//err = true;
		}
		// ===Copy OK:===		// if(!err) {
		LOG_IF_F(INFO, DEBUGMODE, "Copied back %u probable solutions from cuda device# %d ", *solns_found, dev_no);	//<-- remove
		//*solns_found = *h_solutions_count;	//for caller genericSolver::Solve(). [redundant?] <---

		bool err{ false };
		if (!this->enqueue_solutions(/*howmany?*/)) err = true;	//<-- proceed to clear any solutions before returning. (redundant, just clean up?)
		if (!this->clear_solutions()) err = true;				//<-- do in the calling function instead?

		if (!err) return true;	// OK: Solution(s) found!
		 else {
			//<--- device lost?	or bug. [WIP] <----
			this->status = DeviceStatus::MemoryFault;			//cuda device status
			solver->device_status = DeviceStatus::MemoryFault;	//solver status
			return false;	//err
		}
		//this->ResetHashrateCalc();	// [MOVEME]? [WIP]: Move setting of tStart/tEnd, etc. to correct location!! <----
	}
	else { // *h_solutions_count >= MAX_SOLUTIONS ? should never happen. but host/device solutions/count must be cleared.
		LOG_IF_F(WARNING, DEBUGMODE, "Too many solutions- bailing out!" BUGIFHAPPENED);
		return false;
	}

	//return true;	//OK: Solution(s) found!
}

// TESTING: this was in cuda_device.cpp
// [WIP]: consider a generic allocate function in genericSolver.
bool cudaDevice::Allocate(void)
{ // allocate memory for host<->device i/o. for sending solutions back to the host

// [WIP] / [NOTE]: no longer using mapped pinned memory!!
// (removed stuff from here)

// get device pointers to memory of solution buffer and count
	if (!Cuda_Call(cudaGetSymbolAddress((void**)&d_solutions, solns_data), "getting address of device symbol solns_data [array]", static_cast<int>(dev_no)))
		return false;
	if (!Cuda_Call(cudaGetSymbolAddress((void**)&d_solutions_count, solns_count), "getting address of device symbol solns_count", static_cast<int>(dev_no)))
		return false;	// [WIP]: addressof(&) on 2nd argument?
// check pointers...
	if (!h_solutions || !h_solutions_count || !d_solutions || !d_solutions_count) {
		LOG_F(ERROR, "Null pointer(s)- CUDA Device# %d unavailable?", dev_no);
		//return false; // redundant
	}
	//else {
	//success! remove these (viewing mem. address of host/device pointers):
		if (DEBUGMODE) printf("h_solutions		: 0x%p \n", (void*)this->h_solutions);			//<-- DEBUG, REMOVE
		if (DEBUGMODE) printf("h_solutions_count	: 0x%p \n", (void*)this->h_solutions_count);	//<--
		if (DEBUGMODE) printf("d_solutions		: 0x%p \n", (void*)this->d_solutions);//<---
		if (DEBUGMODE) printf("d_solutions_count	: 0x%p \n", (void*)this->d_solutions_count);	//<--
	//}
	//initialize the host-side memory (semi-redundant)
	LOG_IF_F(INFO, DEBUGMODE, "DEBUG: initializing cuda dev.# %d memory", this->dev_no);
	try { // [TESTME]: figuring out where exceptions would occur, in the event of a lost device
		memset(this->h_solutions, 0xFF, SOLUTIONS_SIZE);
		*this->h_solutions_count = 0;	//<--- !
	}
	catch (...) {
		LOG_F(ERROR, "in cudaDevice::Allocate(): exception caught. Device unavailable?");	//which one?
		// [TODO]: exception handling, currently aborts on any exception thrown. <---
		return false;
	}

	return true;	//OK
}