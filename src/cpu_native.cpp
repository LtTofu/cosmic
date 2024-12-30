#pragma once
// cpusolver_native.cpp : proto-CPU Solver for COSMiC V4, Native C++ portion
// 2020 LtTofu  (see Thanks in hashburner.cu!)
#include <stdio.h>
#include <cinttypes>
#include <intrin.h>
#include "cpu_solver.h"

#define ROTL64(x, y) (((x) << (y)) ^ ((x) >> (64 - (y))))  // rotate left macro
#define ROTR64(x, y) (((x) >> (y)) ^ ((x) << (64 - (y))))  // rotate right macro

//
// ROUND CONSTANTS
const uint64_t RC[24] =
{
  0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
  0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
  0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
  0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
  0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
  0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
  0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
  0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};

// Globals
uint64_t cpu_mid[25] {0};
uint64_t cpu_target {0};
uint64_t cpuThreadBestResults[DEF_MAX_CPU_SOLVERS] = { 0 };  // will be set to UINT64_MAX

//extern uint64_t g64BitTarget;  // 64-bit version of bignum target. (in: cosmicwind.cpp)

//
// Bytes Reversal function (CPU "generic")
uint64_t bswap_64_cpu(const uint64_t input)
{
	uint64_t output = _byteswap_uint64(input);  // intrinsics are cool
	return output;
}

// ROTL64 : Bitwise Left Rotation (using instrinsic)
uint64_t ROTL64intr(const uint64_t input, const unsigned short offset)
{
	return _rotl64(input, offset);
	//return 0;
}

// ROTR64 : Bitwise Right Rotation (using instrinsic)
uint64_t ROTR64intr(const uint64_t input, const unsigned short offset)
{
	return _rotr64(input, offset);
	//return 0;
}

// XOR5 helper function (CPU "generic")
uint64_t xor5(const uint64_t a, const uint64_t b, const uint64_t c, const uint64_t d, const uint64_t e)
{
	return a ^ b ^ c ^ d ^ e;
}

// XOR3 helper function (CPU "generic")
uint64_t xor3(const uint64_t a, const uint64_t b, const uint64_t c)
{
	return a ^ b ^ c;
}

// CHI: Combination of bitwise operations on 3 inputs
uint64_t chi(const uint64_t a, const uint64_t b, const uint64_t c)
{
	return a ^ ((~b) & c);
}

// cpu_hash: CPU hashing function (new, WIP! might be awful!)
bool cpu_hash(const uint64_t nonce, const unsigned short theThread,
	uint64_t *state, uint64_t *C, uint64_t *D, uint64_t *scratch64)
{
	//uint64_t state[25], C[5], D[5], scratch64;
	//uint64_t* state = new uint64_t[25];
	//uint64_t* C = new uint64_t[5];
	//uint64_t* D = new uint64_t[5];
	//uint64_t scratch64;
	//memset(state, 0, 200);  // 200 bytes emptied
	//uint32_t i{ 0 }, x{ 0 };
	uint8_t i{ 0 }, x{ 0 };

	C[0] = cpu_mid[2] ^ ROTR64(nonce, 20);

	//C[1] = cpu_mid[4] ^ nonce * 16384;               // ROTL by 14
	C[1] = cpu_mid[4] ^ ROTL64(nonce, 14);

	state[0] = chi(cpu_mid[0], cpu_mid[1], C[0]) ^ 0x0000000000000001;   // immediate in order to avoid
	state[1] = chi(cpu_mid[1], C[0], cpu_mid[3]);                        // going out to constant memory
	state[2] = chi(C[0], cpu_mid[3], C[1]);
	state[3] = chi(cpu_mid[3], C[1], cpu_mid[0]);
	state[4] = chi(C[1], cpu_mid[0], cpu_mid[1]);

	//C[0] = cpu_mid[6] ^ nonce * 1048576;            // ROTL by 20
	C[0] = cpu_mid[6] ^ ROTL64(nonce, 20);

	C[1] = cpu_mid[9] ^ ROTR64(nonce, 2);
	state[5] = chi(cpu_mid[5], C[0], cpu_mid[7]);
	state[6] = chi(C[0], cpu_mid[7], cpu_mid[8]);
	state[7] = chi(cpu_mid[7], cpu_mid[8], C[1]);
	state[8] = chi(cpu_mid[8], C[1], cpu_mid[5]);
	state[9] = chi(C[1], cpu_mid[5], C[0]);

	// VER A. C[0] = cpu_mid[11] ^ (nonce * 128);              // ROTL by 7
	// VER B. C[0] = cpu_mid[11] ^ ROTL64(nonce, 7);

	// VER A. C[1] = cpu_mid[13] ^ (nonce * 256);              // ROTL by 8
	// VER B. C[1] = cpu_mid[13] ^ ROTL64(nonce, 8);

	// experimental "should be faster" delta version C
	*scratch64 = ROTL64(nonce, 7);
	C[0] = cpu_mid[11] ^ *scratch64;
	C[1] = cpu_mid[13] ^ ROTL64(*scratch64, 1);
	// end

	state[10] = chi(cpu_mid[10], C[0], cpu_mid[12]);
	state[11] = chi(C[0], cpu_mid[12], C[1]);
	state[12] = chi(cpu_mid[12], C[1], cpu_mid[14]);
	state[13] = chi(C[1], cpu_mid[14], cpu_mid[10]);
	state[14] = chi(cpu_mid[14], cpu_mid[10], C[0]);

	C[0] = cpu_mid[15] ^ ROTL64(nonce, 27);				// ROTL by 27
	C[1] = cpu_mid[18] ^ ROTL64(nonce, 16);              // ROTL by 16 (TODO: implement Delta)
	state[15] = chi(C[0], cpu_mid[16], cpu_mid[17]);
	state[16] = chi(cpu_mid[16], cpu_mid[17], C[1]);
	state[17] = chi(cpu_mid[17], C[1], cpu_mid[19]);
	state[18] = chi(C[1], cpu_mid[19], C[0]);
	state[19] = chi(cpu_mid[19], C[0], cpu_mid[16]);

	C[0] = cpu_mid[20] ^ ROTR64intr(nonce, 1);
	C[1] = cpu_mid[21] ^ ROTR64intr(nonce, 9);
	C[2] = cpu_mid[22] ^ ROTR64intr(nonce, 25);
	state[20] = chi(C[0], C[1], C[2]);
	state[21] = chi(C[1], C[2], cpu_mid[23]);
	state[22] = chi(C[2], cpu_mid[23], cpu_mid[24]);
	state[23] = chi(cpu_mid[23], cpu_mid[24], C[0]);
	state[24] = chi(cpu_mid[24], C[0], C[1]);

	#pragma loop( hint_parallel(23) )
	for (i = 1; i < 23; ++i)
	{
		// Theta
		#pragma loop( hint_parallel(5) )
		for (x = 0; x < 5; ++x)
		{									// OI !! MODULUS (FIXME)
			C[(x + 6) % 5] = xor5(state[x], state[x + 5], state[x + 10], state[x + 15], state[x + 20]);
		}  // fixed this already on the cuda side lol

		// Version A
		#pragma loop( hint_parallel(5) )
		for (x = 0; x < 5; ++x)
		{
			D[x] = ROTL64(C[(x + 2) % 5], 1);
			state[x] = xor3(state[x], D[x], C[x]);
			state[x + 5] = xor3(state[x + 5], D[x], C[x]);
			state[x + 10] = xor3(state[x + 10], D[x], C[x]);
			state[x + 15] = xor3(state[x + 15], D[x], C[x]);
			state[x + 20] = xor3(state[x + 20], D[x], C[x]);
		}
		// Version B
		/*for (uint32_t x{ 0 }; x < 5; ++x)
		{
			D[x] = ROTL64(C[(x + 2) % 5], 1) ^ C[x];
			state[x] = state[x] ^ D[x];
			state[x + 5] = state[x + 5] ^ D[x];
			state[x + 10] = state[x + 10] ^ D[x];
			state[x + 15] = state[x + 15] ^ D[x];
			state[x + 20] = state[x + 20] ^ D[x];
		}*/

		// Rho Pi
		C[0] = state[1];

		//__m128i foo;
		//foo.m128i_i64[0] = state[6];
		//foo.m128i_i64[1] = state[9];
		//_mm_roti_epi64();
		//_mm_rot_epi64();

		//printf("EXPEROT Result:  %" PRIx64 " \n", _mm_roti_epi64(state[6], 20) );
		//printf("REGUROT Result:  %" PRIx64 " \n", ROTR64(state[6], 20));
		state[1] = ROTR64intr(state[6], 20);
		state[6] = ROTL64intr(state[9], 20);
		state[9] = ROTR64intr(state[22], 3);
		state[22] = ROTR64intr(state[14], 25);
		state[14] = ROTL64intr(state[20], 18);
		state[20] = ROTR64intr(state[2], 2);
		state[2] = ROTR64intr(state[12], 21);
		state[12] = ROTL64intr(state[13], 25);
		state[13] = ROTL64intr(state[19], 8);
		state[19] = ROTR64intr(state[23], 8);
		state[23] = ROTR64intr(state[15], 23);
		state[15] = ROTL64intr(state[4], 27);
		state[4] = ROTL64intr(state[24], 14);
		state[24] = ROTL64intr(state[21], 2);
		state[21] = ROTR64intr(state[8], 9);
		state[8] = ROTR64intr(state[16], 19);
		state[16] = ROTR64intr(state[5], 28);
		state[5] = ROTL64intr(state[3], 28);
		state[3] = ROTL64intr(state[18], 21);
		state[18] = ROTL64intr(state[17], 15);
		state[17] = ROTL64intr(state[11], 10);
		state[11] = ROTL64intr(state[7], 6);
		state[7] = ROTL64intr(state[10], 3);
		state[10] = ROTL64intr(C[0], 1);

		// Chi
		#pragma loop( hint_parallel(5) )
		for (x = 0; x < 25; x += 5)
		{
			C[0] = state[x];
			C[1] = state[x + 1];
			C[2] = state[x + 2];
			C[3] = state[x + 3];
			C[4] = state[x + 4];
			state[x] = chi(C[0], C[1], C[2]);
			state[x + 1] = chi(C[1], C[2], C[3]);
			state[x + 2] = chi(C[2], C[3], C[4]);
			state[x + 3] = chi(C[3], C[4], C[0]);
			state[x + 4] = chi(C[4], C[0], C[1]);
		}

		// Iota
		state[0] = state[0] ^ RC[i];
	}

	// FIXME: Still has a modulus !
	#pragma loop( hint_parallel(5) )
	for (x = 0; x < 5; ++x)
	{
		C[(x + 6) % 5] = xor5(state[x], state[x + 5], state[x + 10], state[x + 15], state[x + 20]);
	}

	D[0] = ROTL64(C[2], 1);
	D[1] = ROTL64(C[3], 1);
	D[2] = ROTL64(C[4], 1);

	state[0] = xor3(state[0], D[0], C[0]);
	state[6] = ROTR64intr(xor3(state[6], D[1], C[1]), 20);
	state[12] = ROTR64intr(xor3(state[12], D[2], C[2]), 21);

	state[0] = chi(state[0], state[6], state[12]) ^ 0x8000000080008008;    // RC[23]
	//uint64_t foo = bswap_64(state[0]);
	//if (foo < cpuThreadBestResults[theThread])
	//	cpuThreadBestResults[theThread] = foo;
	
	//printf("Hashing result before bswap is: %" PRIx64 " ( %" PRIu64 " ) \n", state[0], state[0] );
	if (bswap_64_cpu(state[0]) <= cpu_target)
	{
		/*free(state);
		free(C);
		free(D);*/
		printf(">> hashing result %" PRIx64 " is <= cpu_target %" PRIx64 ". Before bswap, state[0]: %" PRIx64 " \n", bswap_64_cpu(state[0]), state[0], cpu_target);  // ew, fixme
		return true;
	}
	else
	{
		/*free(state);
		free(C);
		free(D); */
		return false;
	}
	// neater version:
	// return bswap_64(state[0]) <= cpu_target;  // return TRUE if less than cpu_target after byte reversal
												 // uint64_t foo = bswap_64(state[0]);
}

// STORECPUTHREADMIDSTATE():
void StoreCpuThreadMidState(const uint64_t* message)
{
//	printf("input message for cpu thread (bytes): \n");
//	print_bytes((uint8_t*)message, 84, "cpu thread init-mesg bytes"); */ }
	uint64_t C[4]{ 0 };
	uint64_t D[5]{ 0 };

	C[0] = message[0] ^ message[5] ^ message[10] ^ 0x100000000ull;
	C[1] = message[1] ^ message[6] ^ 0x8000000000000000ull;
	C[2] = message[2] ^ message[7];
	C[3] = message[4] ^ message[9];

	D[0] = ROTL64(C[1], 1) ^ C[3];
	D[1] = ROTL64(C[2], 1) ^ C[0];
	D[2] = ROTL64(message[3], 1) ^ C[1];
	D[3] = ROTL64(C[3], 1) ^ C[2];
	D[4] = ROTL64(C[0], 1) ^ message[3];

// 'th' is thread (function parameter)
	cpu_mid[0] = message[0] ^ D[0];
	cpu_mid[1] = ROTL64(message[6] ^ D[1], 44);
	cpu_mid[2] = ROTL64(D[2], 43);
	cpu_mid[3] = ROTL64(D[3], 21);
	cpu_mid[4] = ROTL64(D[4], 14);
	cpu_mid[5] = ROTL64(message[3] ^ D[3], 28);
	cpu_mid[6] = ROTL64(message[9] ^ D[4], 20);
	cpu_mid[7] = ROTL64(message[10] ^ D[0] ^ 0x100000000ull, 3);
	cpu_mid[8] = ROTL64(0x8000000000000000ull ^ D[1], 45);
	cpu_mid[9] = ROTL64(D[2], 61);
	cpu_mid[10] = ROTL64(message[1] ^ D[1], 1);
	cpu_mid[11] = ROTL64(message[7] ^ D[2], 6);
	cpu_mid[12] = ROTL64(D[3], 25);
	cpu_mid[13] = ROTL64(D[4], 8);
	cpu_mid[14] = ROTL64(D[0], 18);
	cpu_mid[15] = ROTL64(message[4] ^ D[4], 27);
	cpu_mid[16] = ROTL64(message[5] ^ D[0], 36);
	cpu_mid[17] = ROTL64(D[1], 10);
	cpu_mid[18] = ROTL64(D[2], 15);
	cpu_mid[19] = ROTL64(D[3], 56);
	cpu_mid[20] = ROTL64(message[2] ^ D[2], 62);  // TODO: rotr by 2 instead. similar on other lines.
	cpu_mid[21] = ROTL64(D[3], 55);
	cpu_mid[22] = ROTL64(D[4], 39);
	cpu_mid[23] = ROTL64(D[0], 41);
	cpu_mid[24] = ROTL64(D[1], 2);
}
