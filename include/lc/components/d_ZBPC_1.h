#pragma once

#include "include/d_RZE.h"

static __device__ inline bool d_ZBPC_1(int& csize, byte in[CS], byte out[CS], byte temp[CS])
{
  return d_RZE<byte>(csize, in, out, temp);
}

static __device__ inline void d_iZBPC_1(int& csize, byte in[CS], byte out[CS], byte temp[CS])
{
  d_iRZE<byte>(csize, in, out, temp);
}
