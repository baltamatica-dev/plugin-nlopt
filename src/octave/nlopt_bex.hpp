/* SPDX-License-Identifier: MIT */
#pragma once
#include <iostream>
#include <cassert>
#include <bex/bex.hpp>
#include <math.h>


#define BUILD_WITH_BEX_WARPPER
#define bexWarnMsgTxt  bxPrintf
#define bxFree		free
#define bxCalloc	calloc
#define bxIsNaN     isnan

extern double bxGetScalar(const bxArray*);
extern bool bxIsNumeric(const bxArray*);
extern int mexCallMATLAB(int nlhs, bxArray *plhs[], int nrhs, bxArray *prhs[], const char *functionName);

extern void nlopt_optimize(int nlhs, bxArray *plhs[], int nrhs, const bxArray *prhs[]);
