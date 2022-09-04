# SPDX-License-Identifier: MIT

BEX_SDK_DIR:=/s/bex-dev/bex/baltamatica/bex
NLOPT_BUILD_DIR:=../../build
NLOPT_INC_DIR:=../api

CXXFLAGS:=-Wall -std=c++17 -shared -O3 -fPIC -I.. \
	-I$(BEX_SDK_DIR)/include/  \
	-L$(BEX_SDK_DIR)/lib -lbex  \
	-I$(NLOPT_INC_DIR) -I$(NLOPT_BUILD_DIR) \

BEX_EXT:=dll


all: nlopt/main.$(BEX_EXT)

nlopt/main.$(BEX_EXT): nlopt_optimize-bex.c $(NLOPT_INC_DIR)/nlopt.h nlopt_bex.cpp
	$(CXX) $(CXXFLAGS) -o nlopt/main.$(BEX_EXT) nlopt_optimize-bex.c nlopt_bex.cpp $(NLOPT_BUILD_DIR)/libnlopt.dll.a
	cp $(NLOPT_BUILD_DIR)/libnlopt.dll  nlopt/libnlopt.dll

clean:
	rm -f *.o *.obj *.a *.$(BEX_EXT)
	rm -f nlopt/main.$(BEX_EXT)
