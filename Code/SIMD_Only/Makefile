SRC     = RadixSAC_Smd.cpp
OBJECTS = RadixSAC_Smd.o
TARGET  = RadixSAC_Smd

INCLUDES =
LDFLAGS  =
LIBS     = -lm

# For gcc compiler
#CXX      = g++
# For icc compiler
CXX      = icc

CXXFLAGS = -O3 -Wall -std=c++0x -m64  #-pg -fno-inline
CXXFLAGS += -fopenmp -xCORE-AVX512 -mtune=skylake -xHost -DUSE_AVX512BW -qopt-zmm-usage=high #  -AVX512VPOPCNTDQ

#################################################################################
all: $(TARGET)
#################################################################################
$(TARGET): clean $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET) $(OBJECTS) $(LDFLAGS) $(LIBS)

#################################################################################
VLUVPItest.o: RadixSAC_Smd_Omp.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c RadixSAC_Smd_Omp.cpp -o RadixSAC_Smd_Omp.o 

VLUVPItest.o: RadixSAC_Smd_Omp.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c RadixSAC_Smd_Omp.cpp 

clean:
	rm -f $(TARGET) $(OBJECTS)
