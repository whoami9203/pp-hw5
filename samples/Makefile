CXX := g++
CXXFLAGS := -std=c++17 -O3
HIPFLAGS := $(CXXFLAGS)
TARGET := hw5
SEQUENTIAL := nbody


.PHONY: all
all: $(TARGET)

.PHONY: hw5
hw5: hw5.cpp
	hipcc $(HIPFLAGS) -o hw5 hw5.cpp
.PHONY: seq
seq: nbody.cc
	$(CXX) $(CXXFLAGS) -o nbody nbody.cc
.PHONY: cuda
cuda: hw5_cuda2.cu
	nvcc $(CXXFLAGS) -o hw5_cuda hw5_cuda2.cu

.PHONY: clean
clean:
	rm -f $(TARGET) $(SEQUENTIAL)


