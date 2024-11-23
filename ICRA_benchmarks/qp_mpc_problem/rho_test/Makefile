CXX = g++
CXXFLAGS = -Wall -I.

all: test

test: main.cpp rho_benchmark.cpp rho_benchmark_recompute.cpp
	$(CXX) $(CXXFLAGS) -o test $^

clean:
	rm -f test