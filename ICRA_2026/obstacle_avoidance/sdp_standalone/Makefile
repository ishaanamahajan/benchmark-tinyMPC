# Makefile for TinyMPC SDP Solver

CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra
INCLUDES = -I../tinympc/tinympc_teensy/src  # For Eigen headers
TARGET = sdp_test
SOURCES = main.cpp sdp_solver.cpp

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET) $(SOURCES)

# Clean target
clean:
	rm -f $(TARGET) *.csv

# Run target
run: $(TARGET)
	./$(TARGET)

# Help target
help:
	@echo "Available targets:"
	@echo "  all     - Build the SDP solver"
	@echo "  clean   - Remove build files"
	@echo "  run     - Build and run the solver"
	@echo "  help    - Show this help message"

.PHONY: all clean run help


