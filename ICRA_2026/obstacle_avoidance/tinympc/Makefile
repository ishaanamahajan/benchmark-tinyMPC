# Makefile for TinyMPC with integrated SDP projections

CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -DENABLE_SDP_PROJECTION
INCLUDES = -Itinympc_teensy/src
TARGET = test_tinympc_sdp
SOURCES = test_tinympc_with_sdp.cpp tinympc_teensy/src/admm.cpp

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
	@echo "  all     - Build TinyMPC with SDP projections"
	@echo "  clean   - Remove build files"
	@echo "  run     - Build and run the test"
	@echo "  help    - Show this help message"

.PHONY: all clean run help
