CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -mavx512f -march=native -fopenmp -Iinclude
LDFLAGS ?= -lboost_filesystem -lboost_system

BIN_DIR := bin
HEADERS := $(wildcard include/*.h)
TARGETS := $(BIN_DIR)/dataset_gt $(BIN_DIR)/rknn_query

.PHONY: all clean

all: $(TARGETS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/dataset_gt: tools/dataset_gt.cpp $(HEADERS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) tools/dataset_gt.cpp -o $@ $(LDFLAGS)

$(BIN_DIR)/rknn_query: src/rknn_query.cpp $(HEADERS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) src/rknn_query.cpp -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGETS)
