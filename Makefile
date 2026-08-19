CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -pedantic

TARGET := anqi_sample
SOURCES := src/anqi_sample.cpp

.PHONY: all clean sample

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $@

sample: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
