# QuasiGraph Makefile
# Simple build system for development

CXX = g++
CXXFLAGS = -std=c++20 -O3 -march=native -Wall -Wextra -Iinclude
LDFLAGS = 

# Source files
SOURCES = $(wildcard src/*.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
HEADERS = $(wildcard include/QuasiGraph/*.h)

# Targets
LIBRARY = libquasigraph.a
EXAMPLES = examples/quasigraph_examples
BENCHMARK = benchmarks/quasigraph_bench
TESTS = tests/simple_tests

.PHONY: all clean library examples benchmark tests install

all: library examples benchmark tests

# Build static library
library: $(LIBRARY)

$(LIBRARY): $(OBJECTS)
	@echo "Creating static library..."
	ar rcs $@ $^
	@echo "✅ Library created: $(LIBRARY)"

# Build examples
examples: $(EXAMPLES)

$(EXAMPLES): examples/main.cpp $(LIBRARY)
	@echo "Building examples..."
	$(CXX) $(CXXFLAGS) -o $@ $< -L. -lquasigraph
	@echo "✅ Examples built: $(EXAMPLES)"

# Build benchmark
benchmark: $(BENCHMARK)

$(BENCHMARK): benchmarks/benchmark.cpp $(LIBRARY)
	@echo "Building benchmark..."
	$(CXX) $(CXXFLAGS) -o $@ $< -L. -lquasigraph
	@echo "✅ Benchmark built: $(BENCHMARK)"

# Build tests
tests: $(TESTS)

$(TESTS): tests/simple_test_runner.cpp $(LIBRARY)
	@echo "Building tests..."
	$(CXX) $(CXXFLAGS) -o $@ $< -L. -lquasigraph
	@echo "✅ Tests built: $(TESTS)"

# Compile source files
%.o: %.cpp $(HEADERS)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Run tests
test: tests
	@echo "Running tests..."
	./$(TESTS)

# Run examples
run-examples: examples
	@echo "Running examples..."
	./$(EXAMPLES)

# Run benchmark
run-benchmark: benchmark
	@echo "Running benchmark..."
	./$(BENCHMARK)

# Install (simple version)
install: library
	@echo "Installing QuasiGraph..."
	sudo mkdir -p /usr/local/include/QuasiGraph
	sudo cp include/QuasiGraph/*.h /usr/local/include/QuasiGraph/
	sudo cp $(LIBRARY) /usr/local/lib/
	@echo "✅ QuasiGraph installed to /usr/local"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(OBJECTS) $(LIBRARY) $(EXAMPLES) $(BENCHMARK) $(TESTS)
	rm -f benchmark_report.txt
	@echo "✅ Clean completed"

# Development targets
dev-build: clean all test
	@echo "🚀 Development build completed successfully!"

release-build: CXXFLAGS += -DNDEBUG
release-build: clean all
	@echo "🚀 Release build completed!"

# Quick build for development
quick: $(LIBRARY) $(TESTS)
	./$(TESTS)

# Help
help:
	@echo "QuasiGraph Build System"
	@echo "======================="
	@echo ""
	@echo "Targets:"
	@echo "  all           - Build everything (library, examples, benchmark, tests)"
	@echo "  library       - Build static library only"
	@echo "  examples      - Build examples"
	@echo "  benchmark     - Build benchmark suite"
	@echo "  tests         - Build test suite"
	@echo "  test          - Build and run tests"
	@echo "  run-examples  - Build and run examples"
	@echo "  run-benchmark - Build and run benchmark"
	@echo "  install       - Install library system-wide"
	@echo "  clean         - Remove build artifacts"
	@echo "  dev-build     - Full development build with tests"
	@echo "  release-build - Optimized release build"
	@echo "  quick         - Quick build and test"
	@echo "  help          - Show this help"
	@echo ""
	@echo "Examples:"
	@echo "  make dev-build     # Full development build"
	@echo "  make quick         # Quick test"
	@echo "  make run-benchmark # Run performance tests"
