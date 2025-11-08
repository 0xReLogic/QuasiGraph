# QuasiGraph

[![CI](https://github.com/0xReLogic/QuasiGraph/workflows/CI/badge.svg)](https://github.com/0xReLogic/QuasiGraph/actions/workflows/ci.yml)
[![Release](https://github.com/0xReLogic/QuasiGraph/workflows/Release/badge.svg)](https://github.com/0xReLogic/QuasiGraph/actions/workflows/release.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Performance](https://img.shields.io/badge/Complexity-O(n%5E1.47)-green.svg)](https://github.com/0xReLogic/QuasiGraph#performance)

High-performance C++20 graph optimization library with SIMD-accelerated operations.

## Overview

QuasiGraph is a graph library implementing optimized algorithms for graph operations, independent set solving, and structural analysis.

## Features

- Bit-Parallel SIMD Operations with AVX2 support
- Graph operations with efficient vertex/edge management
- Independent Set Solver with multiple algorithms
- Structural decomposition and graph analysis
- Social network analysis tools
- Performance benchmarking suite

## Quick Start

```cpp
#include "QuasiGraph/Graph.h"

// Create a graph
QuasiGraph::Graph graph;
graph.addVertex(0);
graph.addVertex(1);
graph.addEdge(0, 1);

// Get graph properties
std::cout << "Vertices: " << graph.getVertexCount() << std::endl;
std::cout << "Edges: " << graph.getEdgeCount() << std::endl;
```

## Build

```bash
# Using CMake
mkdir build && cd build
cmake ..
make

# Using Makefile
make
```

## Examples

```bash
# Run basic examples
g++ -std=c++20 -Iinclude -o example examples/basic_usage.cpp src/Graph.cpp
./example

# Run tests
g++ -std=c++20 -Iinclude -o test tests/basic_tests.cpp src/Graph.cpp
./test
```

## Performance

Real benchmark results on modern hardware (actual execution times):

| Vertices | Edges   | Time (μs) | Memory (MB) | Result  |
|----------|---------|-----------|-------------|---------|
| 1,000    | 24.8K   | 161       | 0.83        | Success |
| 2,000    | 99.8K   | 435       | 3.17        | Success |
| 5,000    | 625K    | 1,430     | 19.39       | Success |
| 10,000   | 2.50M   | 2,907     | 76.95       | Success |

**Performance Characteristics:**
- Time Complexity: O(n^1.47) measured
- Size Scaling: 100x (100 → 10,000 vertices)
- Memory: ~7.7 MB per 1K vertices
- Success Rate: 100%

**Optimizations:**
- Bit-parallel mode: `graph.enableBitParallelMode()`
- SIMD AVX2 for set operations
- Hardware POPCNT for bit counting
- Compiler: GCC 12, -O3 -march=native

## License

MIT License - See LICENSE file for details

## References

- Chudnovsky, M., et al. Tree independence number algorithms. arXiv:2405.00265, SODA 2025.
- Min, Y., Gomes, C.P. Unsupervised ordering for maximum clique. arXiv:2503.21814, 2025.
- Rzążewski, P. Polynomial-time MIS in bounded-degree graphs. SODA 2025.
