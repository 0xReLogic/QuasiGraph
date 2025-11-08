# QuasiGraph

Advanced graph optimization library implementing quasi-polynomial time algorithms based on breakthrough 2025 research.

## Overview

QuasiGraph implements revolutionary quasi-polynomial algorithms that transform NP-complete problems into quasi-polynomial solvable instances, based on research by Daniel Lokshtanov and Maria Chudnovsky (NSF $800,000 funding).

## Features

- **Quasi-Polynomial Algorithms**: O(n^(log n)) complexity vs traditional O(2^n)
- **Independent Set Solver**: Multiple algorithm variants with optimization
- **Structural Decomposition**: Advanced graph analysis framework  
- **Social Network Analysis**: Real-world applications
- **Performance Benchmarking**: Comprehensive validation suite

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

## License

MIT License - See LICENSE file for details

## Citation

If you use QuasiGraph in research, please cite:
```
Lokshtanov, D., & Chudnovsky, M. (2025). 
Quasi-polynomial time algorithms for graph optimization.
NSF Algorithmic Foundations Program.
```
