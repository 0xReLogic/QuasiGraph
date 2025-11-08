/**
 * Bit-Parallel Optimization Benchmark
 * Tests SIMD AVX2 optimizations vs standard implementation
 */

#include "QuasiGraph/Graph.h"
#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>

using namespace QuasiGraph;
using namespace std::chrono;

struct BenchmarkResult {
    size_t vertices;
    size_t edges;
    double standard_time_us;
    double bitparallel_time_us;
    double speedup;
    size_t test_operations;
};

BenchmarkResult benchmarkCommonNeighbors(size_t num_vertices, double edge_probability) {
    BenchmarkResult result;
    result.vertices = num_vertices;
    result.test_operations = std::min(size_t(1000), num_vertices * num_vertices / 100);
    
    // Create random graph
    Graph graph(false);
    std::mt19937 rng(42);
    std::uniform_real_distribution<> dist(0.0, 1.0);
    
    // Add vertices
    for (size_t i = 0; i < num_vertices; ++i) {
        graph.addVertex(i);
    }
    
    // Add random edges
    size_t edge_count = 0;
    for (size_t i = 0; i < num_vertices; ++i) {
        for (size_t j = i + 1; j < num_vertices; ++j) {
            if (dist(rng) < edge_probability) {
                graph.addEdge(i, j);
                edge_count++;
            }
        }
    }
    result.edges = edge_count;
    
    // Generate test pairs
    std::vector<std::pair<size_t, size_t>> test_pairs;
    std::uniform_int_distribution<size_t> vertex_dist(0, num_vertices - 1);
    for (size_t i = 0; i < result.test_operations; ++i) {
        test_pairs.push_back({vertex_dist(rng), vertex_dist(rng)});
    }
    
    // Benchmark STANDARD mode
    auto start = high_resolution_clock::now();
    size_t standard_total = 0;
    for (const auto& [v1, v2] : test_pairs) {
        standard_total += graph.getCommonNeighborCount(v1, v2);
    }
    auto end = high_resolution_clock::now();
    result.standard_time_us = duration_cast<microseconds>(end - start).count();
    
    // Enable bit-parallel mode
    graph.enableBitParallelMode();
    
    // Benchmark BIT-PARALLEL mode (SIMD AVX2)
    start = high_resolution_clock::now();
    size_t bitparallel_total = 0;
    for (const auto& [v1, v2] : test_pairs) {
        bitparallel_total += graph.getCommonNeighborCount(v1, v2);
    }
    end = high_resolution_clock::now();
    result.bitparallel_time_us = duration_cast<microseconds>(end - start).count();
    
    // Verify correctness
    if (standard_total != bitparallel_total) {
        std::cerr << "ERROR: Results don't match! Standard: " << standard_total 
                  << ", BitParallel: " << bitparallel_total << std::endl;
    }
    
    result.speedup = result.standard_time_us / result.bitparallel_time_us;
    
    return result;
}

int main() {
    std::cout << "=== Bit-Parallel SIMD AVX2 Optimization Benchmark ===" << std::endl;
    std::cout << std::endl;
    
    std::vector<BenchmarkResult> results;
    
    // Test different graph sizes and densities
    struct TestCase { size_t vertices; double density; };
    std::vector<TestCase> test_cases = {
        {100, 0.3},
        {200, 0.3},
        {500, 0.3},
        {1000, 0.3},
        {2000, 0.3},
        {1000, 0.5},  // denser
        {1000, 0.7},  // very dense
    };
    
    std::cout << std::setw(10) << "Vertices" 
              << std::setw(10) << "Edges"
              << std::setw(15) << "Standard (μs)"
              << std::setw(18) << "BitParallel (μs)"
              << std::setw(12) << "Speedup"
              << std::setw(12) << "Operations" << std::endl;
    std::cout << std::string(77, '-') << std::endl;
    
    for (const auto& tc : test_cases) {
        auto result = benchmarkCommonNeighbors(tc.vertices, tc.density);
        results.push_back(result);
        
        std::cout << std::setw(10) << result.vertices
                  << std::setw(10) << result.edges
                  << std::setw(15) << std::fixed << std::setprecision(1) << result.standard_time_us
                  << std::setw(18) << result.bitparallel_time_us
                  << std::setw(11) << std::setprecision(2) << result.speedup << "x"
                  << std::setw(12) << result.test_operations << std::endl;
    }
    
    // Calculate average speedup
    double avg_speedup = 0.0;
    for (const auto& r : results) {
        avg_speedup += r.speedup;
    }
    avg_speedup /= results.size();
    
    std::cout << std::string(77, '-') << std::endl;
    std::cout << "Average speedup: " << std::fixed << std::setprecision(2) 
              << avg_speedup << "x" << std::endl;
    std::cout << std::endl;
    
    if (avg_speedup >= 2.0) {
        std::cout << "Achieved 2x+ speedup target" << std::endl;
    } else if (avg_speedup >= 1.5) {
        std::cout << "Significant speedup achieved" << std::endl;
    } else {
        std::cout << "Speedup below expected (may need AVX2 support)" << std::endl;
    }
    
    return 0;
}
