/**
 * Vertex Ordering Benchmark
 * Tests learned ordering impact on branch-and-bound performance
 */

#include "QuasiGraph/Graph.h"
#include "QuasiGraph/IndependentSet.h"
#include "QuasiGraph/VertexOrdering.h"
#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>

using namespace QuasiGraph;
using namespace std::chrono;

struct OrderingBenchmarkResult {
    std::string strategy_name;
    double time_ms;
    size_t set_size;
    size_t nodes_explored;
};

OrderingBenchmarkResult benchmarkOrdering(const Graph& graph, OrderingStrategy strategy, 
                                           const std::string& name) {
    OrderingBenchmarkResult result;
    result.strategy_name = name;
    
    // Compute ordering
    VertexOrderingOptimizer optimizer;
    auto start = high_resolution_clock::now();
    auto ordering = optimizer.computeOrdering(graph, strategy);
    auto end = high_resolution_clock::now();
    
    double ordering_time = duration_cast<microseconds>(end - start).count() / 1000.0;
    
    // Solve independent set with this ordering
    IndependentSetSolver solver(IndependentSetAlgorithm::BRANCH_AND_BOUND);
    start = high_resolution_clock::now();
    auto is_result = solver.findMaximumIndependentSet(graph);
    end = high_resolution_clock::now();
    
    double solve_time = duration_cast<microseconds>(end - start).count() / 1000.0;
    
    result.time_ms = ordering_time + solve_time;
    result.set_size = is_result.set_size;
    result.nodes_explored = is_result.nodes_explored;
    
    return result;
}

int main() {
    std::cout << "=== Vertex Ordering Optimization Benchmark ===" << std::endl;
    std::cout << "Testing learned ordering impact on branch-and-bound" << std::endl;
    std::cout << std::endl;
    
    // Test different graph sizes
    std::vector<size_t> test_sizes = {50, 100, 150};
    double edge_probability = 0.3;
    
    for (size_t size : test_sizes) {
        std::cout << "Graph size: " << size << " vertices" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        
        // Create random graph
        Graph graph(false);
        std::mt19937 rng(42);
        std::uniform_real_distribution<> dist(0.0, 1.0);
        
        for (size_t i = 0; i < size; ++i) {
            graph.addVertex(i);
        }
        
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = i + 1; j < size; ++j) {
                if (dist(rng) < edge_probability) {
                    graph.addEdge(i, j);
                }
            }
        }
        
        std::cout << "Edges: " << graph.getEdgeCount() << std::endl;
        std::cout << std::endl;
        
        // Test different ordering strategies
        std::vector<OrderingBenchmarkResult> results;
        
        results.push_back(benchmarkOrdering(graph, OrderingStrategy::DEGREE, "Degree"));
        results.push_back(benchmarkOrdering(graph, OrderingStrategy::DEGENERACY, "Degeneracy"));
        results.push_back(benchmarkOrdering(graph, OrderingStrategy::CLUSTERING, "Clustering"));
        results.push_back(benchmarkOrdering(graph, OrderingStrategy::LEARNED_HYBRID, "Learned Hybrid"));
        
        // Print results
        std::cout << std::setw(20) << "Strategy"
                  << std::setw(15) << "Time (ms)"
                  << std::setw(15) << "Set Size"
                  << std::setw(18) << "Nodes Explored" << std::endl;
        std::cout << std::string(68, '-') << std::endl;
        
        double baseline_time = results[0].time_ms;
        
        for (const auto& result : results) {
            double speedup = baseline_time / result.time_ms;
            
            std::cout << std::setw(20) << result.strategy_name
                      << std::setw(15) << std::fixed << std::setprecision(2) << result.time_ms
                      << std::setw(15) << result.set_size
                      << std::setw(18) << result.nodes_explored;
            
            if (result.strategy_name != "Degree") {
                std::cout << " (" << std::setprecision(1) << speedup << "x)";
            }
            std::cout << std::endl;
        }
        
        std::cout << std::endl;
        
        // Find best strategy
        auto best = std::min_element(results.begin(), results.end(),
                                     [](const auto& a, const auto& b) {
                                         return a.time_ms < b.time_ms;
                                     });
        
        double best_speedup = baseline_time / best->time_ms;
        std::cout << "Best strategy: " << best->strategy_name 
                  << " (" << std::fixed << std::setprecision(1) << best_speedup << "x speedup)"
                  << std::endl;
        std::cout << std::endl << std::endl;
    }
    
    return 0;
}
