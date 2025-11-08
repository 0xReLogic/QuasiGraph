#include "QuasiGraph/Graph.h"
#include "QuasiGraph/IndependentSet.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <thread>

using namespace QuasiGraph;

void benchmarkParallelism(size_t graph_size) {
    std::cout << "\n=== Parallel Branch-and-Bound Benchmark ===" << std::endl;
    std::cout << "Graph size: " << graph_size << " vertices\n" << std::endl;
    
    // Generate random graph (sparser for faster solving)
    Graph graph(graph_size);
    std::srand(12345);
    size_t edge_count = graph_size * 2;  // Sparser for faster solving
    for (size_t i = 0; i < edge_count; ++i) {
        size_t v1 = std::rand() % graph_size;
        size_t v2 = std::rand() % graph_size;
        if (v1 != v2) {
            graph.addEdge(v1, v2);
        }
    }
    
    std::cout << "Graph: " << graph_size << " vertices, " << graph.getEdgeCount() << " edges" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << std::left << std::setw(12) << "Threads"
              << std::setw(15) << "Time (ms)"
              << std::setw(15) << "Speedup"
              << std::setw(15) << "Efficiency"
              << std::setw(13) << "Set Size" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    double baseline_time = 0.0;
    size_t baseline_size = 0;
    
    // Test different thread counts
    std::vector<size_t> thread_counts = {1, 2, 4, 8};
    
    for (size_t num_threads : thread_counts) {
        IndependentSetSolver solver(IndependentSetAlgorithm::BRANCH_AND_BOUND);
        
        if (num_threads == 1) {
            // Serial baseline
            solver.disableParallelMode();
        } else {
            // Parallel execution
            solver.enableParallelMode(num_threads);
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        auto result = solver.findMaximumIndependentSet(graph);
        auto end = std::chrono::high_resolution_clock::now();
        
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        if (num_threads == 1) {
            baseline_time = elapsed_ms;
            baseline_size = result.set_size;
        }
        
        double speedup = baseline_time / elapsed_ms;
        double efficiency = (speedup / num_threads) * 100.0;
        
        std::cout << std::left << std::setw(12) << num_threads
                  << std::setw(15) << std::fixed << std::setprecision(2) << elapsed_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << speedup
                  << std::setw(15) << std::fixed << std::setprecision(1) << efficiency << "%"
                  << std::setw(13) << result.set_size << std::endl;
        
        // Verify correctness
        if (result.set_size != baseline_size && num_threads > 1) {
            std::cout << "WARNING: Parallel result differs from serial!" << std::endl;
        }
    }
    
    std::cout << std::string(70, '-') << std::endl;
}

void scalabilityTest() {
    std::cout << "\n=== Scalability Analysis ===" << std::endl;
    std::cout << "Testing parallel speedup across graph sizes\n" << std::endl;
    
    std::vector<size_t> sizes = {30, 35, 40, 45};
    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 8;
    
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::left << std::setw(15) << "Graph Size"
              << std::setw(18) << "Serial (ms)"
              << std::setw(18) << "Parallel (ms)"
              << std::setw(15) << "Speedup"
              << std::setw(14) << "Set Size" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    for (size_t size : sizes) {
        // Generate graph
        Graph graph(size);
        std::srand(54321);
        size_t edge_count = size * 3;
        for (size_t i = 0; i < edge_count; ++i) {
            size_t v1 = std::rand() % size;
            size_t v2 = std::rand() % size;
            if (v1 != v2) {
                graph.addEdge(v1, v2);
            }
        }
        
        // Serial
        IndependentSetSolver serial_solver(IndependentSetAlgorithm::BRANCH_AND_BOUND);
        serial_solver.disableParallelMode();
        
        auto start = std::chrono::high_resolution_clock::now();
        auto serial_result = serial_solver.findMaximumIndependentSet(graph);
        auto end = std::chrono::high_resolution_clock::now();
        double serial_time = std::chrono::duration<double, std::milli>(end - start).count();
        
        // Parallel
        IndependentSetSolver parallel_solver(IndependentSetAlgorithm::BRANCH_AND_BOUND);
        parallel_solver.enableParallelMode(num_threads);
        
        start = std::chrono::high_resolution_clock::now();
        auto parallel_result = parallel_solver.findMaximumIndependentSet(graph);
        end = std::chrono::high_resolution_clock::now();
        double parallel_time = std::chrono::duration<double, std::milli>(end - start).count();
        
        double speedup = serial_time / parallel_time;
        
        std::cout << std::left << std::setw(15) << size
                  << std::setw(18) << std::fixed << std::setprecision(2) << serial_time
                  << std::setw(18) << std::fixed << std::setprecision(2) << parallel_time
                  << std::setw(15) << std::fixed << std::setprecision(2) << speedup
                  << std::setw(14) << parallel_result.set_size << std::endl;
    }
    
    std::cout << std::string(80, '-') << std::endl;
}

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════╗\n";
    std::cout << "║  QuasiGraph Parallel Optimization #3       ║\n";
    std::cout << "║  Branch-and-Bound with Work Stealing       ║\n";
    std::cout << "╚════════════════════════════════════════════╝\n";
    
    size_t hw_threads = std::thread::hardware_concurrency();
    std::cout << "\nHardware threads available: " << hw_threads << std::endl;
    
    // Benchmark parallelism on small graph
    benchmarkParallelism(40);
    
    // Scalability test
    scalabilityTest();
    
    std::cout << "\nBenchmark completed.\n" << std::endl;
    
    return 0;
}
