#pragma once

/**
 * Quasi-Polynomial Algorithm Implementation
 * 
 * Core implementation of Lokshtanov's 2025 breakthrough
 * quasi-polynomial time algorithms for graph optimization.
 */

#include "QuasiGraph/Graph.h"
#include <vector>
#include <chrono>

namespace QuasiGraph {

struct GraphComponent {
    std::vector<size_t> vertices;
    std::string decomposition_method;
    bool is_quasi_solvable;
    size_t treewidth;
    double density;
    std::vector<std::pair<size_t, size_t>> internal_edges;
    std::vector<size_t> boundary_vertices;
};

enum class OptimizationLevel {
    FAST,
    BALANCED,
    OPTIMAL
};

enum class DecompositionType {
    AUTO,
    DIRECT,
    QUASI_POLYNOMIAL
};

struct OptimizationResult {
    bool success;
    std::chrono::milliseconds time_used;
    std::chrono::milliseconds total_time;
    double objective_value;
    size_t iterations;
    std::string error_message;
    bool is_feasible;
    double optimality_gap;
    std::chrono::milliseconds decomposition_time;
    std::chrono::milliseconds optimization_time;
    double final_objective;
    size_t iterations_used;
};

struct DecompositionResult {
    bool success;
    std::chrono::milliseconds time_used;
    DecompositionType decomposition_type;
    size_t components_found;
    std::vector<GraphComponent> components;
    DecompositionType type;
    bool preserves_optimality;
    double approximation_factor;
    double decomposition_quality;
    size_t vertices_processed;
    size_t edges_processed;
    std::chrono::milliseconds decomposition_time;
    size_t total_components;
    std::vector<std::vector<size_t>> component_hierarchy;
};

struct ComplexityEstimate {
    double time_complexity;
    double space_complexity;
    bool is_quasi_polynomial;
};

class QuasiPolynomial {
public:
    QuasiPolynomial();
    ~QuasiPolynomial();
    
    OptimizationResult optimizeGraph(const Graph& graph);
    void setOptimizationLevel(OptimizationLevel level);
    void setTimeLimit(std::chrono::milliseconds limit);
    void setMaxIterations(size_t iterations);
    ComplexityEstimate estimateComplexity(size_t vertex_count, size_t edge_count);

private:
    OptimizationLevel optimization_level_;
    std::chrono::milliseconds time_limit_;
    size_t max_iterations_;
    double quasi_exponent_;
    size_t decomposition_threshold_;
    double approximation_factor_;
    std::unordered_map<size_t, double> complexity_bounds_;
    double previous_objective_;
    
    void initializeAlgorithmParameters();
    void precomputeComplexityBounds();
    DecompositionResult performStructuralDecomposition(const Graph& graph);
    std::vector<GraphComponent> decomposeIntoComponents(const Graph& graph);
    GraphComponent extractComponent(const Graph& graph, size_t start_vertex, 
                                   std::vector<bool>& visited, size_t max_size);
    OptimizationResult applyQuasiPolynomialOptimization(const Graph& graph, 
                                                       const DecompositionResult& decomposition);
    OptimizationResult fastOptimization(const Graph& graph, const DecompositionResult& decomposition);
    OptimizationResult balancedOptimization(const Graph& graph, const DecompositionResult& decomposition);
    OptimizationResult optimalOptimization(const Graph& graph, const DecompositionResult& decomposition);
    double fastQuasiUpdate(const Graph& graph, size_t iteration);
    double balancedQuasiUpdate(const Graph& graph, size_t iteration);
    double optimalQuasiUpdate(const Graph& graph, size_t iteration);
    bool hasConverged(double current_objective, size_t iteration);
    void validateResult(const Graph& graph, OptimizationResult& result);
    bool checkFeasibility(const Graph& graph, const OptimizationResult& result);
    double calculateOptimalityGap(const Graph& graph, const OptimizationResult& result);
    bool maintainsQuasiProperties(const std::vector<size_t>& component, size_t new_vertex, const Graph& graph);
};

} // namespace QuasiGraph
