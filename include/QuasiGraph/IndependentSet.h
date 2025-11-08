#pragma once

/**
 * IndependentSet Problem Solver
 * 
 * Implementation of quasi-polynomial time algorithms for the
 * Independent Set Problem based on Lokshtanov's 2025 research.
 * 
 * Traditional complexity: O(2^n) - NP-complete
 * QuasiGraph complexity: O(n^(log n)) - Quasi-polynomial breakthrough
 */

#include "QuasiGraph/QuasiPolynomial.h"
#include <vector>
#include <unordered_set>
#include <chrono>
#include <memory>

namespace QuasiGraph {

struct Graph; // Forward declaration

struct IndependentSetResult {
    std::vector<size_t> independent_set;  // Vertex IDs in the independent set
    size_t set_size;                      // Size of the independent set
    double solution_quality;              // Quality measure (0.0 to 1.0)
    bool is_optimal;                      // True if provably optimal
    std::chrono::milliseconds computation_time;
    size_t nodes_explored;                // Number of graph nodes explored
    double approximation_ratio;           // Approximation ratio if not optimal
    std::string algorithm_used;           // Name of algorithm used
    bool success;
};

struct BranchAndBoundNode {
    std::vector<size_t> current_set;
    std::vector<size_t> candidates;
    size_t level;
    double upper_bound;
    double lower_bound;
    
    BranchAndBoundNode() : level(0), upper_bound(0.0), lower_bound(0.0) {}
    
    BranchAndBoundNode(const std::vector<size_t>& set, const std::vector<size_t>& cand, size_t lvl)
        : current_set(set), candidates(cand), level(lvl), upper_bound(0.0), lower_bound(0.0) {}
};

enum class IndependentSetAlgorithm {
    QUASI_POLYNOMIAL,    // Quasi-polynomial algorithm
    BRANCH_AND_BOUND,    // Enhanced branch and bound
    APPROXIMATION,       // Fast approximation algorithm
    HYBRID               // Combination of approaches
};

class IndependentSetSolver {
public:
    /**
     * Constructor
     * @param algorithm Choice of algorithm to use
     */
    explicit IndependentSetSolver(IndependentSetAlgorithm algorithm = IndependentSetAlgorithm::QUASI_POLYNOMIAL);
    
    /**
     * Destructor
     */
    ~IndependentSetSolver();
    
    /**
     * Find maximum independent set in the given graph
     * @param graph Input graph
     * @return Result containing the independent set and metadata
     */
    IndependentSetResult findMaximumIndependentSet(const Graph& graph);
    
    /**
     * Find independent set of specified size (decision version)
     * @param graph Input graph
     * @param target_size Target size to achieve
     * @return True if independent set of target size exists
     */
    bool hasIndependentSetOfSize(const Graph& graph, size_t target_size);
    
    /**
     * Get all maximal independent sets (not just maximum)
     * @param graph Input graph
     * @return Vector of all maximal independent sets
     */
    std::vector<std::vector<size_t>> findAllMaximalIndependentSets(const Graph& graph);
    
    /**
     * Set algorithm parameters
     * @param time_limit Maximum time allowed for computation
     * @param max_iterations Maximum iterations for iterative algorithms
     */
    void setParameters(std::chrono::milliseconds time_limit = std::chrono::milliseconds(60000),
                      size_t max_iterations = 1000000);
    
    /**
     * Enable parallel mode with specified thread count
     * @param num_threads Number of threads (0 = auto-detect)
     */
    void enableParallelMode(size_t num_threads = 0);
    
    /**
     * Disable parallel mode
     */
    void disableParallelMode();
    
    /**
     * Get performance statistics
     */
    struct PerformanceStats {
        size_t total_calls;
        std::chrono::milliseconds total_time;
        double average_solution_quality;
        size_t average_set_size;
        double success_rate;
    };
    
    PerformanceStats getPerformanceStats() const;
    
    /**
     * Reset performance statistics
     */
    void resetStats();

private:
    IndependentSetAlgorithm algorithm_;
    std::chrono::milliseconds time_limit_;
    size_t max_iterations_;
    
    // Parallel execution
    bool parallel_enabled_;
    size_t num_threads_;
    
    mutable PerformanceStats stats_;
    
    // Core algorithm implementations
    IndependentSetResult solveQuasiPolynomial(const Graph& graph);
    IndependentSetResult solveBranchAndBound(const Graph& graph);
    IndependentSetResult solveBranchAndBoundParallel(const Graph& graph);
    IndependentSetResult solveApproximation(const Graph& graph);
    IndependentSetResult solveHybrid(const Graph& graph);
    
    // Quasi-polynomial algorithm components
    IndependentSetResult applyQuasiDecomposition(const Graph& graph);
    IndependentSetResult solveQuasiPolynomialLarge(const Graph& graph);
    std::vector<GraphComponent> decomposeForIndependentSet(const Graph& graph);
    IndependentSetResult solveComponent(const GraphComponent& component, const Graph& original_graph);
    
    // Branch and bound enhancements
    void computeBounds(BranchAndBoundNode& node, const Graph& graph);
    std::vector<size_t> selectBranchingVariable(const BranchAndBoundNode& node, const Graph& graph);
    bool pruneByBounds(const BranchAndBoundNode& node, double current_best);
    
    // Approximation algorithms
    std::vector<size_t> greedyIndependentSet(const Graph& graph);
    std::vector<size_t> localSearchImprovement(const std::vector<size_t>& initial_set, const Graph& graph);
    
    // Utility functions
    bool isValidIndependentSet(const std::vector<size_t>& set, const Graph& graph);
    double calculateUpperBound(const Graph& graph, const std::vector<size_t>& candidates);
    std::vector<size_t> orderVerticesByDegree(const Graph& graph);
    std::vector<size_t> getComplement(const std::vector<size_t>& set, size_t graph_size);
    double calculateSolutionQuality(const Graph& graph, const std::vector<size_t>& set);
    double calculateApproximationRatio(const Graph& graph, const std::vector<size_t>& set);
    GraphComponent extractIndependentSetComponent(const Graph& graph, size_t start_vertex, std::vector<bool>& processed, size_t max_size);
    bool maintainsQuasiProperties(const std::vector<size_t>& vertices, size_t density_threshold, const Graph& graph);
    
    // Missing function declarations
    IndependentSetResult solveExactSmallGraph(const Graph& graph);
    Graph extractSubgraph(const Graph& original_graph, const std::vector<size_t>& vertices);
    void bronKerbosch(const std::vector<size_t>& current_set,
                     const std::vector<size_t>& candidates,
                     const std::vector<size_t>& excluded,
                     const Graph& graph,
                     std::vector<std::vector<size_t>>& all_maximal_sets);
    size_t choosePivot(const std::vector<size_t>& candidates,
                      const std::vector<size_t>& excluded,
                      const Graph& graph);
    
    // Performance optimization
    std::vector<std::vector<bool>> adjacency_cache_;
    std::vector<size_t> degree_cache_;
    bool cache_valid_;
    
    void updateCache(const Graph& graph);
    void clearCache();
    
    // Hybrid algorithm components
    IndependentSetResult combineResults(const std::vector<IndependentSetResult>& results);
    bool shouldSwitchAlgorithm(const std::chrono::milliseconds& elapsed_time, 
                              size_t iterations, double progress);
};

/**
 * Factory class for creating specialized independent set solvers
 */
class IndependentSetFactory {
public:
    /**
     * Create solver optimized for dense graphs
     */
    static std::unique_ptr<IndependentSetSolver> createDenseGraphSolver();
    
    /**
     * Create solver optimized for sparse graphs
     */
    static std::unique_ptr<IndependentSetSolver> createSparseGraphSolver();
    
    /**
     * Create solver for social network graphs
     */
    static std::unique_ptr<IndependentSetSolver> createSocialNetworkSolver();
    
    /**
     * Create solver for research applications (highest accuracy)
     */
    static std::unique_ptr<IndependentSetSolver> createResearchSolver();
};

} // namespace QuasiGraph
