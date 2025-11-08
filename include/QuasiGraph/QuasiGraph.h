#pragma once

/**
 * QuasiGraph Library Header
 * 
 * Complete header file including all QuasiGraph components
 */

#include "Graph.h"
#include "QuasiPolynomial.h"
#include "IndependentSet.h"
#include "StructuralDecomposition.h"
#include "SocialNetworkAnalysis.h"
#include "StructuralDecomposition.h"
#include "SocialNetworkAnalysis.h"

#include <vector>
#include <memory>
#include <chrono>

namespace QuasiGraph {

/**
 * Main QuasiGraph class that orchestrates all optimization algorithms
 */
class QuasiGraphEngine {
public:
    /**
     * Constructor with optional graph size
     * @param vertices Initial number of vertices
     */
    explicit QuasiGraphEngine(size_t vertices = 0);
    
    /**
     * Destructor
     */
    ~QuasiGraphEngine();
    
    /**
     * Add vertex to the graph
     * @param id Vertex identifier
     */
    void addVertex(size_t id);
    
    /**
     * Add edge between vertices
     * @param from Source vertex
     * @param to Target vertex
     * @param weight Edge weight (default: 1.0)
     */
    void addEdge(size_t from, size_t to, double weight = 1.0);
    
    /**
     * Apply quasi-polynomial optimization to the graph
     * @return Optimization success status
     */
    bool optimizeQuasiPolynomial();
    
    /**
     * Find maximum independent set using quasi-polynomial algorithm
     * @return Vector of vertex IDs in the independent set
     */
    std::vector<size_t> findMaximumIndependentSet();
    
    /**
     * Perform structural graph decomposition
     * @return Decomposition result
     */
    StructuralDecomposition decomposeGraph();
    
    /**
     * Analyze social network properties
     * @return Social network analysis results
     */
    NetworkMetrics analyzeSocialNetwork();
    
    /**
     * Get performance metrics
     * @return Performance statistics
     */
    ComplexityEstimate getPerformanceMetrics() const;
    
    /**
     * Clear all graph data
     */
    void clear();
    
    /**
     * Get current graph size
     * @return Number of vertices
     */
    size_t getVertexCount() const;
    
    /**
     * Get current edge count
     * @return Number of edges
     */
    size_t getEdgeCount() const;

private:
    std::unique_ptr<Graph> graph_;
    std::unique_ptr<QuasiPolynomial> quasi_optimizer_;
    std::unique_ptr<IndependentSetSolver> independent_set_solver_;
    std::unique_ptr<StructuralDecomposition> decomposition_engine_;
    
    mutable ComplexityEstimate performance_;
    
    void initializeComponents();
    void updatePerformanceMetrics(const std::string& operation, 
                                 std::chrono::milliseconds duration);
};

/**
 * Factory class for creating specialized graph optimizers
 */
class QuasiGraphFactory {
public:
    /**
     * Create optimizer for social networks
     */
    static std::unique_ptr<QuasiGraphEngine> createSocialNetworkOptimizer();
    
    /**
     * Create optimizer for large-scale graphs
     */
    static std::unique_ptr<QuasiGraphEngine> createLargeScaleOptimizer();
    
    /**
     * Create optimizer for research applications
     */
    static std::unique_ptr<QuasiGraphEngine> createResearchOptimizer();
};

} // namespace QuasiGraph
