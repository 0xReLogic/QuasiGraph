#pragma once

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>

namespace QuasiGraph {

class Graph;

/**
 * Vertex Ordering Strategies
 * 
 * Based on Min & Gomes (2025) - learned ordering for improved
 * branch-and-bound performance
 */
enum class OrderingStrategy {
    DEGREE,              // Simple degree ordering
    DEGENERACY,          // Classical degeneracy ordering
    CLUSTERING,          // Clustering coefficient based
    EIGENVECTOR,         // Eigenvector centrality approximation
    LEARNED_HYBRID       // Hybrid heuristic (GNN-inspired)
};

/**
 * Vertex feature vector for ordering heuristics
 */
struct VertexFeatures {
    size_t vertex_id;
    size_t degree;
    double clustering_coefficient;
    double core_number;
    size_t triangle_count;
    double centrality_score;
    
    // Combined heuristic score
    double score;
};

/**
 * Vertex Ordering Optimizer
 * 
 * Computes optimal vertex orderings for branch-and-bound algorithms
 * Based on structural features and learned heuristics
 */
class VertexOrderingOptimizer {
public:
    VertexOrderingOptimizer() = default;
    
    /**
     * Compute optimal vertex ordering
     * @param graph Input graph
     * @param strategy Ordering strategy
     * @return Ordered list of vertices
     */
    std::vector<size_t> computeOrdering(const Graph& graph, 
                                         OrderingStrategy strategy = OrderingStrategy::LEARNED_HYBRID);
    
    /**
     * Get vertex features for all vertices
     * @param graph Input graph
     * @return Map of vertex ID to features
     */
    std::unordered_map<size_t, VertexFeatures> computeFeatures(const Graph& graph);
    
private:
    // Feature computation
    double computeClusteringCoefficient(const Graph& graph, size_t vertex);
    size_t computeTriangleCount(const Graph& graph, size_t vertex);
    double computeCoreNumber(const Graph& graph, size_t vertex);
    double computeCentralityScore(const Graph& graph, size_t vertex);
    
    // Ordering strategies
    std::vector<size_t> orderByDegree(const Graph& graph);
    std::vector<size_t> orderByDegeneracy(const Graph& graph);
    std::vector<size_t> orderByClustering(const Graph& graph);
    std::vector<size_t> orderByEigenvector(const Graph& graph);
    std::vector<size_t> orderByLearnedHybrid(const Graph& graph);
    
    // Hybrid scoring function (GNN-inspired)
    double computeHybridScore(const VertexFeatures& features, const Graph& graph);
    
    // Cache
    std::unordered_map<size_t, VertexFeatures> feature_cache_;
};

} // namespace QuasiGraph
