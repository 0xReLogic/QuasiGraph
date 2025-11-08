/**
 * Vertex Ordering Implementation
 * 
 * Optimized vertex ordering for branch-and-bound algorithms
 */

#include "QuasiGraph/VertexOrdering.h"
#include "QuasiGraph/Graph.h"
#include <queue>
#include <set>
#include <cmath>
#include <algorithm>

namespace QuasiGraph {

std::vector<size_t> VertexOrderingOptimizer::computeOrdering(
    const Graph& graph, OrderingStrategy strategy) {
    
    switch (strategy) {
        case OrderingStrategy::DEGREE:
            return orderByDegree(graph);
        case OrderingStrategy::DEGENERACY:
            return orderByDegeneracy(graph);
        case OrderingStrategy::CLUSTERING:
            return orderByClustering(graph);
        case OrderingStrategy::EIGENVECTOR:
            return orderByEigenvector(graph);
        case OrderingStrategy::LEARNED_HYBRID:
        default:
            return orderByLearnedHybrid(graph);
    }
}

std::unordered_map<size_t, VertexFeatures> 
VertexOrderingOptimizer::computeFeatures(const Graph& graph) {
    
    if (!feature_cache_.empty()) {
        return feature_cache_;
    }
    
    std::unordered_map<size_t, VertexFeatures> features;
    
    size_t vertex_count = graph.getVertexCount();
    if (vertex_count == 0) return features;
    
    // Compute features for each vertex
    for (size_t v = 0; v < vertex_count; ++v) {
        VertexFeatures vf;
        vf.vertex_id = v;
        vf.degree = graph.getDegree(v);
        vf.clustering_coefficient = computeClusteringCoefficient(graph, v);
        vf.triangle_count = computeTriangleCount(graph, v);
        vf.core_number = computeCoreNumber(graph, v);
        vf.centrality_score = computeCentralityScore(graph, v);
        vf.score = 0.0;
        
        features[v] = vf;
    }
    
    feature_cache_ = features;
    return features;
}

double VertexOrderingOptimizer::computeClusteringCoefficient(
    const Graph& graph, size_t vertex) {
    
    auto neighbors = graph.getNeighbors(vertex);
    size_t degree = neighbors.size();
    
    if (degree < 2) return 0.0;
    
    // Count triangles
    size_t triangle_edges = 0;
    for (size_t i = 0; i < neighbors.size(); ++i) {
        for (size_t j = i + 1; j < neighbors.size(); ++j) {
            if (graph.hasEdge(neighbors[i], neighbors[j])) {
                triangle_edges++;
            }
        }
    }
    
    size_t max_edges = (degree * (degree - 1)) / 2;
    return max_edges > 0 ? static_cast<double>(triangle_edges) / max_edges : 0.0;
}

size_t VertexOrderingOptimizer::computeTriangleCount(
    const Graph& graph, size_t vertex) {
    
    auto neighbors = graph.getNeighbors(vertex);
    size_t triangles = 0;
    
    for (size_t i = 0; i < neighbors.size(); ++i) {
        for (size_t j = i + 1; j < neighbors.size(); ++j) {
            if (graph.hasEdge(neighbors[i], neighbors[j])) {
                triangles++;
            }
        }
    }
    
    return triangles;
}

double VertexOrderingOptimizer::computeCoreNumber(
    const Graph& graph, size_t vertex) {
    
    // Approximation: return degree as core number estimate
    return static_cast<double>(graph.getDegree(vertex));
}

double VertexOrderingOptimizer::computeCentralityScore(
    const Graph& graph, size_t vertex) {
    
    // Simple centrality: degree + neighbor degree sum
    auto neighbors = graph.getNeighbors(vertex);
    double score = static_cast<double>(neighbors.size());
    
    for (size_t neighbor : neighbors) {
        score += 0.1 * graph.getDegree(neighbor);
    }
    
    return score;
}

std::vector<size_t> VertexOrderingOptimizer::orderByDegree(const Graph& graph) {
    std::vector<std::pair<size_t, size_t>> vertices; // (degree, vertex_id)
    
    size_t vertex_count = graph.getVertexCount();
    for (size_t v = 0; v < vertex_count; ++v) {
        vertices.push_back({graph.getDegree(v), v});
    }
    
    // Sort by degree descending
    std::sort(vertices.begin(), vertices.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    std::vector<size_t> ordering;
    for (const auto& [deg, v] : vertices) {
        ordering.push_back(v);
    }
    
    return ordering;
}

std::vector<size_t> VertexOrderingOptimizer::orderByDegeneracy(const Graph& graph) {
    size_t vertex_count = graph.getVertexCount();
    std::vector<size_t> ordering;
    std::set<size_t> remaining;
    std::vector<size_t> degrees(vertex_count);
    
    // Initialize
    for (size_t v = 0; v < vertex_count; ++v) {
        remaining.insert(v);
        degrees[v] = graph.getDegree(v);
    }
    
    // Degeneracy ordering: repeatedly remove vertex with minimum degree
    while (!remaining.empty()) {
        // Find vertex with minimum degree
        size_t min_vertex = *remaining.begin();
        size_t min_degree = degrees[min_vertex];
        
        for (size_t v : remaining) {
            if (degrees[v] < min_degree) {
                min_degree = degrees[v];
                min_vertex = v;
            }
        }
        
        ordering.push_back(min_vertex);
        remaining.erase(min_vertex);
        
        // Update degrees of neighbors
        auto neighbors = graph.getNeighbors(min_vertex);
        for (size_t neighbor : neighbors) {
            if (remaining.count(neighbor) && degrees[neighbor] > 0) {
                degrees[neighbor]--;
            }
        }
    }
    
    // Reverse for better branch-and-bound performance
    std::reverse(ordering.begin(), ordering.end());
    return ordering;
}

std::vector<size_t> VertexOrderingOptimizer::orderByClustering(const Graph& graph) {
    auto features = computeFeatures(graph);
    
    std::vector<std::pair<double, size_t>> vertices; // (clustering, vertex_id)
    for (const auto& [v, vf] : features) {
        vertices.push_back({vf.clustering_coefficient, v});
    }
    
    // Sort by clustering descending
    std::sort(vertices.begin(), vertices.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    std::vector<size_t> ordering;
    for (const auto& [cc, v] : vertices) {
        ordering.push_back(v);
    }
    
    return ordering;
}

std::vector<size_t> VertexOrderingOptimizer::orderByEigenvector(const Graph& graph) {
    // Simplified eigenvector centrality approximation
    // Use degree + neighbor degrees as proxy
    auto features = computeFeatures(graph);
    
    std::vector<std::pair<double, size_t>> vertices;
    for (const auto& [v, vf] : features) {
        vertices.push_back({vf.centrality_score, v});
    }
    
    std::sort(vertices.begin(), vertices.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    std::vector<size_t> ordering;
    for (const auto& [score, v] : vertices) {
        ordering.push_back(v);
    }
    
    return ordering;
}

std::vector<size_t> VertexOrderingOptimizer::orderByLearnedHybrid(const Graph& graph) {
    // Hybrid heuristic inspired by GNN features
    // Combines multiple structural features with learned weights
    auto features = computeFeatures(graph);
    
    // Compute hybrid scores
    std::vector<std::pair<double, size_t>> vertices;
    for (auto& [v, vf] : features) {
        vf.score = computeHybridScore(vf, graph);
        vertices.push_back({vf.score, v});
    }
    
    // Sort by score descending
    std::sort(vertices.begin(), vertices.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    std::vector<size_t> ordering;
    for (const auto& [score, v] : vertices) {
        ordering.push_back(v);
    }
    
    return ordering;
}

double VertexOrderingOptimizer::computeHybridScore(
    const VertexFeatures& features, const Graph& /* graph */) {
    
    // Learned weights from Min & Gomes paper (approximated)
    const double w_degree = 0.35;
    const double w_clustering = 0.25;
    const double w_triangles = 0.20;
    const double w_centrality = 0.20;
    
    // Normalize features
    double norm_degree = std::log(1.0 + features.degree);
    double norm_clustering = features.clustering_coefficient;
    double norm_triangles = std::log(1.0 + features.triangle_count);
    double norm_centrality = std::log(1.0 + features.centrality_score);
    
    // Weighted combination
    double score = w_degree * norm_degree +
                   w_clustering * norm_clustering +
                   w_triangles * norm_triangles +
                   w_centrality * norm_centrality;
    
    return score;
}

} // namespace QuasiGraph
