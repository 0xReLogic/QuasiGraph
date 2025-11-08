/**
 * Graph Implementation
 * 
 * Core graph data structure with bit-parallel optimizations
 */

#include "QuasiGraph/Graph.h"
#include "QuasiGraph/IndependentSet.h"
#include "QuasiGraph/VertexOrdering.h"
#include <stdexcept>
#include <algorithm>

namespace QuasiGraph {

Graph::Graph(bool directed) 
    : directed_(directed), edge_count_(0), use_bitset_(false), max_vertex_id_(0) {
}

void Graph::addVertex(size_t vertex_id) {
    vertices_.insert(vertex_id);
    
    // Ensure adjacency list entry exists
    if (adjacency_list_.find(vertex_id) == adjacency_list_.end()) {
        adjacency_list_[vertex_id] = std::unordered_map<size_t, double>();
    }
}

void Graph::addEdge(size_t from, size_t to, double weight) {
    // Add vertices if they don't exist
    addVertex(from);
    addVertex(to);
    
    // Add edge
    adjacency_list_[from][to] = weight;
    
    // For undirected graphs, add reverse edge
    if (!directed_) {
        adjacency_list_[to][from] = weight;
    }
    
    updateEdgeCount();
}

bool Graph::hasEdge(size_t from, size_t to) const {
    auto from_it = adjacency_list_.find(from);
    if (from_it == adjacency_list_.end()) {
        return false;
    }
    
    auto to_it = from_it->second.find(to);
    return to_it != from_it->second.end();
}

std::vector<size_t> Graph::getNeighbors(size_t vertex_id) const {
    std::vector<size_t> neighbors;
    
    auto it = adjacency_list_.find(vertex_id);
    if (it != adjacency_list_.end()) {
        for (const auto& [neighbor, weight] : it->second) {
            neighbors.push_back(neighbor);
        }
    }
    
    return neighbors;
}

size_t Graph::getDegree(size_t vertex_id) const {
    auto it = adjacency_list_.find(vertex_id);
    if (it != adjacency_list_.end()) {
        return it->second.size();
    }
    return 0;
}

size_t Graph::getVertexCount() const {
    return vertices_.size();
}

size_t Graph::getEdgeCount() const {
    return edge_count_;
}

double Graph::getAverageDegree() const {
    if (vertices_.empty()) return 0.0;
    
    double total_degree = 0.0;
    for (size_t vertex : vertices_) {
        total_degree += getDegree(vertex);
    }
    
    return total_degree / vertices_.size();
}

double Graph::getDensity() const {
    size_t vertex_count = getVertexCount();
    if (vertex_count < 2) return 0.0;
    
    size_t max_possible_edges = directed_ ? 
                               vertex_count * (vertex_count - 1) :
                               vertex_count * (vertex_count - 1) / 2;
    
    return max_possible_edges > 0 ? 
           static_cast<double>(edge_count_) / max_possible_edges : 0.0;
}

void Graph::clear() {
    vertices_.clear();
    adjacency_list_.clear();
    edge_count_ = 0;
}

bool Graph::isEmpty() const {
    return vertices_.empty();
}

void Graph::updateEdgeCount() {
    edge_count_ = 0;
    for (const auto& [vertex, neighbors] : adjacency_list_) {
        edge_count_ += neighbors.size();
    }
    
    // For undirected graphs, each edge is counted twice
    if (!directed_) {
        edge_count_ /= 2;
    }
}

// Bit-parallel optimization functions
void Graph::enableBitParallelMode() {
    if (use_bitset_) return; // Already enabled
    
    // Find max vertex ID
    max_vertex_id_ = 0;
    for (size_t v : vertices_) {
        max_vertex_id_ = std::max(max_vertex_id_, v);
    }
    
    // Build vertex mapping
    vertex_to_index_.clear();
    index_to_vertex_.clear();
    index_to_vertex_.reserve(vertices_.size());
    
    size_t index = 0;
    for (size_t v : vertices_) {
        vertex_to_index_[v] = index;
        index_to_vertex_.push_back(v);
        index++;
    }
    
    // Initialize bitsets
    bitset_adjacency_.clear();
    bitset_adjacency_.resize(vertices_.size(), BitSet(vertices_.size()));
    
    // Build bitset adjacency
    buildBitSetRepresentation();
    
    use_bitset_ = true;
}

void Graph::buildBitSetRepresentation() {
    for (size_t i = 0; i < index_to_vertex_.size(); ++i) {
        size_t vertex = index_to_vertex_[i];
        
        if (adjacency_list_.find(vertex) != adjacency_list_.end()) {
            for (const auto& [neighbor, weight] : adjacency_list_[vertex]) {
                if (vertex_to_index_.find(neighbor) != vertex_to_index_.end()) {
                    size_t neighbor_index = vertex_to_index_[neighbor];
                    bitset_adjacency_[i].set(neighbor_index);
                }
            }
        }
    }
}

size_t Graph::getCommonNeighborCount(size_t v1, size_t v2) const {
    if (!use_bitset_) {
        // Fallback to standard implementation
        auto neighbors1 = getNeighbors(v1);
        auto neighbors2 = getNeighbors(v2);
        
        std::unordered_set<size_t> set1(neighbors1.begin(), neighbors1.end());
        size_t count = 0;
        for (size_t n : neighbors2) {
            if (set1.count(n)) count++;
        }
        return count;
    }
    
    // Bit-parallel SIMD optimized path
    if (vertex_to_index_.find(v1) == vertex_to_index_.end() ||
        vertex_to_index_.find(v2) == vertex_to_index_.end()) {
        return 0;
    }
    
    size_t idx1 = vertex_to_index_.at(v1);
    size_t idx2 = vertex_to_index_.at(v2);
    
    // Hardware POPCNT + SIMD intersection
    return bitset_adjacency_[idx1].intersect_count(bitset_adjacency_[idx2]);
}

std::vector<size_t> Graph::findMaximumIndependentSet(bool use_parallel, size_t num_threads) const {
    IndependentSetSolver solver(IndependentSetAlgorithm::BRANCH_AND_BOUND);
    
    if (use_parallel) {
        solver.enableParallelMode(num_threads);
    }
    
    auto result = solver.findMaximumIndependentSet(*this);
    return result.independent_set;
}

std::vector<size_t> Graph::getVertexOrdering(const std::string& strategy) const {
    QuasiGraph::OrderingStrategy strat;
    if (strategy == "degree") {
        strat = QuasiGraph::OrderingStrategy::DEGREE;
    } else if (strategy == "degeneracy") {
        strat = QuasiGraph::OrderingStrategy::DEGENERACY;
    } else if (strategy == "clustering") {
        strat = QuasiGraph::OrderingStrategy::CLUSTERING;
    } else if (strategy == "eigenvector") {
        strat = QuasiGraph::OrderingStrategy::EIGENVECTOR;
    } else if (strategy == "learned") {
        strat = QuasiGraph::OrderingStrategy::LEARNED_HYBRID;
    } else {
        strat = QuasiGraph::OrderingStrategy::DEGENERACY; // default
    }
    
    QuasiGraph::VertexOrderingOptimizer optimizer;
    return optimizer.computeOrdering(*this, strat);
}

double Graph::getClusteringCoefficient(size_t vertex_id) const {
    auto neighbors = getNeighbors(vertex_id);
    size_t degree = neighbors.size();
    
    if (degree < 2) return 0.0;
    
    // Count triangles (edges between neighbors)
    size_t triangles = 0;
    for (size_t i = 0; i < neighbors.size(); ++i) {
        for (size_t j = i + 1; j < neighbors.size(); ++j) {
            if (hasEdge(neighbors[i], neighbors[j])) {
                triangles++;
            }
        }
    }
    
    // Clustering coefficient = (2 * triangles) / (degree * (degree - 1))
    size_t max_triangles = degree * (degree - 1) / 2;
    return max_triangles > 0 ? static_cast<double>(triangles) / max_triangles : 0.0;
}

} // namespace QuasiGraph
