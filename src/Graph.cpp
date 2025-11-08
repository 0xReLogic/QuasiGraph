/**
 * Graph Implementation
 * 
 * Core graph data structure for QuasiGraph algorithms.
 */

#include "QuasiGraph/Graph.h"
#include <stdexcept>

namespace QuasiGraph {

Graph::Graph(bool directed) 
    : directed_(directed), edge_count_(0) {
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

} // namespace QuasiGraph
