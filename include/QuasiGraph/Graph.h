#pragma once

/**
 * Graph Data Structure
 * 
 * Core graph representation for QuasiGraph algorithms.
 * Supports both directed and undirected graphs with efficient
 * operations for quasi-polynomial algorithms.
 */

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

namespace QuasiGraph {

class Graph {
public:
    /**
     * Constructor
     * @param directed Whether the graph is directed (default: false)
     */
    explicit Graph(bool directed = false);
    
    /**
     * Destructor
     */
    ~Graph() = default;
    
    /**
     * Add vertex to the graph
     * @param vertex_id Vertex identifier
     */
    void addVertex(size_t vertex_id);
    
    /**
     * Add edge between vertices
     * @param from Source vertex
     * @param to Target vertex
     * @param weight Edge weight (default: 1.0)
     */
    void addEdge(size_t from, size_t to, double weight = 1.0);
    
    /**
     * Check if edge exists
     * @param from Source vertex
     * @param to Target vertex
     * @return True if edge exists
     */
    bool hasEdge(size_t from, size_t to) const;
    
    /**
     * Get neighbors of a vertex
     * @param vertex_id Vertex identifier
     * @return Vector of neighbor vertex IDs
     */
    std::vector<size_t> getNeighbors(size_t vertex_id) const;
    
    /**
     * Get degree of a vertex
     * @param vertex_id Vertex identifier
     * @return Degree of the vertex
     */
    size_t getDegree(size_t vertex_id) const;
    
    /**
     * Get number of vertices
     * @return Vertex count
     */
    size_t getVertexCount() const;
    
    /**
     * Get number of edges
     * @return Edge count
     */
    size_t getEdgeCount() const;
    
    /**
     * Get average degree of all vertices
     * @return Average degree
     */
    double getAverageDegree() const;
    
    /**
     * Get graph density
     * @return Density (0.0 to 1.0)
     */
    double getDensity() const;
    
    /**
     * Clear all graph data
     */
    void clear();
    
    /**
     * Check if graph is empty
     * @return True if no vertices
     */
    bool isEmpty() const;

private:
    bool directed_;
    std::unordered_set<size_t> vertices_;
    std::unordered_map<size_t, std::unordered_map<size_t, double>> adjacency_list_;
    size_t edge_count_;
    
    void updateEdgeCount();
};

} // namespace QuasiGraph
