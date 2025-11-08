#pragma once

/**
 * Graph Data Structure
 * 
 * Core graph representation with support for both directed and
 * undirected graphs. Includes bit-parallel SIMD optimizations.
 */

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <string>
#include "BitSet.h"

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
    
    /**
     * Enable bit-parallel mode for SIMD optimizations
     * Note: Only for unweighted graphs
     */
    void enableBitParallelMode();
    
    /**
     * Get common neighbor count (bit-parallel optimized)
     * @param v1 First vertex
     * @param v2 Second vertex
     * @return Number of common neighbors
     */
    size_t getCommonNeighborCount(size_t v1, size_t v2) const;
    
    /**
     * Check if using bit-parallel mode
     */
    bool isBitParallelMode() const { return use_bitset_; }
    
    /**
     * Find maximum independent set using optimized algorithms
     * @param use_parallel Enable parallel branch-and-bound (default: false)
     * @param num_threads Number of threads (0 = auto-detect)
     * @return Vertices in maximum independent set
     */
    std::vector<size_t> findMaximumIndependentSet(bool use_parallel = false, size_t num_threads = 0) const;
    
    /**
     * Get vertex ordering for better algorithm performance
     * @param strategy Ordering strategy (degree, degeneracy, clustering, etc.)
     * @return Ordered vertex list
     */
    std::vector<size_t> getVertexOrdering(const std::string& strategy = "degeneracy") const;
    
    /**
     * Compute clustering coefficient for a vertex
     * @param vertex_id Vertex to compute for
     * @return Clustering coefficient (0.0 to 1.0)
     */
    double getClusteringCoefficient(size_t vertex_id) const;

private:
    bool directed_;
    std::unordered_set<size_t> vertices_;
    std::unordered_map<size_t, std::unordered_map<size_t, double>> adjacency_list_;
    size_t edge_count_;
    
    // Bit-parallel adjacency representation
    bool use_bitset_;
    size_t max_vertex_id_;
    std::vector<BitSet> bitset_adjacency_;
    std::unordered_map<size_t, size_t> vertex_to_index_;
    std::vector<size_t> index_to_vertex_;
    
    void updateEdgeCount();
    void buildBitSetRepresentation();
};

} // namespace QuasiGraph
