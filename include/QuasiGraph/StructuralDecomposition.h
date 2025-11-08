#pragma once

/**
 * Structural Graph Decomposition Framework
 * 
 * Advanced decomposition techniques based on Lokshtanov's 2025 research
 * for breaking complex graphs into quasi-polynomial solvable components.
 * 
 * This framework enables the transformation of NP-complete problems
 * into quasi-polynomial time solvable instances through structural analysis.
 */

#include <vector>
#include <memory>
#include <unordered_map>
#include <chrono>
#include <queue>

namespace QuasiGraph {

struct Graph; // Forward declaration

enum class DecompositionType {
    TREE_DECOMPOSITION,      // Tree-width based decomposition
    MODULAR_DECOMPOSITION,   // Module-based decomposition
    QUASI_POLYNOMIAL,       // Novel quasi-polynomial decomposition
    BALANCED_SEPARATOR,     // Balanced separator based
    DEGREE_BASED,           // Degree-based clustering
    HYBRID                  // Combination of techniques
};

struct GraphComponent {
    std::vector<size_t> vertices;           // Vertices in this component
    std::vector<std::pair<size_t, size_t>> internal_edges;  // Edges within component
    std::vector<size_t> boundary_vertices;  // Vertices connecting to other components
    double density;                         // Edge density of component
    size_t treewidth;                       // Tree width estimate
    bool is_quasi_solvable;                 // Can be solved in quasi-polynomial time
    std::string decomposition_method;      // Method used to create this component
};

struct DecompositionResult {
    std::vector<GraphComponent> components;     // All components
    DecompositionType type;                      // Type of decomposition used
    size_t total_components;                     // Number of components created
    double decomposition_quality;               // Quality measure (0.0 to 1.0)
    std::chrono::milliseconds decomposition_time;
    size_t vertices_processed;                   // Total vertices processed
    size_t edges_processed;                      // Total edges processed
    bool preserves_optimality;                   // True if optimal solution can be reconstructed
    double approximation_factor;                // Approximation factor if optimality not preserved
    std::vector<std::vector<size_t>> component_hierarchy;  // Hierarchy of components
};

struct SeparatorInfo {
    std::vector<size_t> separator_vertices;     // Separator vertices
    std::vector<size_t> left_component;         // Vertices on left side
    std::vector<size_t> right_component;        // Vertices on right side
    double separator_size;                      // Size of separator
    double balance_factor;                      // Balance between components (0.0 to 1.0)
    bool is_balanced;                           // True if reasonably balanced
};

struct TreewidthInfo {
    size_t treewidth;                           // Tree width of graph
    std::vector<std::vector<size_t>> bags;      // Tree decomposition bags
    std::vector<std::pair<size_t, size_t>> tree_edges;  // Edges in decomposition tree
    bool is_optimal;                           // True if treewidth is optimal
    std::chrono::milliseconds computation_time;
};

class StructuralDecomposition {
public:
    /**
     * Constructor
     * @param default_type Default decomposition type to use
     */
    explicit StructuralDecomposition(DecompositionType default_type = DecompositionType::QUASI_POLYNOMIAL);
    
    /**
     * Destructor
     */
    ~StructuralDecomposition();
    
    /**
     * Decompose graph using specified or default method
     * @param graph Input graph to decompose
     * @param type Type of decomposition (optional, uses default if not specified)
     * @return Decomposition result with all components and metadata
     */
    DecompositionResult decompose(const Graph& graph, 
                                 DecompositionType type = DecompositionType::AUTO);
    
    /**
     * Find balanced separator for graph splitting
     * @param graph Input graph
     * @param max_separator_size Maximum allowed separator size
     * @return Separator information
     */
    SeparatorInfo findBalancedSeparator(const Graph& graph, size_t max_separator_size = 0);
    
    /**
     * Compute tree width of graph
     * @param graph Input graph
     * @return Tree width information
     */
    TreewidthInfo computeTreewidth(const Graph& graph);
    
    /**
     * Check if graph is quasi-polynomial solvable
     * @param graph Input graph
     * @return True if graph can be solved in quasi-polynomial time
     */
    bool isQuasiPolynomialSolvable(const Graph& graph);
    
    /**
     * Get optimal decomposition type for given graph
     * @param graph Input graph
     * @return Recommended decomposition type
     */
    DecompositionType getOptimalDecompositionType(const Graph& graph);
    
    /**
     * Reconstruct optimal solution from component solutions
     * @param component_solutions Solutions for individual components
     * @param original_decomposition Original decomposition structure
     * @return Reconstructed global solution
     */
    std::vector<size_t> reconstructSolution(
        const std::vector<std::vector<size_t>>& component_solutions,
        const DecompositionResult& original_decomposition);
    
    /**
     * Set decomposition parameters
     * @param max_component_size Maximum size for components
     * @param quality_threshold Minimum quality threshold for components
     * @param time_limit Maximum time for decomposition
     */
    void setParameters(size_t max_component_size = 100,
                      double quality_threshold = 0.8,
                      std::chrono::milliseconds time_limit = std::chrono::milliseconds(30000));
    
    /**
     * Get decomposition statistics
     */
    struct DecompositionStats {
        size_t total_decompositions;
        std::chrono::milliseconds total_time;
        double average_quality;
        size_t average_components;
        double success_rate;
    };
    
    DecompositionStats getStats() const;
    
    /**
     * Reset statistics
     */
    void resetStats();

private:
    DecompositionType default_type_;
    size_t max_component_size_;
    double quality_threshold_;
    std::chrono::milliseconds time_limit_;
    
    mutable DecompositionStats stats_;
    
    // Core decomposition algorithms
    DecompositionResult treeDecomposition(const Graph& graph);
    DecompositionResult modularDecomposition(const Graph& graph);
    DecompositionResult quasiPolynomialDecomposition(const Graph& graph);
    DecompositionResult balancedSeparatorDecomposition(const Graph& graph);
    DecompositionResult degreeBasedDecomposition(const Graph& graph);
    DecompositionResult hybridDecomposition(const Graph& graph);
    
    // Quasi-polynomial decomposition components
    std::vector<GraphComponent> applyQuasiDecomposition(const Graph& graph);
    GraphComponent extractQuasiComponent(const Graph& graph, size_t start_vertex,
                                        std::vector<bool>& processed);
    bool maintainsQuasiProperties(const GraphComponent& component, size_t new_vertex,
                                 const Graph& graph);
    
    // Tree decomposition utilities
    std::vector<std::vector<size_t>> buildTreeDecomposition(const Graph& graph);
    size_t eliminateVertex(size_t vertex, std::vector<std::vector<size_t>>& bags,
                          const Graph& graph);
    
    // Separator finding algorithms
    SeparatorInfo findMinimumSeparator(const Graph& graph);
    SeparatorInfo findFlowBasedSeparator(const Graph& graph);
    SeparatorInfo findSpectralSeparator(const Graph& graph);
    
    // Modular decomposition
    std::vector<GraphComponent> findModules(const Graph& graph);
    bool isModule(const std::vector<size_t>& vertices, const Graph& graph);
    
    // Degree-based decomposition
    std::vector<GraphComponent> degreeBasedClustering(const Graph& graph);
    std::vector<size_t> getVerticesByDegree(const Graph& graph);
    
    // Quality assessment
    double assessComponentQuality(const GraphComponent& component, const Graph& graph);
    double calculateDecompositionQuality(const std::vector<GraphComponent>& components,
                                        const Graph& graph);
    
    // Utility functions
    bool isBalancedSeparator(const SeparatorInfo& separator);
    size_t estimateTreeWidth(const GraphComponent& component);
    double calculateDensity(const GraphComponent& component, const Graph& graph);
    std::vector<size_t> getBoundaryVertices(const GraphComponent& component, 
                                           const Graph& graph);
    
    // Graph analysis utilities
    std::vector<size_t> getConnectedComponent(const Graph& graph, size_t start_vertex);
    std::vector<std::vector<size_t>> getAllConnectedComponents(const Graph& graph);
    size_t calculateVertexConnectivity(const Graph& graph);
    double calculateAveragePathLength(const Graph& graph);
    
    // Optimization utilities
    void optimizeComponentSizes(std::vector<GraphComponent>& components);
    void mergeSmallComponents(std::vector<GraphComponent>& components);
    void splitLargeComponents(std::vector<GraphComponent>& components, 
                             const Graph& graph);
    
    // Caching for performance
    std::unordered_map<size_t, std::vector<size_t>> degree_cache_;
    std::unordered_map<size_t, std::vector<size_t>> neighbor_cache_;
    bool cache_valid_;
    
    void updateCache(const Graph& graph);
    void clearCache();
};

/**
 * Factory class for creating specialized decomposition engines
 */
class DecompositionFactory {
public:
    /**
     * Create decomposition engine for social networks
     */
    static std::unique_ptr<StructuralDecomposition> createSocialNetworkDecomposer();
    
    /**
     * Create decomposition engine for sparse graphs
     */
    static std::unique_ptr<StructuralDecomposition> createSparseGraphDecomposer();
    
    /**
     * Create decomposition engine for dense graphs
     */
    static std::unique_ptr<StructuralDecomposition> createDenseGraphDecomposer();
    
    /**
     * Create decomposition engine for research applications
     */
    static std::unique_ptr<StructuralDecomposition> createResearchDecomposer();
    
    /**
     * Create adaptive decomposition engine
     */
    static std::unique_ptr<StructuralDecomposition> createAdaptiveDecomposer();
};

} // namespace QuasiGraph
