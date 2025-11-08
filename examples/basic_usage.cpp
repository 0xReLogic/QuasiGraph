/**
 * QuasiGraph Basic Usage Example
 * 
 * Simple demonstration of core functionality
 * for getting started with QuasiGraph.
 */

#include "QuasiGraph/Graph.h"
#include <iostream>

using namespace QuasiGraph;

int main() {
    std::cout << "=== QuasiGraph Basic Usage ===" << std::endl;
    
    // Create a simple graph
    Graph graph;
    
    // Add vertices
    graph.addVertex(0);
    graph.addVertex(1);
    graph.addVertex(2);
    graph.addVertex(3);
    
    // Add edges
    graph.addEdge(0, 1);
    graph.addEdge(1, 2);
    graph.addEdge(2, 3);
    graph.addEdge(3, 0);
    
    // Display graph properties
    std::cout << "Graph created:" << std::endl;
    std::cout << "- Vertices: " << graph.getVertexCount() << std::endl;
    std::cout << "- Edges: " << graph.getEdgeCount() << std::endl;
    std::cout << "- Average degree: " << graph.getAverageDegree() << std::endl;
    std::cout << "- Density: " << graph.getDensity() << std::endl;
    
    // Check connections
    std::cout << "\nConnections:" << std::endl;
    for (size_t i = 0; i < graph.getVertexCount(); ++i) {
        auto neighbors = graph.getNeighbors(i);
        std::cout << "Vertex " << i << " connects to: ";
        for (size_t neighbor : neighbors) {
            std::cout << neighbor << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\n✅ Basic usage example completed!" << std::endl;
    return 0;
}
