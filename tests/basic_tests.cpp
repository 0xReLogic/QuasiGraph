/**
 * QuasiGraph Basic Tests
 * 
 * Core functionality tests without external dependencies.
 */

#include "QuasiGraph/Graph.h"
#include <iostream>
#include <cassert>

using namespace QuasiGraph;

bool testGraphCreation() {
    std::cout << "Testing graph creation... ";
    
    Graph graph;
    assert(graph.isEmpty());
    assert(graph.getVertexCount() == 0);
    
    graph.addVertex(0);
    graph.addVertex(1);
    assert(graph.getVertexCount() == 2);
    assert(!graph.isEmpty());
    
    std::cout << "PASSED" << std::endl;
    return true;
}

bool testEdgeOperations() {
    std::cout << "Testing edge operations... ";
    
    Graph graph;
    graph.addVertex(0);
    graph.addVertex(1);
    graph.addVertex(2);
    
    graph.addEdge(0, 1);
    graph.addEdge(1, 2);
    
    assert(graph.getEdgeCount() == 2);
    assert(graph.hasEdge(0, 1));
    assert(graph.hasEdge(1, 2));
    assert(!graph.hasEdge(0, 2));
    
    std::cout << "PASSED" << std::endl;
    return true;
}

bool testGraphProperties() {
    std::cout << "Testing graph properties... ";
    
    Graph graph;
    for (size_t i = 0; i < 4; ++i) {
        graph.addVertex(i);
    }
    
    graph.addEdge(0, 1);
    graph.addEdge(1, 2);
    graph.addEdge(2, 3);
    
    assert(graph.getAverageDegree() > 0.0);
    assert(graph.getDensity() > 0.0);
    
    auto neighbors = graph.getNeighbors(1);
    assert(neighbors.size() == 2);
    
    std::cout << "PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "=== QuasiGraph Basic Tests ===" << std::endl;
    
    int passed = 0;
    int total = 3;
    
    try {
        if (testGraphCreation()) passed++;
        if (testEdgeOperations()) passed++;
        if (testGraphProperties()) passed++;
        
        std::cout << "\nResults: " << passed << "/" << total << " tests passed" << std::endl;
        
        if (passed == total) {
            std::cout << "All tests passed!" << std::endl;
            return 0;
        } else {
            std::cout << "Some tests failed!" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
