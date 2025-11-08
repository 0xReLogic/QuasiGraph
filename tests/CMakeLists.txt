# Test configuration for QuasiGraph

# Find required packages
find_package(GTest QUIET)

if(GTest_FOUND)
    # Test executable
    add_executable(quasigraph_tests
        test_graph.cpp
        test_independent_set.cpp
        test_decomposition.cpp
        test_social_network.cpp
        test_performance.cpp
    )
    
    target_link_libraries(quasigraph_tests
        QuasiGraph
        GTest::gtest_main
    )
    
    # Register tests
    include(GoogleTest)
    gtest_discover_tests(quasigraph_tests)
    
    message(STATUS "Google Test found - building tests")
else()
    message(STATUS "Google Test not found - skipping tests")
endif()

# Simple test runner without GTest
add_executable(simple_tests simple_test_runner.cpp)
target_link_libraries(simple_tests QuasiGraph)
