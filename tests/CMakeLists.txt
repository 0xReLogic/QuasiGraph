# Test configuration for QuasiGraph

# Simple test runner without GTest (always build this)
add_executable(quasigraph_tests basic_tests.cpp)
target_link_libraries(quasigraph_tests QuasiGraph)

# Add as CTest
add_test(NAME BasicTests COMMAND quasigraph_tests)
