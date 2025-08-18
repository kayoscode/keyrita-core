enable_testing()

add_executable(analyze-tests AnalyzeTests.test.cpp)
target_link_libraries(analyze-tests gtest_main keyrita-core)
add_test(analyze_gtests analyze-tests test_trees analyze_gtests)
