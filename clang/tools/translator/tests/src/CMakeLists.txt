# cmake 最低版本要求
cmake_minimum_required(VERSION 3.0)

# 项目信息
project(dac2sycl)

# 设置使用的编译器为Clang
set(CMAKE_C_COMPILER "/usr/bin/clang")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++")

add_executable(blockMatMulTest blockMatMulTest.cpp)
add_executable(blockMatMulTest2 blockMatMulTest2.cpp)
add_executable(blockMatMulTest3 blockMatMulTest3.cpp)
add_executable(matmulTest matmulTest.cpp)
add_executable(packingTest DataReconstructorTest.cpp)
add_executable(packingTest2 DataReconstructorTest2.cpp)
add_executable(packingTest3 DataReconstructorTest3.cpp)
add_executable(packingTest4 DataReconstructorTest4.cpp)
add_executable(packingTest5 DataReconstructorTest5.cpp)
add_executable(packingTest6 DataReconstructorTest6.cpp)
add_executable(packingTest7 DataReconstructorTest7.cpp)
add_executable(vecMulScalarTest vecMulScalarTest.cpp)
add_executable(bindIndexInitTest bindIndexInitTest.cpp)
add_executable(bindIndexInitTest2 bindIndexInitTest2.cpp)
add_executable(matmul2Test matmul2Test.cpp)
# add_executable(UsmTemplateTest UsmTemplateTest.cpp)
# add_executable(BufferTemplateTest BufferTemplateTest.cpp)
add_executable(ParameterGenerationTest ParameterGenerationTest.cpp)
add_executable(ParameterGenerationTest2 ParameterGenerationTest2.cpp)
add_executable(ParameterGenerationTest3 ParameterGenerationTest3.cpp)
add_executable(ParameterGenerationTest4 ParameterGenerationTest4.cpp)
# add_executable(ParameterGenerationTest5 ParameterGenerationTest5.cpp)
add_executable(ParameterGenerationTest6 ParameterGenerationTest6.cpp)
add_executable(ParameterGenerationTest7 ParameterGenerationTest7.cpp)

set_target_properties(blockMatMulTest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(blockMatMulTest2 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(blockMatMulTest3 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(matmulTest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(packingTest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(packingTest2 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(packingTest3 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(packingTest4 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(packingTest5 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(packingTest6 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(packingTest7 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(vecMulScalarTest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(bindIndexInitTest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(bindIndexInitTest2 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(matmul2Test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
# set_target_properties(UsmTemplateTest PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
# )
# set_target_properties(BufferTemplateTest PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
# )
set_target_properties(ParameterGenerationTest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(ParameterGenerationTest2 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(ParameterGenerationTest3 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(ParameterGenerationTest4 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
# set_target_properties(ParameterGenerationTest5 PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
# )
set_target_properties(ParameterGenerationTest6 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(ParameterGenerationTest7 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)

target_link_libraries(blockMatMulTest DacppLib RewriterLib)
target_link_libraries(blockMatMulTest2 DacppLib RewriterLib)
target_link_libraries(blockMatMulTest3 DacppLib RewriterLib)
target_link_libraries(matmulTest DacppLib RewriterLib)
target_link_libraries(packingTest DacppLib RewriterLib)
target_link_libraries(packingTest2 DacppLib RewriterLib)
target_link_libraries(packingTest3 DacppLib RewriterLib)
target_link_libraries(packingTest4 DacppLib RewriterLib)
target_link_libraries(packingTest5 DacppLib RewriterLib)
target_link_libraries(packingTest6 DacppLib RewriterLib)
target_link_libraries(packingTest7 DacppLib RewriterLib)
target_link_libraries(vecMulScalarTest DacppLib RewriterLib)
target_link_libraries(bindIndexInitTest DacppLib RewriterLib)
target_link_libraries(bindIndexInitTest2 DacppLib RewriterLib)
target_link_libraries(matmul2Test DacppLib RewriterLib)
# target_link_libraries(UsmTemplateTest DacppLib RewriterLib)
# target_link_libraries(BufferTemplateTest DacppLib RewriterLib)
target_link_libraries(ParameterGenerationTest DacppLib RewriterLib)
target_link_libraries(ParameterGenerationTest2 DacppLib RewriterLib)
target_link_libraries(ParameterGenerationTest3 DacppLib RewriterLib)
target_link_libraries(ParameterGenerationTest4 DacppLib RewriterLib)
# target_link_libraries(ParameterGenerationTest5 DacppLib RewriterLib)
target_link_libraries(ParameterGenerationTest6 DacppLib RewriterLib)
target_link_libraries(ParameterGenerationTest7 DacppLib RewriterLib)
