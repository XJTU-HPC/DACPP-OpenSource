# cmake 最低版本要求
cmake_minimum_required(VERSION 3.0)

# 项目信息
project(SingleUSMPerformanceTest)

set(CMAKE_CXX_STANDARD 17)

# 设置使用的编译器为 icpx
set(CMAKE_C_COMPILER "/home/tools/intel/oneapi/compiler/2025.2/bin/icpx")
set(CMAKE_CXX_COMPILER "/home/tools/intel/oneapi/compiler/2025.2/bin/icpx")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64")

# 添加 SYCL 头文件路径
set(SYCL_PATH "/home/tools/intel/oneapi/compiler/latest/include")
set(OpenCL_LIBRARY "/home/tools/intel/oneapi/compiler/latest/lib/libOpenCL.so")

include_directories(${SYCL_PATH})
find_package(Threads REQUIRED)
# 添加源文件

add_executable(decay_s_u decay_test.sycl.cpp)
add_executable(decay_s_u_st decay_standard_test.cpp)
add_executable(waveEquation_s_u waveEquation_test_sycl.cpp)
add_executable(waveEquation_s_u_st waveEquation_standard_test.cpp)
add_executable(waveEquation_s_u_dacfor waveEquation_dacfor_test_sycl.cpp)
add_executable(DFT_s_u DFT_test_sycl.cpp)
add_executable(DFT_s_u_st DFT_standard_test.cpp)


set_target_properties(decay_s_u PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests/Performancetesting/SinglemachineUSM
)
set_target_properties(decay_s_u_st PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests/Performancetesting/SinglemachineUSM
)
set_target_properties(waveEquation_s_u PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests/Performancetesting/SinglemachineUSM
)
set_target_properties(waveEquation_s_u_st PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests/Performancetesting/SinglemachineUSM
)
set_target_properties(waveEquation_s_u_dacfor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests/Performancetesting/SinglemachineUSM
)
set_target_properties(DFT_s_u PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests/Performancetesting/SinglemachineUSM
)
set_target_properties(DFT_s_u_st PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests/Performancetesting/SinglemachineUSM
)

target_link_libraries(decay_s_u PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
target_link_libraries(decay_s_u_st PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
target_link_libraries(waveEquation_s_u PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
target_link_libraries(waveEquation_s_u_st PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
target_link_libraries(waveEquation_s_u_dacfor PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
target_link_libraries(DFT_s_u PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
target_link_libraries(DFT_s_u_st PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)