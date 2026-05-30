set(LLVM_LINK_COMPONENTS support)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dacppLib/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dpcppLib/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/parser/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/rewriter/include)
include_directories(/usr/include)

set(DACPPLIB_SOURCES
  dacppLib/lib/exception.cpp
  dacppLib/include/ReconTensor.h
)
set(PARSER_SOURCES
  parser/lib/ASTParse.cpp
  parser/lib/Calc.cpp
  parser/lib/DacppStructure.cpp
  parser/lib/Param.cpp
  parser/lib/Shell.cpp
  parser/lib/Split.cpp
  parser/lib/Dacfor.cpp
)
set(REWRITER_SOURCES
  # Keep this list limited to active rewriter/template sources.
  # Legacy variants were moved to translator/archive/.
  rewriter/lib/usm_time_template.cpp
  rewriter/lib/usm_template.cpp
  rewriter/lib/Rewriter_Buffer_new.cpp
  rewriter/lib/buffer_template_new.cpp
  rewriter/lib/universal_template.cpp
  rewriter/lib/Rewriter_MPI.cpp
  rewriter/lib/Rewriter_MPI_Stencil.cpp
  # --- MPI shared ---
  rewriter/lib/mpi/shared/MpiTypes.cpp
  rewriter/lib/mpi/shared/OutputSyncAnalysis.cpp
  rewriter/lib/mpi/shared/PrintRewrite.cpp
  rewriter/lib/mpi/shared/ParamModeAnalysis.cpp
  rewriter/lib/mpi/shared/PostRegionAnalysis.cpp
  rewriter/lib/mpi/shared/PostRegionCodegen.cpp
  rewriter/lib/mpi/shared/LoopLoweredRewrite.cpp
  rewriter/lib/mpi/shared/MpiPlanBuilder.cpp
  # --- MPI legacy access pattern ---
  rewriter/lib/mpi/legacy_access_pattern/PatternInit.cpp
  rewriter/lib/mpi/legacy_access_pattern/WrapperCodegen.cpp
  # --- MPI operator resident / shell-derived phase 1-3 ---
  rewriter/lib/mpi/operator_resident/SplitBindAnalysis.cpp
  rewriter/lib/mpi/operator_resident/ScalarAccessAnalysis.cpp
  rewriter/lib/mpi/operator_resident/Direct1DPartitionAnalysis.cpp
  rewriter/lib/mpi/operator_resident/RowBlock2DPartitionAnalysis.cpp
  rewriter/lib/mpi/operator_resident/FixedBlockPartitionAnalysis.cpp
  rewriter/lib/mpi/operator_resident/FullPayloadPartitionAnalysis.cpp
  rewriter/lib/mpi/operator_resident/StencilWindow1DPartitionAnalysis.cpp
  rewriter/lib/mpi/operator_resident/StencilWindowPartitionAnalysis.cpp
  rewriter/lib/mpi/operator_resident/ShellPartitionAnalysis.cpp
  rewriter/lib/mpi/operator_resident/OperatorChainAnalysis.cpp
  rewriter/lib/mpi/operator_resident/ResidencyAnalysis.cpp
  rewriter/lib/mpi/operator_resident/OperatorResidentWrapperCodegen.cpp
  rewriter/lib/mpi/operator_resident/CollectiveCodegenUtils.cpp
  rewriter/lib/mpi/operator_resident/PartitionCodegen.cpp
  rewriter/lib/mpi/operator_resident/LocalKernelCodegen.cpp
  rewriter/lib/mpi/operator_resident/ResidentBufferCodegen.cpp
  rewriter/lib/mpi/operator_resident/LoopLoweredDirectCodegen.cpp
  rewriter/lib/mpi/operator_resident/FixedBlockCodegen.cpp
  rewriter/lib/mpi/operator_resident/LoopLoweredFixedBlockPhaseExchangeCodegen.cpp
  rewriter/lib/mpi/operator_resident/LoopLoweredStencil1DCodegen.cpp
  rewriter/lib/mpi/operator_resident/LoopLocalStencilOwnerLoop.cpp
  rewriter/lib/mpi/operator_resident/StencilWindow1DCodegen.cpp
  rewriter/lib/mpi/operator_resident/StencilWindowCodegen.cpp
  rewriter/lib/mpi/operator_resident/OperatorResidentCodegen.cpp
  # --- MPI stencil phase C ---
  rewriter/lib/mpi/stencil_phase_c/StencilAnalysis.cpp
  rewriter/lib/mpi/stencil_phase_c/StencilAnalysisUtils.cpp
  rewriter/lib/mpi/stencil_phase_c/StencilRouteParse.cpp
  rewriter/lib/mpi/stencil_phase_c/StencilFollowupCollect.cpp
  rewriter/lib/mpi/stencil_phase_c/StencilCodegen.cpp
  rewriter/lib/mpi/stencil_phase_c/StencilCodegenUtils.cpp
  rewriter/lib/mpi/stencil_phase_c/WaveSpecialization.cpp
  rewriter/lib/Multiple_template.cpp
)

set(SINGLE_TRANSLATOR_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/single_overlay)
set(SINGLE_DACPPLIB_SOURCES
  ${SINGLE_TRANSLATOR_ROOT}/dacppLib/lib/exception.cpp
  ${SINGLE_TRANSLATOR_ROOT}/dacppLib/include/ReconTensor.h
)
set(SINGLE_PARSER_SOURCES
  ${SINGLE_TRANSLATOR_ROOT}/parser/lib/ASTParse.cpp
  ${SINGLE_TRANSLATOR_ROOT}/parser/lib/Calc.cpp
  ${SINGLE_TRANSLATOR_ROOT}/parser/lib/DacppStructure.cpp
  ${SINGLE_TRANSLATOR_ROOT}/parser/lib/Param.cpp
  ${SINGLE_TRANSLATOR_ROOT}/parser/lib/Shell.cpp
  ${SINGLE_TRANSLATOR_ROOT}/parser/lib/Split.cpp
  ${SINGLE_TRANSLATOR_ROOT}/parser/lib/Dacfor.cpp
)
set(SINGLE_REWRITER_SOURCES
  ${SINGLE_TRANSLATOR_ROOT}/rewriter/lib/Rewriter_Buffer_new.cpp
  ${SINGLE_TRANSLATOR_ROOT}/rewriter/lib/buffer_template_new.cpp
)

add_library(DacppLib ${DACPPLIB_SOURCES})
add_library(ParserLib ${PARSER_SOURCES})
add_library(RewriterLib ${REWRITER_SOURCES})
add_library(DacppLibSingle ${SINGLE_DACPPLIB_SOURCES})
add_library(ParserLibSingle ${SINGLE_PARSER_SOURCES})
add_library(RewriterLibSingle ${SINGLE_REWRITER_SOURCES})

function(target_use_single_overlay target_name)
  target_include_directories(${target_name} BEFORE PRIVATE
    ${SINGLE_TRANSLATOR_ROOT}/dacppLib/include
    ${SINGLE_TRANSLATOR_ROOT}/dpcppLib/include
    ${SINGLE_TRANSLATOR_ROOT}/parser/include
    ${SINGLE_TRANSLATOR_ROOT}/rewriter/include
    ${CMAKE_CURRENT_SOURCE_DIR}/dacppLib/include
    ${CMAKE_CURRENT_SOURCE_DIR}/dpcppLib/include
    ${CMAKE_CURRENT_SOURCE_DIR}/parser/include
    ${CMAKE_CURRENT_SOURCE_DIR}/rewriter/include
  )
endfunction()

target_use_single_overlay(DacppLibSingle)
target_use_single_overlay(ParserLibSingle)
target_use_single_overlay(RewriterLibSingle)

add_clang_executable(translator
  translator.cpp
)

add_clang_executable(translator_single
  ${SINGLE_TRANSLATOR_ROOT}/translator.cpp
)


set_target_properties(translator PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator
)
set_target_properties(translator_single PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator
)

target_use_single_overlay(translator_single)


clang_target_link_libraries(translator
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
  DacppLib
  ParserLib
  RewriterLib
)

clang_target_link_libraries(translator_single
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
  DacppLibSingle
  ParserLibSingle
  RewriterLibSingle
)

add_dependencies(translator translator_single)

add_subdirectory(tests)
