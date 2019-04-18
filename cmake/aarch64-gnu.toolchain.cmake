set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(GCC_COMPILER_VERSION "" CACHE STRING "GCC Compiler version")
set(GNU_MACHINE "aarch64-linux-gnu" CACHE STRING "GNU compiler triple")
set(CUDA_TARGET "aarch64-linux")
include("${CMAKE_CURRENT_LIST_DIR}/arm.toolchain.cmake")
## https://stackoverflow.com/questions/30124264/undefined-reference-to-googleprotobufinternalempty-string-abicxx11
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
