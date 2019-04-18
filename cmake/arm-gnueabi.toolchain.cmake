set(GCC_COMPILER_VERSION "" CACHE STRING "GCC Compiler version")
set(GNU_MACHINE "arm-linux-gnueabi" CACHE STRING "GNU compiler triple")
set(CUDA_TARGET "armv7-linux-gnueabihf")
include("${CMAKE_CURRENT_LIST_DIR}/arm.toolchain.cmake")
