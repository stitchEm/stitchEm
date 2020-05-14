# try to determine the architecture of this machine's GPU
if(CUDA_LOCAL_ARCH_ONLY)

  # findCUDA does not set a variable to raw nvcc
  if (WINDOWS)
    set(NVCC ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc.exe)
  else(WINDOWS)
    set(NVCC ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)
  endif(WINDOWS)

  if(EXISTS ${NVCC})
    message(STATUS "Using ${NVCC} to query CUDA device properties")
  else(EXISTS ${NVCC})
    message(STATUS "CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
    message(FATAL_ERROR "Can't find nvcc to query local CUDA device, assumed to be at ${NVCC}")
  endif(EXISTS ${NVCC})

  execute_process(COMMAND ${NVCC} --run cmake/configuration/queryCUDAProps.cu -Wno-deprecated-gpu-targets --output-file ${CMAKE_CURRENT_BINARY_DIR}/queryCUDAProps
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  RESULT_VARIABLE QUERY_CUDA_EXIT
                  OUTPUT_VARIABLE QUERY_CUDA_OUT
                  ERROR_VARIABLE QUERY_CUDA_ERR)

  if((QUERY_CUDA_EXIT EQUAL 1) OR ${QUERY_CUDA_ERR})
    message(FATAL_ERROR "Query for CUDA_LOCAL_ARCH_ONLY failed with message: ${QUERY_CUDA_OUT}\n${QUERY_CUDA_ERR}")
  endif()

  # try out these settings with nvcc
  execute_process(COMMAND ${NVCC} --run cmake/configuration/helloWorld.cu ${CUDA_LOCAL_ARCH_FLAGS} --output-file ${CMAKE_CURRENT_BINARY_DIR}/testCUDACompilationFlags
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  RESULT_VARIABLE NVCC_EXIT
                  OUTPUT_VARIABLE NVCC_OUT
                  ERROR_VARIABLE NVCC_ERR)

  if(NOT (NVCC_EXIT EQUAL 0))
    message(SEND_ERROR "Trying to run nvcc with the runtime queried compute capability failed.\
                         Please make sure the value that was determined is correct\
                         and that your local CUDA installation supports this architecture.")

    message(SEND_ERROR "Computed nvcc flags: ${CUDA_LOCAL_ARCH_FLAGS}")
    message(SEND_ERROR "nvcc output for helloWorld.cu: ${NVCC_OUT}")
    message(FATAL_ERROR "nvcc err for helloWorld.cu: ${NVCC_ERR}")
  else()
    message(STATUS "CUDA runtime queried compute capability: ${QUERY_CUDA_EXIT}")
    set(CUDA_LOCAL_ARCH_FLAGS -gencode=arch=compute_${QUERY_CUDA_EXIT},code="sm_${QUERY_CUDA_EXIT}" )
  endif(NOT (NVCC_EXIT EQUAL 0))
  if(CUDA_TARGET_ARCH)
    message(SEND_ERROR "Discarding CUDA_TARGET_ARCH option incompatible with CUDA_LOCAL_ARCH_ONLY")
  endif(CUDA_TARGET_ARCH)
elseif(CUDA_TARGET_ARCH)
  string(REGEX REPLACE "[ ,:]" ";" CUDA_TARGET_ARCH_LIST "${CUDA_TARGET_ARCH}")
  unset(CUDA_LOCAL_ARCH_FLAGS)
  foreach(CUDA_ARCH IN ITEMS ${CUDA_TARGET_ARCH_LIST})
    set(CUDA_LOCAL_ARCH_FLAGS ${CUDA_LOCAL_ARCH_FLAGS} -gencode=arch=compute_${CUDA_ARCH},code="sm_${CUDA_ARCH}" )
  endforeach(CUDA_ARCH)
endif(CUDA_LOCAL_ARCH_ONLY)
