find_program(XXD xxd DOC "Create C include style files from OpenCL kernel sources/SPIR" NO_CMAKE_FIND_ROOT_PATH)
if(NOT XXD)
  message(FATAL_ERROR "xxd not found!")
endif(NOT XXD)

set(OPENCL_SOURCES
    core1/boundsKernel.cl
    core1/mergerKernel.cl
    core1/undistortKernel.cl
    core1/voronoi.cl
    core1/warpCoordInputKernel.cl
    core1/warpKernel.cl
    core1/warpFaceKernel_x.cl
    core1/warpFaceKernel_y.cl
    core1/warpFaceKernel_z.cl
    core1/warpCoordKernel.cl
    core1/warpLookupKernel.cl
    core1/cubemapMapKernel.cl
    core1/zoneKernel.cl
    # coredepth/sphereSweep.cl
    exampleKernel.cl
    image/blur.cl
    image/downsampler.cl
    image/imageOps.cl
    image/imgExtract.cl
    image/imgInsert.cl
    image/sampling.cl
    image/rotate.cl
    image/unpack.cl
    input/checkerBoard.cl
    input/grid.cl
    memset.cl
    render/geometry.cl
    render/render.cl
    score/scoringKernel.cl
    )

# add the OpenCL files to the StitchEm sources so they will be shown in IDE projects created from CMake
foreach(opencl_source ${OPENCL_SOURCES})
  list(APPEND CL_BACKEND_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/${opencl_source}")
endforeach()

set(VS_LIB_ROOT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)


# use this compiler to create llvm bitcode / SPIR from the kernel source
if(APPLE)

  find_program(OPENCL_OFFLINE_COMPILER openclc
               PATHS ${OpenCL_LIBRARY}/Libraries/
                     /System/Library/Frameworks/OpenCL.framework/Libraries
               DOC "OpenCL offline compiler to build spir files from kernel sources")
  if (ALTERNATIVE_OPENCL_SPIR)
    find_program(OPENCL_OFFLINE_COMPILER_ALTERNATIVE openclc
                        PATHS /System/Library/PrivateFrameworks/GPUCompiler.framework/Libraries
                              /System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/A/Libraries
                        DOC "Alternative OpenCL Offline Compiler")
  endif (ALTERNATIVE_OPENCL_SPIR)

  set(OPENCL_COMPILER_FLAGS
   -emit-llvm
   -c
   -cl-std=CL1.2
   -cl-fast-relaxed-math
   -cl-mad-enable
   -O3
   -Wall -Weverything -Wpedantic
   -Wno-unused-parameter
   -Wno-unused-function
   -Wno-sign-conversion
   -I ${VS_LIB_ROOT_SRC_DIR})

  set(OPENCL_COMPILER_FLAGS_32
      ${OPENCL_COMPILER_FLAGS}
      -arch gpu_32)

  set(OPENCL_COMPILER_FLAGS_64
      ${OPENCL_COMPILER_FLAGS}
      -arch gpu_64)

  if(CMAKE_GENERATOR STREQUAL Ninja)
    set(OPENCL_COMPILER_FLAGS ${OPENCL_COMPILER_FLAGS} -fcolor-diagnostics)
  endif()
endif(APPLE)

if(LINUX)
  if(LINUX_PKG)
    if(NOT EXISTS ${CMAKE_SOURCE_DIR}/external_deps/include/clangcl/opencl_spir.h)
      file (DOWNLOAD https://raw.githubusercontent.com/KhronosGroup/SPIR-Tools/9c498a665a08db583c61fd34b9e2486978c2431b/headers/opencl_spir.h ${CMAKE_SOURCE_DIR}/external_deps/include/clangcl/opencl_spir.h)
    endif()
    find_program(OPENCL_OFFLINE_COMPILER clang
                 DOC "OpenCL offline compiler to build spir files from kernel sources")
    if(NOT OPENCL_OFFLINE_COMPILER)
      message(FATAL_ERROR "openCL compiler (CLANG) not found")
    endif()
  else()
	  find_program(OPENCL_OFFLINE_COMPILER ${CMAKE_EXTERNAL_DEPS}/lib/clang
                       DOC "OpenCL offline compiler to build spir files from kernel sources")
  endif()
elseif(ANDROID)
  find_program(OPENCL_OFFLINE_COMPILER ${ANDROID_CLANG_TOOLCHAIN_ROOT}/bin/clang
               DOC "OpenCL offline compiler to build spir files from kernel sources")
endif()
if(LINUX OR ANDROID)
  set(OPENCL_COMPILER_FLAGS
    -cc1
    -emit-llvm-bc
    -cl-std=CL1.2
    -cl-fast-relaxed-math
    -cl-mad-enable
    -O3
    -I${CMAKE_EXTERNAL_DEPS}/include/clangcl
    -I${VS_LIB_ROOT_SRC_DIR}
    -I${CMAKE_SOURCE_DIR}/lib/src/backend/cl
    -include opencl_spir_wrapper.h)

  set(OPENCL_COMPILER_FLAGS_32
      ${OPENCL_COMPILER_FLAGS}
      -triple spir-unknown-unknown)

  set(OPENCL_COMPILER_FLAGS_64
      ${OPENCL_COMPILER_FLAGS}
      -triple spir64-unknown-unknown)
endif(LINUX OR ANDROID)

if(WINDOWS)
  find_program(OPENCL_OFFLINE_COMPILER ioc64
                   DOC "Intel OpenCL offline compiler to build spir files from kernel sources")
  set(OPENCL_COMPILER_FLAGS
  -cmd=build
  -device=gpu
  -bo="-cl-std=CL1.2 -cl-fast-relaxed-math -cl-mad-enable -I ${VS_LIB_ROOT_SRC_DIR}")

  if (ALTERNATIVE_OPENCL_SPIR)
    find_program(OPENCL_OFFLINE_COMPILER_ALTERNATIVE ${CMAKE_SOURCE_DIR}/bin/clang.exe
                  DOC "OpenCL offline compiler to build spir files from kernel sources")
    set(OPENCL_COMPILER_FLAGS_ALTERNATIVE
      -cc1
      -emit-llvm-bc
      -cl-std=CL1.2
      -cl-fast-relaxed-math
      -cl-mad-enable
      -O0
      -I${CMAKE_EXTERNAL_DEPS}/include/clangcl
      -I${VS_LIB_ROOT_SRC_DIR}
      -include opencl_spir.h)

    set(OPENCL_COMPILER_FLAGS_32
        ${OPENCL_COMPILER_FLAGS_ALTERNATIVE}
        -triple spir-unknown-unknown)

    set(OPENCL_COMPILER_FLAGS_64
        ${OPENCL_COMPILER_FLAGS_ALTERNATIVE}
        -triple spir64-unknown-unknown)
  endif (ALTERNATIVE_OPENCL_SPIR)
endif(WINDOWS)

message(STATUS "Main OpenCL SPIR compiler: ${OPENCL_OFFLINE_COMPILER}")
message(STATUS "Alternative OpenCL SPIR compiler: ${OPENCL_OFFLINE_COMPILER_ALTERNATIVE}")

function(create_spir_with_clang output_32 output_64 compiler)
    # compile OpenCL kernel to SPIR-32
    add_custom_command(OUTPUT ${output_32}
                       COMMAND "${compiler}"
                       ARGS ${OPENCL_COMPILER_FLAGS_32} ${cl_file_full} -o ${output_32}
                       IMPLICIT_DEPENDS C ${cl_file_full}
                       DEPENDS ${cl_file} ${cl_file_deps})
    # compile OpenCL kernel to SPIR-64
    add_custom_command(OUTPUT ${output_64}
                       COMMAND "${compiler}"
                       ARGS ${OPENCL_COMPILER_FLAGS_64} ${cl_file_full} -o ${output_64}
                       IMPLICIT_DEPENDS C ${cl_file_full}
                       DEPENDS ${cl_file} ${cl_file_deps})
   list(APPEND compiled_list ${output_32} ${output_64})
endfunction(create_spir_with_clang)

#put SPIR result into a C file that can be included directly in the source code
function(put_spir_into_file output_32 spir_32 xxd_32 compiled_32 output_64 spir_64 xxd_64 compiled_64)
    add_custom_command(OUTPUT ${output_32}
                       COMMAND ${XXD}
                       ARGS -i "${spir_32}" "${xxd_32}"
                       WORKING_DIRECTORY ${OPENCL_BINARY_DIR}
                       DEPENDS ${compiled_32})

    # put 64 Bit SPIR result into a C file that can be included directly in the source code
    add_custom_command(OUTPUT ${output_64}
                       COMMAND ${XXD}
                       ARGS -i "${spir_64}" "${xxd_64}"
                       WORKING_DIRECTORY ${OPENCL_BINARY_DIR}
                       DEPENDS ${compiled_64})
endfunction(put_spir_into_file)


# combine into a single file
function(concatenate_xxd_files output_xxd)
    if(WINDOWS)
      add_custom_command(OUTPUT ${output_xxd}
                          COMMAND type
                          ARGS ${ARGV} >  ${output_xxd} 2> NUL
                          WORKING_DIRECTORY ${OPENCL_BINARY_DIR}
                          DEPENDS ${ARGN})
    else()
      add_custom_command(OUTPUT ${output_xxd}
                          COMMAND sh
                          ARGS -c 'cat ${ARGV} > ${output_xxd}'
                          WORKING_DIRECTORY ${OPENCL_BINARY_DIR}
                          DEPENDS ${ARGN})
    endif(WINDOWS)
endfunction(concatenate_xxd_files)

include_directories(${VS_LIB_ROOT_SRC_DIR})

make_directory(${OPENCL_BINARY_DIR})

set(target_to_serialize_compilation )

foreach(cl_file ${OPENCL_SOURCES})

  get_filename_component(cl_file_noext ${cl_file} NAME_WE)
  get_filename_component(cl_file_full ${cl_file} ABSOLUTE)
  get_filename_component(cl_file_dir ${cl_file_full} DIRECTORY)

  set(cl_file_xxd "${OPENCL_BINARY_DIR}/${cl_file_noext}.xxd")
  set(cl_file_xxd_32 "${OPENCL_BINARY_DIR}/${cl_file_noext}.xxd32")
  set(cl_file_xxd_64 "${OPENCL_BINARY_DIR}/${cl_file_noext}.xxd64")
  if (ALTERNATIVE_OPENCL_SPIR)
    set(cl_file_xxd_32_alternative "${OPENCL_BINARY_DIR}/${cl_file_noext}_alternative.xxd32")
    set(cl_file_xxd_64_alternative "${OPENCL_BINARY_DIR}/${cl_file_noext}_alternative.xxd64")
  endif (ALTERNATIVE_OPENCL_SPIR)
  # Create a list of deps for OpenCL kernels
  if(WINDOWS)
    # ${CMAKE_C_COMPILER} would be a better choice, but it is set to a 32 Bit variant that doesn't
    # run when generating a Visual Studio project
    find_program(CONFIGURE_TIME_COMPILER cl DOC "Compiler to preprocess OpenCL files at configure time to resolve includes. Use default cl.exe. Did you run vcvarsall.bat?")
    execute_process(COMMAND ${CONFIGURE_TIME_COMPILER} /P /I ${VS_LIB_ROOT_SRC_DIR} /Tc ${cl_file_full} /showIncludes
                    ERROR_VARIABLE cl_file_deps)
    string(REGEX MATCHALL "Note: including file: .*" cl_file_deps "${cl_file_deps}")

    # might not have any includes!
    if (cl_file_deps)
      # only keep the file name (didn't find a way to keep just the subexpressions in MATCHALL)
      string(REPLACE "Note: including file: " " " cl_file_deps ${cl_file_deps})
      string(REPLACE "\n" " " cl_file_deps ${cl_file_deps}) # we don't want newlines
      string(REPLACE "\\" "/" cl_file_deps ${cl_file_deps}) # path backslash to forward slash
      string(REPLACE " " ";" cl_file_deps ${cl_file_deps}) # string --> CMake list
    endif(cl_file_deps)
    string(REPLACE "/" "\\" cl_file_xxd_32 ${cl_file_xxd_32})
    string(REPLACE "/" "\\" cl_file_xxd_64 ${cl_file_xxd_64})
    string(REPLACE "/" "\\" cl_file_xxd ${cl_file_xxd})
    if (ALTERNATIVE_OPENCL_SPIR)
      string(REPLACE "/" "\\" cl_file_xxd_32_alternative ${cl_file_xxd_32_alternative})
      string(REPLACE "/" "\\" cl_file_xxd_64_alternative ${cl_file_xxd_64_alternative})
    endif(ALTERNATIVE_OPENCL_SPIR)

  else(WINDOWS)
    execute_process(COMMAND ${CMAKE_C_COMPILER} -E -M -x c -I${VS_LIB_ROOT_SRC_DIR} ${cl_file_full}
                   OUTPUT_VARIABLE cl_file_deps)
    string(REPLACE "\\" " " cl_file_deps ${cl_file_deps}) # we don't want backslashes
    string(REPLACE "\n" " " cl_file_deps ${cl_file_deps}) # we don't want newlines
    string(REPLACE " " ";" cl_file_deps ${cl_file_deps}) # string --> CMake list
    list(REMOVE_AT cl_file_deps 0) # drop the generated target name
  endif(WINDOWS)

  list(APPEND CL_BACKEND_SOURCES ${cl_file_deps}) # show them in the IDE

  if(DISABLE_OPENCL_SPIR)

    set(cl_file_preprocessed "${OPENCL_BINARY_DIR}/${cl_file_noext}.pre")

    if(WINDOWS)
      set(CL_PREPROCESSOR_ARGS /I ${VS_LIB_ROOT_SRC_DIR} /P /Tc ${cl_file_full} /Fi${cl_file_preprocessed})
      if(CL_ARGS_WORKAROUND)
        set(CL_PREPROCESSOR_ARGS ${CL_PREPROCESSOR_ARGS} /DCL_ARGS_WORKAROUND)
      endif(CL_ARGS_WORKAROUND)
    else(WINDOWS)
      set(CL_PREPROCESSOR_ARGS -E -x c ${cl_file_full} -o ${cl_file_preprocessed} -I${VS_LIB_ROOT_SRC_DIR})
      if(CL_ARGS_WORKAROUND)
        set(CL_PREPROCESSOR_ARGS ${CL_PREPROCESSOR_ARGS} -DCL_ARGS_WORKAROUND)
      endif(CL_ARGS_WORKAROUND)
    endif(WINDOWS)

    # preprocess OpenCL kernel to resolve #include's
    add_custom_command(OUTPUT ${cl_file_preprocessed}
                       COMMAND ${CMAKE_C_COMPILER}
                       ARGS ${CL_PREPROCESSOR_ARGS}
                       IMPLICIT_DEPENDS C ${cl_file_full}
                       DEPENDS ${cl_file} ${cl_file_deps})

    # put preprocessed file into a C file that can be included directly in the source code
    add_custom_command(OUTPUT ${cl_file_xxd}
                       COMMAND ${XXD}
                       ARGS -i ${cl_file_noext}.pre ${cl_file_xxd}
                       WORKING_DIRECTORY ${OPENCL_BINARY_DIR}
                       DEPENDS ${cl_file_preprocessed})

    list(APPEND compiled_list ${cl_file_xxd})

  else(DISABLE_OPENCL_SPIR)

    set(cl_file_compiled_32 "${OPENCL_BINARY_DIR}/${cl_file_noext}.spir32")
    set(cl_file_compiled_64 "${OPENCL_BINARY_DIR}/${cl_file_noext}.spir64")

    if(WINDOWS)
      set(cl_file_compiler_msg_32 "${OPENCL_BINARY_DIR}/${cl_file_noext}.log32.txt")
      set(cl_file_compiler_msg_64 "${OPENCL_BINARY_DIR}/${cl_file_noext}.log64.txt")
      # compile OpenCL kernel to SPIR
      add_custom_command(OUTPUT ${cl_file_compiler_msg_32}
                         COMMAND ${OPENCL_OFFLINE_COMPILER}
                         ARGS ${OPENCL_COMPILER_FLAGS} -input="${cl_file_full}" -spir32="${cl_file_compiled_32}" -output=${cl_file_compiler_msg_32}
                         WORKING_DIRECTORY ${cl_file_dir}
                         IMPLICIT_DEPENDS C ${cl_file_full}
                         DEPENDS ${cl_file} ${cl_file_deps} ${target_to_serialize_compilation})

      add_custom_command(OUTPUT ${cl_file_compiled_32}
                         COMMAND ${CMAKE_COMMAND}
                         -Dcompiler_output=${cl_file_compiler_msg_32}
                         -Dspir_output=${cl_file_compiled_32}
                         -P ${CMAKE_CURRENT_SOURCE_DIR}/windows_opencl_compiler.cmake
                         WORKING_DIRECTORY ${cl_file_dir}
                         DEPENDS ${cl_file_compiler_msg_32})

      add_custom_command(OUTPUT ${cl_file_compiler_msg_64}
                         COMMAND ${OPENCL_OFFLINE_COMPILER}
                         ARGS ${OPENCL_COMPILER_FLAGS} -input="${cl_file_full}" -spir64="${cl_file_compiled_64}" -output=${cl_file_compiler_msg_64}
                         WORKING_DIRECTORY ${cl_file_dir}
                         IMPLICIT_DEPENDS C ${cl_file_full}
                         DEPENDS ${cl_file} ${cl_file_deps} ${cl_file_compiled_32})

      add_custom_command(OUTPUT ${cl_file_compiled_64}
                         COMMAND ${CMAKE_COMMAND}
                         -Dcompiler_output=${cl_file_compiler_msg_64}
                         -Dspir_output=${cl_file_compiled_64}
                         -P ${CMAKE_CURRENT_SOURCE_DIR}/windows_opencl_compiler.cmake
                         WORKING_DIRECTORY ${cl_file_dir}
                         DEPENDS ${cl_file_compiler_msg_64})

      set(target_to_serialize_compilation ${cl_file_compiled_64})

    else(WINDOWS)
      create_spir_with_clang(${cl_file_compiled_32} ${cl_file_compiled_64} ${OPENCL_OFFLINE_COMPILER})
    endif(WINDOWS)

    put_spir_into_file(${cl_file_xxd_32} ${cl_file_noext}.spir32 ${cl_file_noext}.xxd32 ${cl_file_compiled_32} ${cl_file_xxd_64} ${cl_file_noext}.spir64 ${cl_file_noext}.xxd64 ${cl_file_compiled_64})

    if (ALTERNATIVE_OPENCL_SPIR)
      set(cl_file_compiled_32_alternative "${OPENCL_BINARY_DIR}/${cl_file_noext}_alternative.spir32")
      set(cl_file_compiled_64_alternative "${OPENCL_BINARY_DIR}/${cl_file_noext}_alternative.spir64")
      #create the alternative SPIR
      create_spir_with_clang(${cl_file_compiled_32_alternative} ${cl_file_compiled_64_alternative} ${OPENCL_OFFLINE_COMPILER_ALTERNATIVE})

      # put Alternative SPIR result into a C file
      put_spir_into_file(${cl_file_xxd_32_alternative} ${cl_file_noext}_alternative.spir32 ${cl_file_noext}_alternative.xxd32 ${cl_file_compiled_32_alternative}
             ${cl_file_xxd_64_alternative} ${cl_file_noext}_alternative.spir64 ${cl_file_noext}_alternative.xxd64 ${cl_file_compiled_64_alternative})
      # combine into a single file
      concatenate_xxd_files(${cl_file_xxd} ${cl_file_xxd_32} ${cl_file_xxd_64} ${cl_file_xxd_32_alternative} ${cl_file_xxd_64_alternative})


    else(ALTERNATIVE_OPENCL_SPIR)
      concatenate_xxd_files(${cl_file_xxd} ${cl_file_xxd_32} ${cl_file_xxd_64})
    endif(ALTERNATIVE_OPENCL_SPIR)

    list(APPEND compiled_list ${cl_file_xxd})

  endif(DISABLE_OPENCL_SPIR)

endforeach()

set(CL_BACKEND_SOURCES ${CL_BACKEND_SOURCES} PARENT_SCOPE)

add_custom_target(opencl_compilation
                  DEPENDS ${compiled_list})
set_property(TARGET opencl_compilation PROPERTY FOLDER "lib")
