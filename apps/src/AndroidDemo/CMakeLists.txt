cmake_minimum_required(VERSION 3.4.0)

# Main example
set(APP_TARGET_NAME stitch360)

if (NOT PROJECT_NAME)
project(${APP_TARGET_NAME})
endif()


set(APP_SRC jni/NativeApp.cpp)
source_group("src" FILES ${APP_SRC})
add_definitions(-DANDROID_ARCH_NAME="${ANDROID_ARCH_NAME}")
set_source_files_properties(jni/NativeApp.cpp PROPERTIES COMPILE_FLAGS "-std=c++11")
add_library(${APP_TARGET_NAME} SHARED ${APP_SRC})

set(VS_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/../../..")

# ----------------------------------------------------------------------------
# Copy precompiled libraries headers
# ----------------------------------------------------------------------------
file(GLOB MY_DEPS "${VS_ROOT_DIR}/external_deps/lib64/lib*.so")
file(GLOB MY_PLUGINS "${VS_ROOT_DIR}/bin/x64/release/*_plugins/lib*Plugin.so")

find_library(NVGameWorks NAMES NVGameWorks PATHS "${VS_ROOT_DIR}/external_deps/lib64" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
target_include_directories(${APP_TARGET_NAME} PRIVATE ${VS_ROOT_DIR}/external_deps/include/NVGameWorks)
target_include_directories(${APP_TARGET_NAME} PRIVATE ${VS_ROOT_DIR}/external_deps/include/NVGameWorks/NvFoundation)

#target_include_directories(${APP_TARGET_NAME} PRIVATE ${ANDROID_NDK}/sources/android/native_app_glue/)
if(PROJECT_NAME STREQUAL ${APP_TARGET_NAME})
#additional commands for standalone compilation (outside VideoStitch project)
add_custom_command(
         TARGET ${APP_TARGET_NAME} PRE_BUILD
         COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_LIST_DIR}/libs/${ANDROID_ABI}
         COMMAND ${CMAKE_COMMAND} -E copy ${VS_ROOT_DIR}/bin/x64/release/libvideostitch_opencl.so ${CMAKE_CURRENT_LIST_DIR}/libs/${ANDROID_ABI}
         COMMAND ${CMAKE_COMMAND} -E copy ${VS_ROOT_DIR}/bin/x64/release/libvideostitch-gpudiscovery.so ${CMAKE_CURRENT_LIST_DIR}/libs/${ANDROID_ABI}
         COMMAND ${CMAKE_COMMAND} -E copy ${MY_DEPS} ${CMAKE_CURRENT_LIST_DIR}/libs/${ANDROID_ABI}
         COMMAND ${CMAKE_COMMAND} -E copy ${MY_PLUGINS} ${CMAKE_CURRENT_LIST_DIR}/libs/${ANDROID_ABI})
find_library(VS_LIB_OPENCL NAMES videostitch_opencl PATHS "${VS_ROOT_DIR}/bin/x64/release" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
find_library(VS_DISCOVERY NAMES videostitch-gpudiscovery PATHS "${VS_ROOT_DIR}/bin/x64/release" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
target_include_directories(${APP_TARGET_NAME} PRIVATE ${VS_ROOT_DIR}/lib/include)
target_include_directories(${APP_TARGET_NAME} PRIVATE ${VS_ROOT_DIR}/discovery/include)
else()
include_lib_vs_headers(${APP_TARGET_NAME})
include_discovery_vs_headers(${APP_TARGET_NAME})
endif()

find_library(log-lib log)
find_library(android-lib android)
find_library(EGL-lib EGL)
find_library(GLESv2-lib GLESv2)
if (WINDOWS)
  set_target_properties(${APP_TARGET_NAME} PROPERTIES COMPILE_FLAGS "/Wall")
else(WINDOWS)
  set_target_properties(${APP_TARGET_NAME} PROPERTIES COMPILE_FLAGS "-Wno-unused-parameter -DNDEBUG")
endif(WINDOWS)

target_link_libraries(${APP_TARGET_NAME} ${VS_LIB_OPENCL} ${VS_DISCOVERY} ${log-lib} ${android-lib} ${EGL-lib} ${GLESv2-lib} ${NVGameWorks})



