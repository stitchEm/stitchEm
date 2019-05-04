cmake_minimum_required(VERSION 3.5)
project(r8b)

#Determine the current system
if(APPLE)
  message(STATUS "System: APPLE")
endif(APPLE)

if(${CMAKE_SYSTEM_NAME} MATCHES "Android")
  set(ANDROID True)
  message(STATUS "System: ANDROID")
endif(${CMAKE_SYSTEM_NAME} MATCHES "Android")

if(UNIX AND NOT APPLE AND NOT ANDROID)
  set(LINUX True)
  message(STATUS "System: LINUX")
endif(UNIX AND NOT APPLE AND NOT ANDROID)

if(WIN32)
  set(WINDOWS True)
  message(STATUS "System: WINDOWS")
endif(WIN32)


# find needed packages
if(LINUX OR APPLE)
  find_package(Threads REQUIRED)
endif(LINUX OR APPLE)

set(R8B_SOURCES
    r8bbase.cpp)

add_library(r8b STATIC ${R8B_SOURCES})
target_include_directories(r8b PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# link with the needed libraries
if(LINUX OR APPLE)
  target_link_libraries(r8b Threads::Threads)
  set_target_properties(r8b PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif(LINUX OR APPLE)

if(WINDOWS)
  target_link_libraries(r8b Kernel32)
endif(WINDOWS)

set(R8B_LIBRARY r8b PARENT_SCOPE)
