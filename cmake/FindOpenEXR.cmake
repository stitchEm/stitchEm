if(APPLE_MACPORTS)
  find_path(OpenEXR_ROOT_DIR include/openexr/half.h HINTS /opt/local)
else()
  find_path(OpenEXR_ROOT_DIR include/openexr/half.h HINTS "${CMAKE_EXTERNAL_DEPS}" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
endif()

if (APPLE)
  if(APPLE_MACPORTS)
    find_path(OpenEXR_LIBRARY_DIR libHalf.dylib HINTS /opt/local/lib)
  else()
    find_path(OpenEXR_LIBRARY_DIR libHalf.dylib HINTS "${CMAKE_EXTERNAL_DEPS}" PATH_SUFFIXES lib/openexr/ NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
  endif()
elseif (LINUX OR ANDROID)
  if(LINUX_PKG)
    find_library(OpenEXR_LIBRARY_DIR libHalf.so)
  else(LINUX_PKG)
    find_path(OpenEXR_LIBRARY_DIR libHalf.so HINTS "${CMAKE_EXTERNAL_DEPS}" PATH_SUFFIXES lib/openexr/ NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
  endif(LINUX_PKG)
elseif (WIN32)
  find_path(OpenEXR_LIBRARY_DIR Half.lib HINTS "${CMAKE_EXTERNAL_DEPS}" PATH_SUFFIXES lib/openexr/ NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
endif()

if(LINUX_PKG)
  find_path(OpenEXR_INCLUDE_DIR OpenEXR/half.h)
elseif(APPLE_MACPORTS)
  find_path(OpenEXR_INCLUDE_DIR OpenEXR/half.h HINTS /opt/local/include)
else()
  find_path(OpenEXR_INCLUDE_DIR openexr/half.h HINTS "${CMAKE_EXTERNAL_DEPS}" PATH_SUFFIXES include/ NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
endif()

if(LINUX_PKG OR APPLE_MACPORTS)

  if(OpenEXR_INCLUDE_DIR AND EXISTS "${OpenEXR_INCLUDE_DIR}/OpenEXR/OpenEXRConfig.h")
    file(STRINGS
         ${OpenEXR_INCLUDE_DIR}/OpenEXR/OpenEXRConfig.h
         TMP
         REGEX "#define OPENEXR_VERSION_STRING.*$")
    string(REGEX MATCHALL "[0-9.]+" OPENEXR_VERSION ${TMP})
  endif()

else()

  if(OpenEXR_INCLUDE_DIR AND EXISTS "${OpenEXR_INCLUDE_DIR}/openexr/OpenEXRConfig.h")
    file(STRINGS
         ${OpenEXR_INCLUDE_DIR}/openexr/OpenEXRConfig.h
         TMP
         REGEX "#define OPENEXR_VERSION_STRING.*$")
    string(REGEX MATCHALL "[0-9.]+" OPENEXR_VERSION ${TMP})
  endif()
endif()

foreach(OpenEXR_LIB Half Iex Imath IlmImf IlmThread)
    if (LINUX OR ANDROID)
      if(LINUX_PKG)
        find_library(OpenEXR_${OpenEXR_LIB}_LIBRARY ${OpenEXR_LIB})
      else()
        find_library(OpenEXR_${OpenEXR_LIB}_LIBRARY ${OpenEXR_LIB} NO_DEFAULT_PATH HINTS "${CMAKE_EXTERNAL_DEPS}" PATH_SUFFIXES lib/ NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
      endif()
    elseif (APPLE_MACPORTS)
      find_library(OpenEXR_${OpenEXR_LIB}_LIBRARY ${OpenEXR_LIB} NO_DEFAULT_PATH HINTS ${OpenEXR_ROOT_DIR} PATH_SUFFIXES lib/)
    else()
      find_library(OpenEXR_${OpenEXR_LIB}_LIBRARY ${OpenEXR_LIB} NO_DEFAULT_PATH HINTS "${CMAKE_EXTERNAL_DEPS}" PATH_SUFFIXES lib/openexr/ NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    endif()

    if(OpenEXR_${OpenEXR_LIB}_LIBRARY)
        list(APPEND OpenEXR_LIBRARIES ${OpenEXR_${OpenEXR_LIB}_LIBRARY})
    endif()
endforeach(OpenEXR_LIB)

# So #include <half.h> works
list(APPEND OpenEXR_INCLUDE_DIRS ${OpenEXR_INCLUDE_DIR})
list(APPEND OpenEXR_INCLUDE_DIRS ${OpenEXR_INCLUDE_DIR}/openexr)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenEXR
    REQUIRED_VARS
        OpenEXR_INCLUDE_DIRS
        OpenEXR_LIBRARY_DIR
    VERSION_VAR
        OpenEXR_VERSION
        )
