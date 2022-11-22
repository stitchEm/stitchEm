if(NOT ANDROID AND NOT LINUX)
  set(COMMON_SOURCES
    src/export.cpp)
endif(NOT ANDROID AND NOT LINUX)
set(COMMON_SOURCES
    ${COMMON_SOURCES}
    src/filesystem.cpp
    src/frameRateHelpers.cpp
    src/io.cpp
    src/multiFileInput.cpp
    src/numberedFilesOutput.cpp
    )

if(NOT ANDROID AND NOT LINUX)
  set(COMMON_HEADERS
    include/export.hpp)
endif(NOT ANDROID AND NOT LINUX)
set(COMMON_HEADERS
    ${COMMON_HEADERS}
    include/filesystem.hpp
    include/extensionChecker.hpp
    include/frameRateHelpers.hpp
    include/io.hpp
    include/iopacket.hpp
    include/multiFileInput.hpp
    include/numberedFilesOutput.hpp
    )

add_library(common OBJECT ${COMMON_SOURCES} ${COMMON_HEADERS})
set_property(TARGET common PROPERTY FOLDER "plugins")
set_property(TARGET common PROPERTY POSITION_INDEPENDENT_CODE TRUE)

include_lib_vs_headers(common)
include_discovery_vs_headers(common)
target_include_directories(common PRIVATE include)

add_subdirectory(format)
add_cppcheck(common VS)

