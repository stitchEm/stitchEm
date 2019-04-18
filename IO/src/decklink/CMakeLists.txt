# safeguard against accidental misuse
if(NOT WINDOWS)
  message(FATAL_ERROR "decklink for Windows only!")
endif(NOT WINDOWS)

set(PLUGIN_NAME decklink)

add_custom_command(
  OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/include/DeckLinkAPI_h.h" "${CMAKE_CURRENT_SOURCE_DIR}/DeckLinkAPI_i.c"
  COMMAND midl.exe /D _DEBUG /W1 /nologo /char signed /env x64 /iid ${CMAKE_CURRENT_SOURCE_DIR}/DeckLinkAPI_i.c /h "${CMAKE_CURRENT_SOURCE_DIR}/include/DeckLinkAPI_h.h" /tlb "${VS_VAHANA_PLUGIN_DIR}/decklink.tlb" "${CMAKE_EXTERNAL_DEPS}/include/decklink/DeckLinkAPI.idl"
  )

set(SOURCE_FILES
    src/export.cpp
    src/decklink_helpers.cpp
    src/decklink_writer.cpp
    src/decklink_discovery.cpp
    src/decklink_reader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DeckLinkAPI_i.c
    )

set(HEADER_FILES
    include/decklink_helpers.hpp
    include/decklink_writer.hpp
    include/decklink_discovery.hpp
    include/decklink_reader.hpp
    )

vs_add_IO_library(${PLUGIN_NAME} SHARED ${SOURCE_FILES} ${HEADER_FILES} $<TARGET_OBJECTS:common>)
include_lib_vs_headers(${PLUGIN_NAME})
include_discovery_vs_headers(${PLUGIN_NAME})

target_include_directories(${PLUGIN_NAME} PRIVATE include)
target_include_directories(${PLUGIN_NAME} PRIVATE .)
# NOTE: L23 was added as a quick fix.
# @w-m: It's related to giving MIDL a path for /h instead of just a filename,
# so the create source file will include <include/DeckLinkAPI_h.h> or something.
# Maybe the MIDL command can be fixed to put the header in include/DeckLinkAPI_h.h
# but only reference DeckLinkAPI_h.h? Then the search path should be fine.

target_include_directories(${PLUGIN_NAME} PRIVATE ../common/include)
target_include_directories(${PLUGIN_NAME} PRIVATE "${CMAKE_EXTERNAL_DEPS}/include/decklink")

set_property(TARGET ${PLUGIN_NAME} PROPERTY CXX_STANDARD 14)

link_target_to_libvideostitch(${PLUGIN_NAME})

target_compile_definitions(${PLUGIN_NAME} PRIVATE NOMINMAX _USE_MATH_DEFINES)
target_link_libraries(${PLUGIN_NAME} PRIVATE ${VS_DISCOVERY})

