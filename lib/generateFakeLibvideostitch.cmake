# Note: this cmake script runs at build time (not at configure time).

message("Creating empty ${VS_LIB_FAKE}.dll")
file(WRITE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${VS_LIB_FAKE}.dll "")
