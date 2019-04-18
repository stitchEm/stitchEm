# Check the result of the Intel OpenCL offline compiler_output

# <rant>
# The Intel OpenCL offline compiler, ioc64 will return a non-0 exit code
# when being invoked with illegal parameters.
#
# It will by default print a lot of stuff to stdout, even if compilation
# does not encounter any errors.
#
# If there is a _build error_ (not an invocation error), it lets you know
# by printing "Build failed!" to stdout. The process exit code will be 0 (!).
# 
# I tried to invoke the compiler with execute_process from here at build time
# capturing its output and printing it and failing if there's 'Build failed' in it.
# 
# Unfortunately, the build option argument of ioc64 needs to be in quotes (""),
# which is not really possible to be done from CMake, due to a limitation in the
# CreateProcess API: http://stackoverflow.com/questions/34905194/cmake-how-to-call-execute-process-with-a-double-quote-in-the-argument-a-k-a-u
#
# I did not find a way to provide the OpenCL build options to the compiler
# without double quotes.
#
# I don't have it in me to follow the proposal from the stackoverflow answer
# (add_custom_command --> call cmake script --> file WRITE a batch script --> execute batch script --> parse output in cmake script)
#
# Now it parses the output frmo the compiler from a file.
# It should normally procduce the compiler error message now, unless the compiler invocation
# itself fails (will stay silent) or the compiler has an error that isn't reported
# with "Build failed", then the error will be hidden / ignored on rebuilds.
# 
# May a soul luckier than me find a sane way to do OpenCL SPIR compilation
# on Windows some day.
# 
# Godspeed.
# </rant>
#
 

file(READ ${compiler_output} OUT)
string(FIND ${OUT} "Build failed!" CONTAINS_BUILD_ERROR)

if((CONTAINS_BUILD_ERROR GREATER 0) OR (NOT EXISTS ${spir_output}))
  message(STATUS "OpenCL SPIR build failed!")
  if(NOT EXISTS ${spir_output})
    message(STATUS "Output file ${spir_output} is missing")
  endif()
  message(STATUS "OpenCL SPIR compiler build log:")
  message(FATAL_ERROR ${OUT})
endif()
