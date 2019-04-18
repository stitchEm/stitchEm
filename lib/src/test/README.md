# libvideostitch unit tests

## How to run the tests locally

#### CMake (single config command line builds):

After building with your build system (e.g. make or ninja), run `ctest`.

Notes:
- Tests can be run in parallel, `ctest -j 20`
- A subset of tests can be run with a regex, e.g. `ctest -R controller`. Refer to `ctest --help`.
- Tests are labeled. Unit tests can be run with `ctest -L unit`.

#### CMake (projects / IDEs):

A target is created for each test (to run and debug), and an additional target that runs all tests (RUN_TESTS).

Some lib unit tests load data from a relative path and thus require the working directory to be set lib/src/test.



#### Debugging tests
`ctest --output-on-failure` will print the stdout of the test should it fail while being silent for passing tests.

It can be useful to run it directly instead of using the ctest command. On a Ninja build for example, the tests are placed in `<build dir>/src/test/Test`. They can be run like any program with a debugger (e.g. lldb/gdb).

`ctest --repeat-until-fail 100` can be useful to determine whether a previously randomly failing test now works consistently. It can be used in conjunction with `--output-on-failure`
if running a test from your IDE that loads data, you may need to manually set the working directory to lib/src/test



## How to add new tests

#### To add new unit test cases
- Add a new file to lib/src/test, that starts with a lower case word and ends in ...Test.cpp
- Add the file to lib/src/test/CMakeLists.txt

#### Content of the unit test

- `#include` the testing framework, from test/gpu, with its main file called testing.hpp
Note: some unit tests depend on the old testing framework, which was CUDA only and remains in common/testing.hpp of now to keep these tests running. For new tests, be sure to use `gpu/testing.hpp`.

#### Setting up the test

Provide a C++ main from where the setup, test and teardown functions are called.
To properly initialize the unit test, call `VideoStitch::Testing::initTest()`. This sets up a crash handler that will print a backtrace, should the test crash at any time.

#### Testing and assertions

The testing framework in testing.hpp provides a set of equality assertion functions, like `ENSURE`, `ENSURE_EQ` and `ENSURE_ARRAY_EQ`.

#### Mock objects

If you mock parts of the lib as part of your testing that could be usable by other tests, put them in test/common/. You can currently find some mock classes for ptv handling and input/readers.

#### Test data
Prefer data generated in the test over data loaded from disk. If the test depends on some external data, put it in test/data. As this is checked into the repository, the data must not surpass a couple of kiloBytes in size.

#### Best practice
- Single Unit tests should take no more than 10 seconds to complete
- Unit tests may be run in parallel
-- Unit tests must not depend on 100% of the resources being available to a single test
-- Unit tests should not depend on timing information
- If using a random number generator, give it a static seed in the test function





