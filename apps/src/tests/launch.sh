#!/bin/sh -eu

MY_PATH="$(dirname ${0})"
MY_PATH="$(cd ${MY_PATH} && pwd)"
if [ -z "$MY_PATH" ] ; then
    # error; for some reason, the path is not accessible
    # to the script (e.g. permissions re-evaled after suid)
    exit 1  # fail
fi

#exports
OS="$(uname)"
if [ "${OS}" = "CYGWIN_NT-6.3" ] ; then
	export EXT=".exe"
else
  LIBVS_PATH="${MY_PATH}/../../../bin/x64/${1}"
	if [ "${OS}" = "Darwin" ] ; then
		export DYLD_LIBRARY_PATH="../../../external_deps/lib/opencv2/lib/:../../../external_deps/lib/ceres/:$LIBVS_PATH:$DYLD_LIBRARY_PATH:/usr/local/cuda/lib/"
	else
		export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$LIBVS_PATH:/opt/libjpeg-turbo/lib/:/usr/local/lib/:/usr/local/cuda/lib64/"
	fi
	export EXT=""
	export DISPLAY=:0.0
fi

#executing unit-tests
export BINDIR="../../../bin/x64/${1}/test"
FILTER='./*/*.cpp'
NUMTESTS=`find . -maxdepth 1 -type d | wc -l`
SEP="*****"
FAILED_TESTS=""
for i in `ls $FILTER` ; do
	#test name: everything before Test.cpp
	n=`basename "$i" "Test.cpp"`
	echo "$SEP Executing test $n [$BINDIR$n$EXT] $SEP"
	"$BINDIR$n$EXT"
	if [ $? -ne 0 ] ; then
		echo "$SEP Test failed: $n [$BINDIR$n$EXT] $SEP"
		FAILED_TESTS="$FAILED_TESTS\t$n\n"
	fi
	echo "$SEP$SEP$SEP$SEP$SEP$SEP$SEP$SEP$SEP$SEP"
	echo
done

#return value
if [ $(echo -e "$FAILED_TESTS" | wc -l) != 1 ] ; then
	echo "$(($(echo -e $FAILED_TESTS | wc -l)-1)) tests (out of $NUMTESTS) failed:"
	echo "$FAILED_TESTS"
	echo "FAILURE"
	exit 1
fi

echo "SUCCESS"
