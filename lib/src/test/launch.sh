#!/bin/sh

#exports
if [ `uname -o` = "Cygwin" ] ; then
	export BINDIR="../../bin/x64/release/test"
	export EXT=".exe"
else
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/../../bin/x64/release:/opt/libjpeg-turbo/lib/:/usr/local/lib/:/opt/cuda/lib64/:/usr/local/cuda/lib64:/usr/local/cuda/lib64/"
	export BINDIR="./"
	export EXT="Test.test"
fi

#executing unit-tests
FILTER='*.test'
NUMTESTS=`ls $FILTER | wc -l`
SEP="*****"
FAILED_TESTS=""
for i in `ls $FILTER` ; do
	#test name: everything before Test.cpp
	n=`basename "$i" "Test.test"`
	echo "$SEP Executing test $n [$BINDIR$i] $SEP"
	"$BINDIR$i"
	if [ $? -ne 0 ] ; then
		echo "$SEP Test failed: $n [$BINDIR$i] $SEP"
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
