#!/bin/bash -u

# Usage:
# ./behave.sh will launch tests with videostitch-cmd in ASSETS
# ./behave.sh local will launch all tests with videostitch-cmd from the local build (debug)
# ./behave.sh local release will launch all tests ... (release)
# ./behave.sh local release slow will launch tests tagged as slow ...
# ./behave.sh local release noslow will launch all tests but those tagged as slow ...

MODE="${1-"default"}"
CFG="${2-"debug"}"
SLOW="${3-"both"}"
if [ "${MODE}" = "local" ]; then
    LOCAL="yes"
else
    LOCAL="no"
fi
ARGS=("-t=-perf" "-D" "LOCAL=${LOCAL}" "-D" "CFG=${CFG}")
if [ "${SLOW}" = "slow" ]; then
    ARGS+=("-t=slow")
fi
if [ "${SLOW}" = "noslow" ]; then
    ARGS+=("-t=-slow")
fi

OS="$(uname)"
if [ "${OS}" = "CYGWIN_NT-6.3" ]; then
    cmd /c behave ${ARGS[@]}
    RET=$?
else
    behave ${ARGS[@]}
    RET=$?
fi
cd "reports"
ant "-buildfile" "build.tpl"
rm "TESTS-perf.xml"
exit "${RET}"
