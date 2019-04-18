#!/bin/bash -u
# This script runs VideoStitch Studio with ThreadSanitizer, setting suppression options
# and saving output to a log file
# Suppression file tsan_supp.txt should be available in $HOME path
echo "This is a VideoStitch Studio TSAN auto script"
# 1) Export TSan options environment variable
export TSAN_OPTIONS="suppressions=${HOME}/tsan_supp.txt"
echo "TSAN_OPTIONS=${TSAN_OPTIONS}"
now=$(date +%Y%m%d%H%M%S)
# 2) Set log file name using current timestamp
#    File name is tsan_studio_logs_YYYYmmddHHMMSS.log
logs="${HOME}/tsan_studio_logs_${now}.log"
# 3) Set command name to VideoStitch Studio within app bundle
studio="/Applications/VideoStitch Studio.app/Contents/MacOS/VideoStitch Studio"
echo "Executing ${studio}"
# 4) Launch VideoStitch Studio redirecting output and errors to log file
"${studio}" > "${logs}" 2>&1
# 5) Print log file
echo "=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+"
echo "=+=+=+=+=+=+=+=+=+=+=+=+=+      LOGS     +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+"
echo "=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+"
cat "${logs}"
