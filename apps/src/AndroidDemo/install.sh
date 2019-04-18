#!/bin/bash -eu

# define Android SDK location
export ANDROID_HOME=$HOME/Android/Sdk

# build and install apk on connected device
./gradlew installDebug

#upload ptv from assets
adb devices | while read line
do
    if [ ! "$line" = "" ] && [ $(echo $line | awk '{print $2}') = "device" ]
    then
        device=$(echo $line | awk '{print $1}')
        echo "$device ..."
        find app/src/main/assets/*.ptv -exec adb -s $device push {} /mnt/sdcard/Android/data/co.orah.stitch360/files \;
#        find app/src/main/assets/*.mp4 -exec adb -s $device push {} /mnt/sdcard/Android/data/co.orah.stitch360/files \;
    fi
done
