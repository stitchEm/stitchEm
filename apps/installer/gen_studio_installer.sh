#!/bin/sh

#handle version
vsver=$(./genappsVersion.sh Studio)
export QT_VERSION=$(cat ../qt.version)
export CUDA_VERSION=$(cat ../../cuda.version)

source ./gen_common.sh

#GPU
optionalGpu=""
if [ "$2" != "" ]
then
  optionalGpu="#include \"$2-dll.iss\""
fi

#set details and sign executables
array=("..\..\bin\x64\Release\batchstitcher.exe" "..\..\bin\x64\Release\videostitch-studio.exe" "..\..\bin\x64\Release\videostitch-cmd.exe")
for exe in "${array[@]}" ; do
	verpatch.exe /va $exe $vsver4 /pv $vsver4 /s desc "VideoStitch Studio" /s copyright "Copyright VideoStitch SAS 2017" /s product "VideoStitch Studio"
done

#generate installer			  
sed -e "s/__APPVERSION__/$vsver/g" -e "s/__INFOVERSION__/$vsver3/g" -e "s/__GPU__/$1/g" -e "s/__OTHER_GPU__/$optionalGpu/g" VideoStitch64_studio.iss > tmp.iss
ISCC.exe tmp.iss
cmd.exe /c call sign.bat ../Output/VideoStitch-Studio-Setup.exe
