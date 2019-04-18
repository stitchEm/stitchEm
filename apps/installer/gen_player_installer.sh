#!/bin/sh

#handle version
vsver=$(./genappsVersion.sh Player)
export QT_VERSION=$(cat ../qt.version)

source ./gen_common.sh

#set details and sign executables
exe="..\..\bin\x64\Release\VideoStitchPlayer360.exe"
verpatch.exe /va $exe $vsver4 /pv $vsver4 /s desc "VideoStitch Player 360" /s copyright "Copyright VideoStitch SAS 2017" /s product "VideoStitch Player 360"
./sign.bat $exe

#generate installer
sed -e "s/__APPVERSION__/$vsver/g" -e "s/__INFOVERSION__/$vsver3/g" VideoStitch64_player.iss > tmp.iss
"ISCC.exe" tmp.iss
cmd.exe /c call sign.bat ../Output/VideoStitch-Player-Setup.exe
