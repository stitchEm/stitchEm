#!/bin/sh

#handle version
vsver=$(./genappsVersion.sh Studio)
export QT_VERSION=$(cat ../qt.version)

source ./gen_common.sh

#set details 
verpatch.exe /va "..\..\bin\x64\Release\startmyorah.exe" $vsver4 /pv $vsver4 /s desc "Orah StartMyOrah" /s copyright "Copyright Orah 2017" /s product "Orah StartMyOrah"

#generate installer			  
sed -e "s/__APPVERSION__/$vsver/g" -e "s/__INFOVERSION__/$vsver3/g" Orah64_startmyorah.iss > tmp.iss
ISCC.exe tmp.iss
