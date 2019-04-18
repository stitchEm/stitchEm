#!/bin/sh

#handle version
vsver_major="1"
vsver_minor="0"
vsver_micro="0"
vsver="$vsver_major.$vsver_minor.$vsver_micro"

#generate installer			  
sed -e "s/__APPVERSION__/$vsver/g" -e "s/__INFOVERSION__/$vsver/g" VideoStitch64_autocrop.iss > tmp.iss
"/cygdrive/c/Program Files (x86)/Inno Setup 5/ISCC.exe" tmp.iss
