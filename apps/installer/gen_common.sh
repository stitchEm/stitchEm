vsver_major=`echo $vsver | sed "s/^v\([0-9]*\).*/\1/"`
vsver_minor=`echo $vsver | sed "s/v[0-9]*\.\([0-9]*\).*/\1/"`
vsver_micro=`echo $vsver | sed "s/v[0-9]*\.[0-9]*\.\([0-9]*\).*/\1/"`
branch=$(git rev-parse --abbrev-ref HEAD)
if [[ $vsver_minor == $vsver ]]
then 
	let vsver_minor = 0
fi

if [[ $vsver_micro == $vsver ]]
then 
	let vsver_micro=0
fi

vsver_build=`echo $vsver | sed "s/v[0-9]*\.[0-9]*\.[0-9]*-\([0-9]*\).*/\1/"`
vsver3=`echo "$vsver_major.$vsver_minor.$vsver_micro"`
vsver4=`echo "$vsver3.$vsver_build.$"`