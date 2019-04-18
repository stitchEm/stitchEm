#!/bin/bash -eu

# This script should be launched from the release branch (studio23 or vahana12 for instance).
# It will:
# - Merge this branch into the stable branch and push to this branch
# - Create the tag that you set in argument and push this tag
# - Create a branch that merges the release branch to master and push it
#   (After that you should create the pull request manually)

PRODUCT="${1-}"
TAG="${2-}"
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
STABLE_BRANCH="stable-${PRODUCT}"
MERGE_BRANCH="merge_${CURRENT_BRANCH}_master"


if [ -z "${PRODUCT}" ] || [ -z "${TAG}" ]; then
    echo "Usage:"
    echo "./release.sh PRODUCT TAG"
    echo "./release.sh studio Studio-v2.4.0"
    echo "./release.sh vahanaVR VahanaVR-v1.2.1"
    exit 1
fi
if [ "${PRODUCT}" = "studio" ]; then
    PRODUCT="Studio"
elif [ "${PRODUCT}" = "vahanaVR" ]; then
    PRODUCT="VahanaVR"
else
    echo "Product should be studio or vahanaVR"
    exit 1
fi
REGEX="${PRODUCT}-v[0-9]\.[0-9]\.[0-9](\.(RC|beta|alpha)[0-9])?$"

if [[ "${TAG}" =~ ${REGEX} ]]; then
    echo "tag is ${TAG}"
else
    echo "invalid tag, format should be ${REGEX}"
    echo "${PRODUCT}-v2.1.1"
    echo "${PRODUCT}-v1.4.0.alpha1"
    echo "${PRODUCT}-v2.3.0.beta2"
    echo "${PRODUCT}-v1.3.4.RC3"
    exit 1
fi

echo "current branch is ${CURRENT_BRANCH}"
echo "stable branch is ${STABLE_BRANCH}"
echo "the merge pull request is ${MERGE_BRANCH}"
echo "All sounds good? [y/N]"
read val
if [ "${val}" != "y" ]; then
    exit 0
fi

git checkout "${STABLE_BRANCH}"
git pull --rebase
git merge "${CURRENT_BRANCH}"
git tag -a "${TAG}"
git push --tags
git push
git checkout master
git pull --rebase
git checkout -b "${MERGE_BRANCH}"
git merge "${STABLE_BRANCH}"
git push --set-upstream origin "${MERGE_BRANCH}"

