#!/bin/bash

configFile=${1:-}
nbRuns=${2:-1}
imageName=${3:-"qdpy-bipedal_walker"}

memoryLimit=48G
resultsPathInContainer=/home/user/qdpy/examples/bipedal_walker/results
finalresultsPath=$(pwd)/results
finalresultsPathInContainer=/home/user/finalresults
uid=$(id -u)
#confPath=$(pwd)/qdpy/examples/bipedal_walker/conf
confPath=$(pwd)/conf
confPathInContainer=/home/user/qdpy/examples/bipedal_walker/conf
priorityParam="-c 128"

if [ ! -d $finalresultsPath ]; then
    mkdir -p $finalresultsPath
fi

inDockerGroup=`id -Gn | grep docker`
if [ -z "$inDockerGroup" ]; then
    sudoCMD="sudo"
else
    sudoCMD=""
fi
dockerCMD="$sudoCMD docker"

if [ -d "$confPath" ]; then
    confVolParam="-v $confPath:$confPathInContainer"
else
    confVolParam=""
fi

exec $dockerCMD run -i -m $memoryLimit --rm $priorityParam --mount type=tmpfs,tmpfs-size=8589934592,target=$resultsPathInContainer --mount type=bind,source=$finalresultsPath,target=$finalresultsPathInContainer  $confVolParam $imageName  "$uid" "$nbRuns" "$configFile"

# MODELINE	"{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
