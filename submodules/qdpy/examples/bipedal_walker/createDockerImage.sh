#!/bin/sh

imageName=${1:-qdpy-bipedal_walker:latest}

inDockerGroup=`id -Gn | grep docker`
if [ -z "$inDockerGroup" ]; then
	sudoCMD="sudo"
else
	sudoCMD=""
fi

$sudoCMD docker build --no-cache -t $imageName .

