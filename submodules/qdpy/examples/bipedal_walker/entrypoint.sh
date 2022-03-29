#!/bin/bash
set -e

uid=$1 #${1:-1000}
nbRuns=$2
configName=$3
shift; shift; shift;

useradd -d /home/user -Ms /bin/bash -u $uid user
chown -R $uid /home/user

# Launch illumination
exec gosu user bash -c "cd /home/user/qdpy/examples/bipedal_walker; for i in $(seq 1 $nbRuns | tr '\n' ' '); do sleep 1; ./bipedal_walker.py -c conf/$configName & sleep 1; done; wait; rsync -avz results/ /home/user/finalresults/"
#exec gosu user bash -c "cd /home/user/qdpy/examples/bipedal_walker; for i in $(seq 1 $nbRuns | tr '\n' ' '); do sleep 1; ./bipedal_walker.py -c conf/$configName; rsync -avz results/ /home/user/finalresults/; done"

# MODELINE	"{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
