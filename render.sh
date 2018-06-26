#!/bin/bash

disablesleepac='gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-timeout 0'
disablesleepbattery='gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-timeout 0'
enablesleepac='gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-timeout 1200'
enablesleepbattery='gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-timeout 600'
disabledim='gsettings set org.gnome.settings-daemon.plugins.power idle-dim false'
enabledim='gsettings set org.gnome.settings-daemon.plugins.power idle-dim true'

echo 'Turning off sleep and dim on monitor'
$disablesleepac
$disablesleepbattery
$disabledim

rm ecubic3 -rf

for xyPlainIndex in {0..23}
do
  echo '.........'
  echo starting $xyPlainIndex
  echo '.........'
  # FRACTAL_MOVEMENT=`echo $xyPlainIndex / 3 - 5.5 | bc`
  FRACTAL_MOVEMENT=`echo $(bc -q <<< scale=1\;$xyPlainIndex/3-5.5)`
  echo $FRACTAL_MOVEMENT
  time python3 fractal_mandlebrot.py -axisModifier $FRACTAL_MOVEMENT -saveIndex $xyPlainIndex
  ffmpeg -loglevel panic -y -f image2 -framerate 100 -i ecubic3/output%001d_$xyPlainIndex.jpeg -vf scale=200x200 out_$xyPlainIndex.gif
  gvfs-open out_$xyPlainIndex.gif
done
echo 'done'

echo 'Turning on sleep and dim on monitor'
$enablesleepac
$enablesleepbattery
$enabledim
