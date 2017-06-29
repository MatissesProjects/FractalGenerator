#!/bin/bash
FIRST_HALF=vids/$1fps_$2.mp4
REV=vids/$1fps_$2_rev.mp4
SECOND_HALF=vids/$1fps_$2_rev_flip.mp4
FILE_LIST=vidList.txt

ffmpeg -r $1 -f image2 -i $2/output%d.jpeg -vcodec libx264 -crf 10 $FIRST_HALF
ffmpeg -i $FIRST_HALF -vf reverse $REV
ffmpeg -i $REV -vf vflip -c:a copy $SECOND_HALF

echo -e "file '$FIRST_HALF'\nfile '$SECOND_HALF'" > vidList.txt

ffmpeg -y -f concat -i $FILE_LIST -c copy vids/$2.mp4
rm $FIRST_HALF $REV $SECOND_HALF $FILE_LIST
