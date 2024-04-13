@echo off
rem Run rectify_ec.py
python data_preparation\real\rectify_ec.py %1 %2

rem Run colmap.py generate
python data_preparation\colmap.py generate %1/%2 EC

rem Run colmap.py extract
python data_preparation\colmap.py extract %1/%2 EC

rem Run prepare_eds_pose_supervision.py
python data_preparation\real\prepare_ec_pose_supervision.py %1/%2 

set "folder_name= %1\%2\events\pose_5\time_surfaces_v2_5"

rem create a new folder
mkdir %folder_name%

rem save the.h5 file in the new folder
move %1\%2\events\pose_5\*.* %folder_name%

rem copy groundtruth to another file
set "file_to_move=%1\%2\groundtruth.txt"
set "destination_folder=%1\%2\colmap"
copy "%file_to_move%" "%destination_folder%"