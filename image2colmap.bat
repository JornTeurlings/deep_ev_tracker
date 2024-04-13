@echo off
rem Run rectify_ec.py
python data_preparation\real\rectify_ec.py %1 %2

rem Run colmap.py generate
python data_preparation\colmap.py generate %1/%2 EC
