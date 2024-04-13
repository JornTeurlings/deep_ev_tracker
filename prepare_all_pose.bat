@echo off
setlocal enabledelayedexpansion

set "directory=%1"

rem 遍历目录中的所有子文件夹
for /d %%i in ("%directory%\*") do (
    call prepare_pose.bat %1 %%~nxi
)

endlocal
