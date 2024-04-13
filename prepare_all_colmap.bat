@echo off
setlocal enabledelayedexpansion

set "directory=%1"

for /d %%i in ("%directory%\*") do (
    call image2colmap.bat %1 %%~nxi
)

endlocal
