@echo off
setlocal enabledelayedexpansion

set "directory=%1"

for /d %%i in ("%directory%\*") do (
    call prepare_colmap2image %1 %%~nxi
)

endlocal
