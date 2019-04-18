@echo off
echo.
echo process with a specific CUDA device
echo.
set /p cuda= Specify the index of the CUDA device to use for processing: 
echo.
"%~dp0\videostitch-cmd" -i %1 -d %cuda% -v 4
echo.
pause