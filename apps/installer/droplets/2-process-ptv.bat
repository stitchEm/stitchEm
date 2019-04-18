@echo off
echo Convert processing VideoStitch project %1
echo.
"%~dp0\videostitch-cmd" -i %1
echo.
pause