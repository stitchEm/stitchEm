@rem 
@echo off

if "%1" == "" (
	echo batch processing failed
	pause

) else (

:again
    if not "%1" == "" (
        call :process %1
        shift
        goto again
    )
)

echo batch processing complete
pause
exit

:process
echo Processing file %1...
"%~dp0\videostitch-cmd" -i "%~1" -v 4