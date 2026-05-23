@echo off
REM Build math_target.exe. Locates vcvars64.bat via vswhere (or fall back
REM to common install paths). Override with VCVARS env var if needed.

setlocal EnableDelayedExpansion

if "%VCVARS%"=="" (
    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    if exist "!VSWHERE!" (
        for /f "usebackq tokens=*" %%i in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
            set "VSINSTALL=%%i"
        )
        if defined VSINSTALL set "VCVARS=!VSINSTALL!\VC\Auxiliary\Build\vcvars64.bat"
    )
)
if "%VCVARS%"=="" (
    for %%E in (Enterprise Professional Community BuildTools) do (
        if exist "%ProgramFiles%\Microsoft Visual Studio\2022\%%E\VC\Auxiliary\Build\vcvars64.bat" (
            set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\%%E\VC\Auxiliary\Build\vcvars64.bat"
        )
    )
)
if not exist "%VCVARS%" (
    echo Cannot find vcvars64.bat. Set VCVARS env var to override.
    exit /b 1
)
call "%VCVARS%" >nul
cl /nologo /O2 /EHsc math_target.cpp /link /OUT:math_target.exe
