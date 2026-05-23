@echo off
REM Build script for picoCPU. Run from any cmd; the script locates MSVC,
REM CMake, and Ninja itself. Override any of the three with env vars:
REM   set VCVARS=C:\path\to\vcvars64.bat
REM   set CMAKE=C:\path\to\cmake.exe
REM   set NINJA=C:\path\to\ninja.exe

setlocal EnableDelayedExpansion

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

REM ---- Locate vcvars64.bat ----
if "%VCVARS%"=="" (
    REM Prefer vswhere if present (proper MS-recommended way).
    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    if exist "!VSWHERE!" (
        for /f "usebackq tokens=*" %%i in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
            set "VSINSTALL=%%i"
        )
        if defined VSINSTALL set "VCVARS=!VSINSTALL!\VC\Auxiliary\Build\vcvars64.bat"
    )
)
REM Fallback common install paths.
if "%VCVARS%"=="" (
    for %%E in (Enterprise Professional Community BuildTools) do (
        if exist "%ProgramFiles%\Microsoft Visual Studio\2022\%%E\VC\Auxiliary\Build\vcvars64.bat" (
            set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\%%E\VC\Auxiliary\Build\vcvars64.bat"
        )
    )
)
if not exist "%VCVARS%" (
    echo build: vcvars64.bat not found. Set VCVARS env var to override.
    exit /b 1
)

REM ---- Locate cmake.exe ----
if "%CMAKE%"=="" (
    for /f "delims=" %%i in ('where cmake 2^>nul') do (
        if "%CMAKE%"=="" set "CMAKE=%%i"
    )
)
if not exist "%CMAKE%" (
    echo build: cmake.exe not found on PATH. Install CMake or set CMAKE env var.
    exit /b 1
)

REM ---- Locate ninja.exe ----
if "%NINJA%"=="" (
    for /f "delims=" %%i in ('where ninja 2^>nul') do (
        if "%NINJA%"=="" set "NINJA=%%i"
    )
)
if not exist "%NINJA%" (
    echo build: ninja.exe not found on PATH. Install Ninja or set NINJA env var.
    exit /b 1
)

set "CONFIG=%1"
if "%CONFIG%"=="" set "CONFIG=Release"

set "BUILD_DIR=%ROOT%\build\%CONFIG%"
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

call "%VCVARS%" >nul
if errorlevel 1 (
    echo build: vcvars64.bat failed
    exit /b 1
)

pushd "%BUILD_DIR%"

"%CMAKE%" -G "Ninja" ^
    -DCMAKE_MAKE_PROGRAM="%NINJA%" ^
    -DCMAKE_BUILD_TYPE=%CONFIG% ^
    "%ROOT%"
if errorlevel 1 ( popd & exit /b 1 )

"%NINJA%"
if errorlevel 1 ( popd & exit /b 1 )

popd
echo.
echo Build OK. Binaries in: %BUILD_DIR%\bin\
exit /b 0
