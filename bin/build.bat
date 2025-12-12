@echo off
setlocal enabledelayedexpansion

:: Default flags
set BUILD=0
set TEST=0
set INSTALL=0
set WHEEL=0
set DOC=0

:: Function: print help
:print_help
if "%1"=="SHOW" (
    echo Usage: %~nx0 [OPTIONS]
    echo.
    echo Options:
    echo   --build,   -b     Run the build process ^(without tests^)
    echo   --test,    -t     Run the build process with tests
    echo   --install, -i     Install python interface
    echo   --wheel,   -w     Build the python interface wheel
    echo   --doc,     -d     Generate documentation
    echo   --help,    -h     Show this help message and exit
    exit /b 0
)

:: If no arguments, show help
if "%~1"=="" (
    call :print_help SHOW
)

:: Parse arguments
:parse_args
if "%~1"=="" goto after_args

if "%~1"=="--build"  set BUILD=1
if "%~1"=="-b"       set BUILD=1

if "%~1"=="--test"    set TEST=1
if "%~1"=="-t"        set TEST=1

if "%~1"=="--install" set INSTALL=1
if "%~1"=="-i"        set INSTALL=1

if "%~1"=="--wheel"   set WHEEL=1
if "%~1"=="-w"        set WHEEL=1

if "%~1"=="--doc"     set DOC=1
if "%~1"=="-d"        set DOC=1

if "%~1"=="--help"    call :print_help SHOW
if "%~1"=="-h"        call :print_help SHOW

:: Unknown option
if not "%BUILD%"=="1" if not "%TEST%"=="1" if not "%INSTALL%"=="1" if not "%WHEEL%"=="1" if not "%DOC%"=="1" (
    echo Unknown option: %1
    echo Use --help to see available options.
    exit /b 1
)

:: Unknown option
if not "%BUILD%"=="1" if not "%TEST%"=="1" if not "%DOC%"=="1" (
    echo Unknown option: %1
    echo Use --help to see available options.
    exit /b 1
)

shift
goto parse_args

:after_args

:: BUILD (no tests)
if %BUILD%==1 (
    echo Running build (without tests)...
    cmake -S . -B build -DBUILD_TEST=OFF || (
        echo CMake configuration failed
        exit /b 1
    )

    pushd build || (
        echo Cannot change directory to build
        exit /b 1
    )
    cmake --build . || (
        echo Build failed
        popd
        exit /b 1
    )
    popd
)

:: TEST (with tests)
if %TEST%==1 (
    echo Running tests...
    cmake -S . -B build -DBUILD_TEST=ON || (
        echo CMake configuration failed
        exit /b 1
    )

    pushd build || (
        echo Cannot change directory to build
        exit /b 1
    )
    cmake --build . || (
        echo Build failed
        popd
        exit /b 1
    )
    popd

    pip install .[dev] -v --log build.log || (
        echo Install failed
        exit /b 1
    )

    bin\test_gauss_seidel_c || (
        echo C tests failed
        exit /b 1
    )

    bin\test_gauss_seidel_cxx || (
        echo C++ tests failed
        exit /b 1
    )

    pytest -s || (
        echo Python tests failed
        exit /b 1
    )
)

:: INSTALL
if %INSTALL%==1 (
    echo Installing python interface...
    pip install . -v --log build.log || (
        echo Install failed
        exit /b 1
    )
)

:: WHEEL
if %WHEEL%==1 (
    echo Building wheel...
    pip wheel . -v -w dist --log build.log || (
        echo Wheel build failed
        exit /b 1
    )
)

:: DOC
if %DOC%==1 (
    echo Building docs...
    doxygen Doxyfile || (
        echo Doxygen failed
        exit /b 1
    )
)

exit /b 0
