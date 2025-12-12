#!/bin/bash

print_help() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --build,   -b     Run the build process (without tests)
  --test,    -t     Run the build process with tests
  --install, -i     Install python interface
  --wheel,   -w     Build the python interface wheel
  --doc,     -d     Generate documentation
  --help,    -h     Show this help message and exit
EOF
    exit 0
}

# Default flags
BUILD=0
TEST=0
INSTALL=0
WHEEL=0
DOC=0

# No arguments = show help
if [ $# -eq 0 ]; then
    print_help
fi

# Parse arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --build|-b)
            BUILD=1
            ;;
        --test|-t)
            TEST=1
            ;;
        --install|-i)
            INSTALL=1
            ;;
        --wheel|-w)
            WHEEL=1
            ;;
        --doc|-d)
            DOC=1
            ;;
        --help|-h)
            print_help
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help to see available options."
            exit 1
            ;;
    esac
    shift
done

# BUILD (without tests)
if [ $BUILD -eq 1 ]; then
    echo "Running build (without tests)..."
    cmake -S . -B build -DBUILD_TEST=OFF || { echo "CMake configuration failed"; exit 1; }

    cd build || { echo "Cannot change directory to build"; exit 1; }
    make || { echo "Build failed"; exit 1; }
    cd .. || { echo "Cannot change back to project root"; exit 1; }
fi

# TEST BUILD (with tests)
if [ $TEST -eq 1 ]; then
    echo "Running tests..."
    cmake -S . -B build -DBUILD_TEST=ON || { echo "CMake configuration failed"; exit 1; }

    cd build || { echo "Cannot change directory to build"; exit 1; }
    make || { echo "Build failed"; exit 1; }
    cd .. || { echo "Cannot change back to project root"; exit 1; }

    pip install .[dev] -v --log build.log || { echo "Install failed"; exit 1; }

    bin/test_gauss_seidel_c || { echo "C Tests failed"; exit 1; }
    bin/test_gauss_seidel_cxx || { echo "C++ Tests failed"; exit 1; }
    pytest -s || { echo "Python Tests failed"; exit 1; }
fi

# INSTALL
if [ $INSTALL -eq 1 ]; then
    echo "Installing python interface..."
    pip install . -v --log build.log || { echo "Install failed"; exit 1; }
fi

# WHEEL
if [ $WHEEL -eq 1 ]; then
    echo "Building wheel..."
    pip wheel . -v -w dist --log build.log || { echo "Wheel build failed"; exit 1; }
fi

# DOCUMENTATION
if [ $DOC -eq 1 ]; then
    echo "Building docs..."
    doxygen Doxyfile || { echo "Doxygen failed"; exit 1; }
fi
