#!/bin/bash

function compilar() {
    echo "== Compilando proyecto con CMake =="
    [ -d build ] || mkdir build
    cd build
    [ -f Makefile ] || cmake ..
    make -j$(nproc)
    cd ..
}

function ejecutar_tests() {
    echo "== Ejecutando tests =="
    if [ ! -d build ]; then
        echo "No existe la carpeta build, compilando primero..."
        compilar
    fi
    cd build
    CTEST_OUTPUT_ON_FAILURE=1 ctest --output-on-failure
    cd ..
}

function ejecutar_main() {
    echo "== Ejecutando alineador =="
    if [ ! -f build/alineador ]; then
        echo "No está compilado el ejecutable, compilando primero..."
        compilar
    fi
    ./build/kohonen_visualizer
}

if [ $# -eq 0 ]; then
    echo "Uso: $0 {build|test|run}"
    exit 1
fi

case $1 in
    build)
        compilar
        ;;
    test)
        ejecutar_tests
        ;;
    main)
        ejecutar_main
        ;;
    *)
        echo "Opción no válida. Usa: build, test, o run"
        exit 1
        ;;
esac