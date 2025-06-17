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
    if [ ! -f build/kohonen_test ]; then
        echo "No está compilado el ejecutable, compilando primero..."
        compilar
    fi
    ./build/kohonen_test
}

function ejecutar_main() {
    echo "== Ejecutando visualizador de red Kohonen =="

    if [ ! -f build/kohonen_visualizer ]; then
        echo "No está compilado el ejecutable, compilando primero..."
        compilar
    fi

    if [ $# -eq 0 ]; then
        echo "Usando parámetros por defecto:"
        echo "Dimensiones: 10x10x10 | Epochs: 5 | LR: 0.1 | Radius: 3.0 | Samples: 5000 | Labels: 10000"
        ./build/kohonen_visualizer 10 10 10 3 0.1 3.0 1000 10000
    elif [ $# -eq 8 ]; then
        ./build/kohonen_visualizer "$@"
    else
        echo "Uso: $0 main <x> <y> <z> <epochs> <learning_rate> <radius> <samples> <labels>"
        echo "O deja los parámetros vacíos para usar los valores por defecto"
        exit 1
    fi
}

if [ $# -lt 1 ]; then
    echo "Uso: $0 {build|test|main ...}"
    exit 1
fi

cmd=$1
shift

case $cmd in
    build)
        compilar
        ;;
    test)
        ejecutar_tests
        ;;
    main)
        ejecutar_main "$@"
        ;;
    *)
        echo "Opción no válida. Usa: build, test, o main"
        exit 1
        ;;
esac
