
# Proyecto de redes Kohonen

## Estructura del Proyecto

```
.
├── CMakeLists.txt
├── data
│   ├── original.cpp
│   ├── t10k-images.idx3-ubyte
│   ├── t10k-labels.idx1-ubyte
│   ├── train-images.idx3-ubyte
│   └── train-labels.idx1-ubyte
├── include
│   ├── KohonenNetwork.hpp
│   ├── KohonenVisualizer.hpp
│   └── MNISTLoader.hpp
├── README.md
├── src
│   ├── KohonenNetwork.cpp
│   ├── KohonenVisualizer.cpp
│   ├── main.cpp
│   └── MNISTLoader.cpp
└── textures
    └── images.jpeg

````

---

## Compilación y Ejecución

El proyecto utiliza CMake para la compilación y un script `run.sh` que facilita las tareas comunes:

### Otorgar permisos al script (solo la primera vez):

```bash
chmod +x run.sh
````

### Comandos disponibles en el script

```bash
./run.sh build    # Construye el proyecto con CMake y Make
./run.sh test     # Ejecuta las pruebas unitarias
./run.sh main     # Ejecuta el programa principal (alineamiento múltiple y estrella)
```

El script crea un directorio `build/`, genera los archivos de construcción con CMake y compila usando todos los núcleos disponibles.



## Resultados y Archivos Generados

* Archivo `alineamiento_estrella.txt`: contiene el alineamiento múltiple generado por el método estrella.
* Archivos de resultados parciales para alineamientos pares.
* Archivos con matrices de puntuación y detalles de alineamientos.



**Resultado de la prueba 1:**

![Resultado de la prueba 1](img/result1.png)


**Resultado de la prueba 2 con FASTA:**

![Resultado de la prueba 2](img/result2.png)

---

## Pruebas Unitarias

Se incluyen pruebas automáticas usando Google Test que validan:

* Lectura correcta de archivos FASTA.
* Precisión del algoritmo Needleman-Wunsch en alineamientos pares.
* Correcta construcción del alineamiento estrella.
* Funciones utilitarias para manejo de secuencias y cálculo de scores.

Ejecuta las pruebas con:

```bash
./run.sh test
```

**Resultados test:**

![Resultado de la prueba 2](img/test.png)

---

## Requisitos del Sistema

* Compilador C++ con soporte para C++11 o superior
* CMake 3.10 o superior
* Google Test (para ejecutar pruebas unitarias)
* Sistema Linux, macOS o Windows con entorno compatible


