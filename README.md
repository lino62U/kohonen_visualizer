
# Visualización y Clasificación de Caracteres con Redes Neuronales de Kohonen 3D

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


## Introducción Teórica

Las redes de **Kohonen** o **Mapas Autoorganizados (SOM, por sus siglas en inglés)** constituyen un tipo de red neuronal no supervisada capaz de proyectar datos de alta dimensión en un espacio de menor dimensión (por ejemplo, 2D o 3D), preservando relaciones topológicas. Esto permite visualizar, agrupar y clasificar patrones complejos como imágenes, secuencias o vectores multivariantes.

Cada neurona en la red de Kohonen tiene un vector de pesos del mismo tamaño que las entradas. Durante el entrenamiento, las neuronas "ganadoras" —aquellas más similares a un vector de entrada— actualizan sus pesos junto con las neuronas vecinas, formando así una representación organizada del espacio de datos.

---

## Implementación de la Red Kohonen 3D

La red desarrollada es una versión tridimensional del SOM. Se inicializa con pesos aleatorios y se entrena de forma competitiva.

A continuación, se muestra un fragmento clave del constructor de la red neuronal:

```cpp
Kohonen3D::Kohonen3D(int sizeX, int sizeY, int sizeZ, int input_dim)
    : sizeX_(sizeX), sizeY_(sizeY), sizeZ_(sizeZ), input_dim_(input_dim),
      neuron_distances_3d(sizeX * sizeY * sizeZ, std::vector<float>(sizeX * sizeY * sizeZ))
{
  ...
  for (int i = 0; i < total_neurons; ++i)
    for (int j = 0; j < input_dim_; ++j)
      weights_[i][j] = dist(gen);
  ...
}
````

El entrenamiento se realiza por épocas, donde cada dato busca su **Best Matching Unit (BMU)**, y las neuronas cercanas a esta ajustan sus pesos:

```cpp
void Kohonen3D::train(const std::vector<Vector> &data, const std::vector<int> &labels,
                      int epochs, float learning_rate_initial, float neighborhood_radius_initial)
{
    ...
    for (int epoch = 0; epoch < epochs; epoch++) {
        ...
        for (int idx = 0; idx < data.size(); ++idx) {
            int winner_idx = findBMU(data[idx]);
            for (int i = 0; i < total_neurons; i++) {
                float dist = neuron_distances_3d[i][winner_idx];
                if (dist <= radius_sq) {
                    float h = std::exp(-dist / (2 * radius_sq));
                    for (int j = 0; j < input_size; j++)
                        weights_[i][j] += lr * h * (input[j] - weights_[i][j]);
                }
            }
        }
    }
}
```

Una vez entrenada, la red asocia etiquetas a las neuronas, permitiendo clasificar nuevas entradas:

```cpp
void Kohonen3D::assignLabels(const std::vector<Vector> &data, const std::vector<int> &labels)
{
    ...
    for (size_t i = 0; i < data.size(); ++i) {
        int bmu = findBMU(data[i]);
        label_counts[bmu][labels[i]]++;
    }
    ...
}
```

---

## Visualizador 3D Interactivo

Se ha desarrollado un **visualizador 3D en OpenGL** que representa cada neurona de la red como un cubo en un espacio tridimensional. Cada cubo muestra la imagen asociada al peso de la neurona (ej. una letra) y permite interacción mediante rotación, zoom y selección.

Al seleccionar una imagen en el visor, se muestra en la interfaz:

* El índice de la imagen.
* La etiqueta real.
* La etiqueta asignada por la red (predicción).

Este componente fue desarrollado en `KohonenVisualizer.cpp` utilizando **OpenGL + GLFW**.

---

## Conjunto de Datos: AfroMNIST

El modelo fue entrenado con la base de datos **AfroMNIST**, una variante del conjunto MNIST adaptada para alfabetos africanos.

**AfroMNIST** contiene caracteres manuscritos pertenecientes a varios sistemas de escritura:

* **Vai**: Sistema silábico usado en Liberia y Sierra Leona.
* **N’Ko**: Escritura del idioma mandinga, utilizada en África Occidental.
* **Ethiopic (Ge’ez)**: Sistema de escritura utilizado para el amárico y otras lenguas etíopes.
* **Osmanya**: Escritura creada para representar el idioma somalí.

Cada imagen es de 28×28 píxeles, similar a MNIST, y el conjunto incluye sus etiquetas correspondientes.

---

## Resultados del Modelo

La red Kohonen fue entrenada con éxito sobre AfroMNIST en una configuración 3×3×3 de neuronas (27 neuronas en total). A continuación, se presentan algunas de las imágenes asociadas a las neuronas tras el entrenamiento:

### Imagenes Generadas por el Visualizador

| Neuronas entrenadas | Representación visual         |
| ------------------- | ----------------------------- |
| Vai                 | ![Vai](img/vai.png)           |
| N'Ko                | ![NKo](img/nko.png)           |
| Ethiopic            | ![Ethiopic](img/ethiopic.png) |
| Osmanya             | ![Osmanya](img/osmanya.png)   |

Estas imágenes corresponden a las neuronas más representativas de cada clase en el conjunto AfroMNIST. Se observa que la red ha logrado **agrupar visualmente caracteres similares**, reflejando la capacidad de autoorganización del SOM.

---

## Conclusiones

* Se ha implementado exitosamente una **red neuronal de Kohonen tridimensional**, capaz de clasificar y visualizar caracteres de múltiples alfabetos africanos.
* La red logra **autoorganizar los datos sin supervisión directa**, asignando etiquetas con alta precisión a través de la identificación del BMU.
* El **visualizador 3D** permite comprender mejor la organización interna del mapa, observando la distribución espacial de los patrones aprendidos.
* AfroMNIST resultó ser un conjunto de datos útil y desafiante para evaluar la robustez del modelo en contextos de alfabetos diversos y no latinos.
* Esta herramienta es extensible a otros conjuntos de datos, y puede adaptarse fácilmente para representar redes 2D o 3D de mayor resolución.

---

## Compilación y Ejecución

El proyecto se compila con CMake. Puede usar el script `run.sh` para automatizar las tareas principales:

```bash
chmod +x run.sh
./run.sh build    # Construir el proyecto
./run.sh main     # Ejecutar la aplicación
```

---

## Dependencias

* C++11 o superior
* CMake 3.10+
* OpenGL, GLFW
* Sistema Linux/macOS/Windows

---

## Créditos

Desarrollado como parte de un proyecto de visualización interactiva de redes SOM aplicado a clasificación de escrituras no latinas.

```

