cmake_minimum_required(VERSION 3.10)
project(kohonen_visualizer)

set(CMAKE_CXX_STANDARD 17)

# Buscar OpenGL y GLUT
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

# Buscar SOIL
find_path(SOIL_INCLUDE_DIR NAMES SOIL.h PATHS /usr/include/SOIL)
find_library(SOIL_LIBRARY NAMES SOIL PATHS /usr/lib)

if(NOT SOIL_INCLUDE_DIR OR NOT SOIL_LIBRARY)
    message(FATAL_ERROR "SOIL no encontrado. Asegúrate de que está instalado.")
endif()

# Incluir directorios
include_directories(
    include
    ${OPENGL_INCLUDE_DIRS}
    ${GLUT_INCLUDE_DIRS}
    ${SOIL_INCLUDE_DIR}
)

# Recolectar todos los archivos fuente (.cpp) en src/
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Crear ejecutable
add_executable(kohonen_visualizer ${SOURCES})

# Enlazar librerías necesarias
target_link_libraries(kohonen_visualizer
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARIES}
    GLU
    ${SOIL_LIBRARY}
)
