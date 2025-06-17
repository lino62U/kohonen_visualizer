cmake_minimum_required(VERSION 3.10)
project(kohonen_visualizer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Optimizaciones agresivas ---
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(
        -O3
        -march=native
        -ffast-math
    )
endif()

# --- Buscar dependencias ---
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenMP REQUIRED)

# Buscar SOIL
find_path(SOIL_INCLUDE_DIR NAMES SOIL.h PATHS /usr/include/SOIL /usr/local/include/SOIL)
find_library(SOIL_LIBRARY NAMES SOIL PATHS /usr/lib /usr/local/lib)

if(NOT SOIL_INCLUDE_DIR OR NOT SOIL_LIBRARY)
    message(FATAL_ERROR "SOIL no encontrado. Instálalo con 'sudo apt install libsoil-dev'")
endif()

# --- Incluir directorios ---
include_directories(
    include
    ${OPENGL_INCLUDE_DIRS}
    ${GLUT_INCLUDE_DIRS}
    ${SOIL_INCLUDE_DIR}
)

# --- Archivos fuente comunes (sin main.cpp ni test.cpp) ---
set(COMMON_SOURCES
    src/KohonenNetwork.cpp
    src/KohonenVisualizer.cpp
    src/MNISTLoader.cpp
)

# === Ejecutable principal ===
add_executable(kohonen_visualizer
    src/main.cpp
    ${COMMON_SOURCES}
)

target_link_libraries(kohonen_visualizer
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARIES}
    GLU
    ${SOIL_LIBRARY}
    OpenMP::OpenMP_CXX
)

# === Ejecutable de prueba ===
add_executable(kohonen_test
    src/test.cpp
    ${COMMON_SOURCES}
)

target_link_libraries(kohonen_test
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARIES}
    GLU
    ${SOIL_LIBRARY}
    OpenMP::OpenMP_CXX
)

# === PGO opcional para el ejecutable principal ===
if(CMAKE_BUILD_TYPE STREQUAL "Release" AND UNIX)
    option(USE_PGO "Enable Profile-Guided Optimization" OFF)
    if(USE_PGO)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-generate")
        add_custom_command(TARGET kohonen_main POST_BUILD
            COMMAND ./kohonen_main --pgo-training
            COMMENT "Generating profiling data..."
        )
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-use" PARENT_SCOPE)
    endif()
endif()
