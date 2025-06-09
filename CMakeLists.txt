cmake_minimum_required(VERSION 3.10)
project(kohonen_visualizer)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

find_library(SOIL_LIBRARY NAMES SOIL PATHS /usr/lib)
find_path(SOIL_INCLUDE_DIR SOIL.h PATHS /usr/include/SOIL)


include_directories(
    ${OPENGL_INCLUDE_DIRS}
    ${GLUT_INCLUDE_DIRS}
    ${SOIL_INCLUDE_DIR}
)

add_executable(kohonen_visualizer
    src/main.cpp
)

target_link_libraries(kohonen_visualizer
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARIES}
    GLU
    ${SOIL_LIBRARY}
)
