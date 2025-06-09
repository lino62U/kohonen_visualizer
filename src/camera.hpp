#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <GL/gl.h>
#include <GL/glu.h>

class Camera {
public:
    float angleX = 20.0f;
    float angleY = -30.0f;
    float distance = 10.0f;

    void apply() {
        glTranslatef(0.0f, 0.0f, -distance);
        glRotatef(angleX, 1.0f, 0.0f, 0.0f);
        glRotatef(angleY, 0.0f, 1.0f, 0.0f);
    }

    void rotate(float dx, float dy) {
        angleY += dx;
        angleX += dy;
    }

    void zoom(float dz) {
        distance += dz;
        if (distance < 1.0f) distance = 1.0f;
    }
};

#endif
