#include "KohonenVisualizer.hpp"
#include <SOIL/SOIL.h>

KohonenVisualizer::KohonenVisualizer(Kohonen3D* net) : kohonenNet(net) {}

void KohonenVisualizer::initGL() {
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
    glClearColor(0.2f, 0.2f, 0.3f, 1.0f);
}

GLuint KohonenVisualizer::createTextureFromMNIST(const std::vector<float>& image, int width, int height) {
    std::vector<unsigned char> pixels(width * height * 3);
    for (int i = 0; i < width * height; ++i) {
        unsigned char val = static_cast<unsigned char>(image[i] * 255);
        pixels[i*3 + 0] = val;
        pixels[i*3 + 1] = val;
        pixels[i*3 + 2] = val;
    }

    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    return textureID;
}

void KohonenVisualizer::initNeurons() {
    if (!kohonenNet) return;

    const auto& weights = kohonenNet->getWeights();
    int sizeX = kohonenNet->getSizeX();
    int sizeY = kohonenNet->getSizeY();
    int sizeZ = kohonenNet->getSizeZ();

    for (int x = 0; x < sizeX; ++x) {
        for (int y = 0; y < sizeY; ++y) {
            for (int z = 0; z < sizeZ; ++z) {
                int idx = x * sizeY * sizeZ + y * sizeZ + z;
                Neuron n;
                n.x = x * 2.0f;
                n.y = y * 2.0f;
                n.z = z * 2.0f;
                n.textureID = createTextureFromMNIST(weights[idx], 28, 28);
                neurons.push_back(n);
            }
        }
    }
}

void KohonenVisualizer::drawTexturedQuad(float x, float y, float z, GLuint textureID) {
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, textureID);

    glPushMatrix();
    glTranslatef(x, y, z);
    glBegin(GL_QUADS);
    glColor3f(1, 1, 1);
    glTexCoord2f(0, 0); glVertex3f(-0.8, -0.8, 0.0);
    glTexCoord2f(1, 0); glVertex3f( 0.8, -0.8, 0.0);
    glTexCoord2f(1, 1); glVertex3f( 0.8,  0.8, 0.0);
    glTexCoord2f(0, 1); glVertex3f(-0.8,  0.8, 0.0);
    glEnd();
    glPopMatrix();

    glDisable(GL_TEXTURE_2D);
}

void KohonenVisualizer::renderScene() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    glTranslatef(0, 0, zoom);
    glRotatef(angleX, 1, 0, 0);
    glRotatef(angleY, 0, 1, 0);

    if (kohonenNet) {
        float offsetX = -kohonenNet->getSizeX() + 1.0f;
        float offsetY = -kohonenNet->getSizeY() + 1.0f;
        float offsetZ = -kohonenNet->getSizeZ() + 1.0f;
        glTranslatef(offsetX, offsetY, offsetZ);
    }

    for (const auto& n : neurons) {
        drawTexturedQuad(n.x, n.y, n.z, n.textureID);
    }

    glutSwapBuffers();
}

void KohonenVisualizer::reshape(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45, float(w)/h, 0.1, 100);
    glMatrixMode(GL_MODELVIEW);
}

void KohonenVisualizer::onMouse(int btn, int state, int x, int y) {
    if (btn == GLUT_LEFT_BUTTON) mouseDown = (state == GLUT_DOWN);
    if (btn == 3) zoom += 1.0f;
    if (btn == 4) zoom -= 1.0f;
    lastX = x; lastY = y;
    glutPostRedisplay();
}

void KohonenVisualizer::onMotion(int x, int y) {
    if (mouseDown) {
        angleX += (y - lastY);
        angleY += (x - lastX);
        lastX = x;
        lastY = y;
    }
    glutPostRedisplay();
}

void KohonenVisualizer::onKeyboard(unsigned char key, int, int) {
    if (key == 27) exit(0);
    glutPostRedisplay();
}
