#include <GL/glut.h>
#include <SOIL/SOIL.h> // Para cargar imágenes
#include <vector>
#include <iostream>

struct Neuron {
    float x, y, z;
    GLuint textureID;
};

std::vector<Neuron> neurons;

float zoom = -15.0f, angleX = 20.0f, angleY = -30.0f;
int lastX, lastY; bool mouseDown = false;

GLuint loadTexture(const char* filename) {
    GLuint tex = SOIL_load_OGL_texture(
        filename,
        SOIL_LOAD_AUTO,
        SOIL_CREATE_NEW_ID,
        SOIL_FLAG_INVERT_Y
    );
    if (tex == 0) std::cerr << "Error loading texture: " << filename << "\n";
    return tex;
}

void initNeurons() {
    for (int x = 0; x < 3; ++x)
        for (int y = 0; y < 3; ++y)
            for (int z = 0; z < 3; ++z) {
                Neuron n;
                n.x = x; n.y = y; n.z = z;
                n.textureID = loadTexture("../textures/images.jpeg");
                neurons.push_back(n);
            }
}

void drawTexturedQuad(float x, float y, float z, GLuint textureID) {
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, textureID);

    glPushMatrix();
    glTranslatef(x, y, z);
    glBegin(GL_QUADS);
    glColor3f(1, 1, 1); // color blanco base para no oscurecer la textura
    glTexCoord2f(0, 0); glVertex3f(-0.4, -0.4, 0.0);
    glTexCoord2f(1, 0); glVertex3f( 0.4, -0.4, 0.0);
    glTexCoord2f(1, 1); glVertex3f( 0.4,  0.4, 0.0);
    glTexCoord2f(0, 1); glVertex3f(-0.4,  0.4, 0.0);
    glEnd();
    glPopMatrix();

    glDisable(GL_TEXTURE_2D);
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    glTranslatef(0, 0, zoom);
    glRotatef(angleX, 1, 0, 0);
    glRotatef(angleY, 0, 1, 0);
    glTranslatef(-1.5f, -1.5f, -1.5f);

    for (const auto& n : neurons)
        drawTexturedQuad(n.x, n.y, n.z, n.textureID);

    glutSwapBuffers();
}

void reshape(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45, float(w)/h, 0.1, 100);
    glMatrixMode(GL_MODELVIEW);
}

void mouse(int btn, int state, int x, int y) {
    if (btn == GLUT_LEFT_BUTTON) mouseDown = (state == GLUT_DOWN);
    if (btn == 3) zoom += 1.0f;
    if (btn == 4) zoom -= 1.0f;
    lastX = x; lastY = y;
    glutPostRedisplay();
}

void motion(int x, int y) {
    if (mouseDown) {
        angleX += (y - lastY);
        angleY += (x - lastX);
        lastX = x;
        lastY = y;
    }
    glutPostRedisplay();
}

void keyboard(unsigned char key, int, int) {
    if (key == 27) exit(0);
    glutPostRedisplay();
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(800, 800);
    glutCreateWindow("Kohonen Visualizer con Imágenes");

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);

    initNeurons();

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutKeyboardFunc(keyboard);

    glutMainLoop();
    return 0;
}
