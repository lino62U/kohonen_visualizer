#include "KohonenVisualizer.hpp"
#include <SOIL/SOIL.h>

KohonenVisualizer::KohonenVisualizer(Kohonen3D *net) : kohonenNet(net) {}

void KohonenVisualizer::initGL()
{
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_TEXTURE_2D);
  glClearColor(0.2f, 0.2f, 0.3f, 1.0f);
}

GLuint KohonenVisualizer::createTextureFromMNIST(const std::vector<float> &image, int width, int height)
{
  std::vector<unsigned char> pixels(width * height * 3);
  for (int i = 0; i < width * height; ++i)
  {
    unsigned char val = static_cast<unsigned char>(image[i] * 255);
    pixels[i * 3 + 0] = val;
    pixels[i * 3 + 1] = val;
    pixels[i * 3 + 2] = val;
  }

  GLuint textureID;
  glGenTextures(1, &textureID);
  glBindTexture(GL_TEXTURE_2D, textureID);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  return textureID;
}

void KohonenVisualizer::initNeurons()
{
  if (!kohonenNet)
    return;

  const auto &weights = kohonenNet->getWeights();
  int sizeX = kohonenNet->getSizeX();
  int sizeY = kohonenNet->getSizeY();
  int sizeZ = kohonenNet->getSizeZ();

  for (int x = 0; x < sizeX; ++x)
  {
    for (int y = 0; y < sizeY; ++y)
    {
      for (int z = 0; z < sizeZ; ++z)
      {
        int idx = x * sizeY * sizeZ + y * sizeZ + z;
        Neuron n;
        n.x = x * 2.0f;
        n.y = y * 2.0f;
        n.z = z * 7.0f;
        n.textureID = createTextureFromMNIST(weights[idx], 28, 28);
        neurons.push_back(n);
      }
    }
  }
}
void KohonenVisualizer::drawTexturedQuad(float x, float y, float z, GLuint textureID, bool selected, const float *borderColor)
{
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, textureID);

  glPushMatrix();
  glTranslatef(x, y, z);
  glBegin(GL_QUADS);
  glColor3f(1, 1, 1);
  glTexCoord2f(0, 0);
  glVertex3f(-0.8, -0.8, 0.0);
  glTexCoord2f(1, 0);
  glVertex3f(0.8, -0.8, 0.0);
  glTexCoord2f(1, 1);
  glVertex3f(0.8, 0.8, 0.0);
  glTexCoord2f(0, 1);
  glVertex3f(-0.8, 0.8, 0.0);
  glEnd();

  if (selected)
  {
    glDisable(GL_TEXTURE_2D);
    glLineWidth(5.0f);
    glColor3fv(borderColor); // usar color recibido
    glBegin(GL_LINE_LOOP);
    glVertex3f(-0.8, -0.8, 0.01);
    glVertex3f(0.8, -0.8, 0.01);
    glVertex3f(0.8, 0.8, 0.01);
    glVertex3f(-0.8, 0.8, 0.01);
    glEnd();
  }

  glPopMatrix();
  glDisable(GL_TEXTURE_2D);
}
void KohonenVisualizer::renderScene()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glLoadIdentity();
  glTranslatef(0, 0, zoom);
  glRotatef(angleX, 1, 0, 0);
  glRotatef(angleY, 0, 1, 0);

  if (kohonenNet)
  {
    float offsetX = -kohonenNet->getSizeX() + 1.0f;
    float offsetY = -kohonenNet->getSizeY() + 1.0f;
    float offsetZ = -kohonenNet->getSizeZ() + 1.0f;
    glTranslatef(offsetX, offsetY, offsetZ);
  }

  for (int i = 0; i < neurons.size(); ++i)
  {
    const auto &n = neurons[i];
    bool isSelected = (i == highlightedNeuronIndex);
    const float greenColor[3] = {0.0f, 1.0f, 0.0f};
    const float yellowColor[3] = {1.0f, 1.0f, 0.0f};

    if (i == highlightedNeuronIndex)
      drawTexturedQuad(n.x, n.y, n.z, n.textureID, true, greenColor);
    else if (i == selectedNeuronIndex)
      drawTexturedQuad(n.x, n.y, n.z, n.textureID, true, yellowColor);
    else
      drawTexturedQuad(n.x, n.y, n.z, n.textureID, false, nullptr);
  }

  glutSwapBuffers();
}

void KohonenVisualizer::reshape(int w, int h)
{
  glViewport(0, 0, w, h);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(45, float(w) / h, 0.1, 100);
  glMatrixMode(GL_MODELVIEW);
}

void KohonenVisualizer::onMouse(int btn, int state, int x, int y)
{
  if (btn == GLUT_LEFT_BUTTON)
  {
    if (state == GLUT_DOWN)
    {
      mouseDown = true;
      lastX = x;
      lastY = y;

      // Detectar neurona seleccionada
      GLint viewport[4];
      GLdouble modelview[16], projection[16];
      GLfloat winX, winY, winZ;
      GLdouble posX, posY, posZ;

      glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
      glGetDoublev(GL_PROJECTION_MATRIX, projection);
      glGetIntegerv(GL_VIEWPORT, viewport);

      winX = (float)x;
      winY = (float)viewport[3] - (float)y;
      glReadPixels(x, int(winY), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &winZ);

      gluUnProject(winX, winY, winZ, modelview, projection, viewport, &posX, &posY, &posZ);

      // Buscar neurona más cercana al clic
      float minDist = 1.0f;
      selectedNeuronIndex = -1;
      for (int i = 0; i < neurons.size(); ++i)
      {
        float dx = neurons[i].x - posX;
        float dy = neurons[i].y - posY;
        float dz = neurons[i].z - posZ;
        float dist = sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < minDist)
        {
          minDist = dist;
          selectedNeuronIndex = i;
        }
      }
    }
    else if (state == GLUT_UP)
    {
      mouseDown = false;
    }
  }

  // Scroll del mouse
  if (btn == 3 && state == GLUT_DOWN) // Rueda arriba
    zoom += 1.0f;
  else if (btn == 4 && state == GLUT_DOWN) // Rueda abajo
    zoom -= 1.0f;

  glutPostRedisplay();
}

void KohonenVisualizer::onMotion(int x, int y)
{
  if (mouseDown)
  {
    angleX += (y - lastY);
    angleY += (x - lastX);
    lastX = x;
    lastY = y;
  }
  glutPostRedisplay();
}

void KohonenVisualizer::onKeyboard(unsigned char key, int, int)
{
  if (key == 27)
    exit(0);
  glutPostRedisplay();
}

void KohonenVisualizer::highlightInput(const std::vector<float> &input)
{
  int bmuIndex = kohonenNet->findBMU(input);
  highlightedNeuronIndex = bmuIndex;
  currentInput = input;
  glutPostRedisplay();
}
void KohonenVisualizer::renderInputDigit()
{
  glClear(GL_COLOR_BUFFER_BIT);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0, 28, 0, 28);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // Render 28x28 image
  glBegin(GL_QUADS);
  for (int y = 0; y < 28; ++y)
  {
    for (int x = 0; x < 28; ++x)
    {
      float pixel = currentInput[y * 28 + x];
      glColor3f(pixel, pixel, pixel);
      glVertex2f(x, 28 - y - 1);
      glVertex2f(x + 1, 28 - y - 1);
      glVertex2f(x + 1, 28 - y);
      glVertex2f(x, 28 - y);
    }
  }
  glEnd();
}
void KohonenVisualizer::setCurrentInput(const std::vector<float> &input)
{
  currentInput = input;
}