#pragma once

#include "KohonenNetwork.hpp"
#include <vector>
#include <GL/glut.h>

class KohonenVisualizer
{
public:
  KohonenVisualizer(Kohonen3D *net);

  void initGL();
  void initNeurons();
  void renderScene();
  void reshape(int w, int h);
  void onMouse(int btn, int state, int x, int y);
  void onMotion(int x, int y);
  void onKeyboard(unsigned char key, int x, int y);
  int selectedNeuronIndex = -1; // -1 si no hay ninguna seleccionada
  void highlightInput(const std::vector<float> &input);
  int highlightedNeuronIndex = -1;

  void setCurrentInput(const std::vector<float> &input);
  void renderInputDigit();

private:
  struct Neuron
  {
    float x, y, z;
    GLuint textureID;
    int label = -1; // -1 si no tiene etiqueta
  };

  GLuint createTextureFromMNIST(const std::vector<float> &image, int width, int height);
  void drawTexturedQuad(float x, float y, float z, GLuint textureID, bool isSelected, const float *color);
  std::vector<Neuron> neurons;
  Kohonen3D *kohonenNet;
  std::vector<float> currentInput; // Store current input image
  float zoom = -15.0f, angleX = 20.0f, angleY = -30.0f;
  int lastX = 0, lastY = 0;
  bool mouseDown = false;
};
