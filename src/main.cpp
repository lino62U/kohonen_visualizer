#include "KohonenNetwork.hpp"
#include "MNISTLoader.hpp"
#include "KohonenVisualizer.hpp"
#include <iostream>

Kohonen3D *kohonenNet = nullptr;
KohonenVisualizer *visualizer = nullptr;

void displayWrapper() { visualizer->renderScene(); }
void reshapeWrapper(int w, int h) { visualizer->reshape(w, h); }
void mouseWrapper(int btn, int state, int x, int y) { visualizer->onMouse(btn, state, x, y); }
void motionWrapper(int x, int y) { visualizer->onMotion(x, y); }
void keyboardWrapper(unsigned char key, int x, int y) { visualizer->onKeyboard(key, x, y); }

int main(int argc, char **argv)
{
  // Inicializar GLUT
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutInitWindowSize(1000, 800);
  glutCreateWindow("Kohonen 3D con MNIST");

  // Cargar y entrenar red
  std::string dataset_path = "data/";
  int samples = 5000;

  auto images = MNISTDataset::loadImages(dataset_path + "train-images.idx3-ubyte", samples);
  auto labels = MNISTDataset::loadLabels(dataset_path + "train-labels.idx1-ubyte", samples);

  kohonenNet = new Kohonen3D(10, 10, 10, 28 * 28);
  kohonenNet->train(images, labels, 5, 0.1f, 3.0f);
  kohonenNet->saveModel("save_models/kohonen-5-epochs.txt");
  visualizer = new KohonenVisualizer(kohonenNet);
  visualizer->initGL();
  visualizer->initNeurons();

  glutDisplayFunc(displayWrapper);
  glutReshapeFunc(reshapeWrapper);
  glutMouseFunc(mouseWrapper);
  glutMotionFunc(motionWrapper);
  glutKeyboardFunc(keyboardWrapper);

  glutMainLoop();

  delete visualizer;
  delete kohonenNet;
  return 0;
}
