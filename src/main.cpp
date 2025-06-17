#include "KohonenNetwork.hpp"
#include "MNISTLoader.hpp"
#include "KohonenVisualizer.hpp"
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <sstream>

Kohonen3D *kohonenNet = nullptr;
KohonenVisualizer *visualizer = nullptr;

void displayWrapper() { visualizer->renderScene(); }
void reshapeWrapper(int w, int h) { visualizer->reshape(w, h); }
void mouseWrapper(int btn, int state, int x, int y) { visualizer->onMouse(btn, state, x, y); }
void motionWrapper(int x, int y) { visualizer->onMotion(x, y); }
void keyboardWrapper(unsigned char key, int x, int y) { visualizer->onKeyboard(key, x, y); }

void printUsage() {
    std::cout << "Uso: ./kohonen3d <x> <y> <z> <epochs> <learning_rate> <radius> <samples> <labels>\n";
    std::cout << "Ejemplo: ./kohonen3d 10 10 10 5 0.1 3.0 5000 10000\n";
}

std::string buildModelFilename(int x, int y, int z, int epochs, float lr, float radius, int samples, int labels) {
    std::ostringstream ss;
    ss << "save_models/kohonen-"
       << "x" << x << "-y" << y << "-z" << z
       << "-e" << epochs
       << "-lr" << std::fixed << std::setprecision(2) << lr
       << "-r" << std::fixed << std::setprecision(2) << radius
       << "-s" << samples
       << "-l" << labels << ".txt";
    return ss.str();
}

void printModelConfiguration(int x, int y, int z, int epochs, float lr, float radius, int samples, int labels, const std::string& filename) {
    std::cout << "\n========== Configuración del modelo Kohonen ==========\n";
    std::cout << "Tamaño de red           : " << x << " x " << y << " x " << z << "\n";
    std::cout << "Épocas de entrenamiento : " << epochs << "\n";
    std::cout << "Tasa de aprendizaje     : " << lr << "\n";
    std::cout << "Radio inicial           : " << radius << "\n";
    std::cout << "Número de muestras (X)  : " << samples << "\n";
    std::cout << "Número de labels (Y)    : " << labels << "\n";
    std::cout << "Archivo de salida       : " << filename << "\n";
    std::cout << "======================================================\n\n";
}

int main(int argc, char **argv)
{
    if (argc != 9) {
        printUsage();
        return 1;
    }

    // Parsear argumentos
    int dimX = std::atoi(argv[1]);
    int dimY = std::atoi(argv[2]);
    int dimZ = std::atoi(argv[3]);
    int epochs = std::atoi(argv[4]);
    float learningRate = std::atof(argv[5]);
    float radius = std::atof(argv[6]);
    int samples = std::atoi(argv[7]);
    int labels = std::atoi(argv[8]);

    // Inicializar GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(1000, 800);
    glutCreateWindow("Kohonen 3D con MNIST");

    // Cargar y entrenar red
    std::string dataset_path = "data/";

    auto images = MNISTDataset::loadImages(dataset_path + "train-images.idx3-ubyte", samples);
    auto labelList = MNISTDataset::loadLabels(dataset_path + "train-labels.idx1-ubyte", labels);

    kohonenNet = new Kohonen3D(dimX, dimY, dimZ, 28 * 28);

    std::string filename = buildModelFilename(dimX, dimY, dimZ, epochs, learningRate, radius, samples, labels);
    printModelConfiguration(dimX, dimY, dimZ, epochs, learningRate, radius, samples, labels, filename);

    kohonenNet->train(images, labelList, epochs, learningRate, radius);
    kohonenNet->saveModel(filename);

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
