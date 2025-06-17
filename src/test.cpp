#include "KohonenNetwork.hpp"
#include "MNISTLoader.hpp"
#include "KohonenVisualizer.hpp"
#include <iostream>
#include <random>
#include <vector>

Kohonen3D *kohonenNet = nullptr;
KohonenVisualizer *visualizer = nullptr;
int digitWindow = 0;
std::vector<std::vector<float>> images;
std::vector<int> labels;
std::vector<int> neuronLabels(1000, 0); // Adjust size as needed

void displayWrapper() { visualizer->renderScene(); }
void reshapeWrapper(int w, int h) { visualizer->reshape(w, h); }
void mouseWrapper(int btn, int state, int x, int y) { visualizer->onMouse(btn, state, x, y); }
void motionWrapper(int x, int y) { visualizer->onMotion(x, y); }

void processNewNumber() {
    int userLabel = 0;
    std::cout << "Enter a number (1-9): ";
    std::cin >> userLabel;
    if (userLabel < 1 || userLabel > 9) {
        std::cout << "Invalid input. Please enter a number between 1 and 9." << std::endl;
        return;
    }

    // Find all images with the selected label
    std::vector<int> matchingIndices;
    for (int i = 0; i < labels.size(); ++i) {
        if (labels[i] == userLabel) {
            matchingIndices.push_back(i);
        }
    }

    if (matchingIndices.empty()) {
        std::cout << "No images found for label " << userLabel << std::endl;
        return;
    }

    // Select random image
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, matchingIndices.size() - 1);
    int selectedIndex = matchingIndices[dis(gen)];

    // Print actual and predicted label
    std::cout << "Actual label: " << labels[selectedIndex] << std::endl;
    int prediction = kohonenNet->classify(images[selectedIndex], neuronLabels);
    std::cout << "Predicted label: " << prediction << std::endl;

    // Update visualization
    visualizer->setCurrentInput(images[selectedIndex]);
    visualizer->highlightInput(images[selectedIndex]);
    glutSetWindow(digitWindow);
    glutPostRedisplay();
    glutSetWindow(1); // Switch back to main window
    glutPostRedisplay();
}

void displayDigitWrapper() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    visualizer->renderInputDigit();
    glutSwapBuffers();
}

void keyboardWrapper(unsigned char key, int x, int y) {
    if (key == 'n' || key == 'N') {
        processNewNumber();
    } else {
        visualizer->onKeyboard(key, x, y);
    }
}

int main(int argc, char **argv)
{
    // Initialize GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(1000, 800);
    glutCreateWindow("Kohonen 3D con MNIST");

    // Load and train network
    std::string dataset_path = "data/Osmanya/";
    int samples = 5000;

    images = MNISTDataset::loadImages(dataset_path + "train-images.idx3-ubyte", samples);
    labels = MNISTDataset::loadLabels(dataset_path + "train-labels.idx1-ubyte", samples);

    kohonenNet = new Kohonen3D(10, 10, 10, 28 * 28);
    kohonenNet->loadModel("save_models/kohonen-x10-y10-z10-e5-lr0.05-r2.50-s8000-l8000.txt");
    visualizer = new KohonenVisualizer(kohonenNet);
    visualizer->initGL();
    visualizer->initNeurons();

    glutDisplayFunc(displayWrapper);
    glutReshapeFunc(reshapeWrapper);
    glutMouseFunc(mouseWrapper);
    glutMotionFunc(motionWrapper);
    glutKeyboardFunc(keyboardWrapper);

    // Create new window for digit display
    digitWindow = glutCreateWindow("Input Digit");
    glutDisplayFunc(displayDigitWrapper);
    glutInitWindowSize(56, 56); // 28x28 pixels scaled up 10x

    // Initial prompt for number
    processNewNumber();

    glutMainLoop();

    delete visualizer;
    delete kohonenNet;
    return 0;
}