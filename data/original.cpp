#include <GL/glut.h>
#include <SOIL/SOIL.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <utility>

using Vector = std::vector<float>;

// *********************** Clase Kohonen 3D ***********************
class Kohonen3D {
public:
    Kohonen3D(int sizeX, int sizeY, int sizeZ, int input_dim)
        : sizeX_(sizeX), sizeY_(sizeY), sizeZ_(sizeZ), input_dim_(input_dim) {
        
        int total_neurons = sizeX * sizeY * sizeZ;
        weights_.resize(total_neurons, Vector(input_dim));

        for (auto& w : weights_) {
            for (auto& v : w) {
                v = static_cast<float>(rand()) / RAND_MAX;
            }
        }
    }

    void train(const std::vector<Vector>& data, int epochs, float learning_rate_initial, float neighborhood_radius_initial) {
        int total_neurons = sizeX_ * sizeY_ * sizeZ_;
        int input_size = input_dim_;

        for (int epoch = 0; epoch < epochs; epoch++) {
            float lr = learning_rate_initial * (1.0f - (float)epoch / epochs);
            float radius = neighborhood_radius_initial * (1.0f - (float)epoch / epochs);

            for (const auto& input : data) {
                int winner_idx = 0;
                float min_dist = euclideanDistanceVec(input, weights_[0]);
                for (int i = 1; i < total_neurons; i++) {
                    float dist = euclideanDistanceVec(input, weights_[i]);
                    if (dist < min_dist) {
                        min_dist = dist;
                        winner_idx = i;
                    }
                }

                int wx = winner_idx / (sizeY_ * sizeZ_);
                int wy = (winner_idx / sizeZ_) % sizeY_;
                int wz = winner_idx % sizeZ_;

                for (int i = 0; i < total_neurons; i++) {
                    int x = i / (sizeY_ * sizeZ_);
                    int y = (i / sizeZ_) % sizeY_;
                    int z = i % sizeZ_;

                    float dist_to_winner = euclideanDistance3D(x, y, z, wx, wy, wz);
                    if (dist_to_winner <= radius) {
                        float h = std::exp(-(dist_to_winner * dist_to_winner) / (2 * radius * radius));
                        for (int j = 0; j < input_size; j++) {
                            weights_[i][j] += lr * h * (input[j] - weights_[i][j]);
                        }
                    }
                }
            }
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << " done.\n";
        }
    }

    const std::vector<Vector>& getWeights() const { return weights_; }
    int getSizeX() const { return sizeX_; }
    int getSizeY() const { return sizeY_; }
    int getSizeZ() const { return sizeZ_; }

private:
    float euclideanDistanceVec(const Vector& a, const Vector& b) {
        float dist = 0.0f;
        for (size_t i = 0; i < a.size(); i++) {
            float diff = a[i] - b[i];
            dist += diff * diff;
        }
        return std::sqrt(dist);
    }

    float euclideanDistance3D(int x1, int y1, int z1, int x2, int y2, int z2) {
        return std::sqrt(
            (x1 - x2)*(x1 - x2) +
            (y1 - y2)*(y1 - y2) +
            (z1 - z2)*(z1 - z2)
        );
    }

    int sizeX_, sizeY_, sizeZ_;
    int input_dim_;
    std::vector<Vector> weights_;
};
/**/
// *********************** Clase MNIST ***********************
class MNISTDataset {
public:
    static std::vector<std::vector<float>> loadImages(const std::string& filename, int max_images = -1) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open MNIST images file: " + filename);

        int32_t magic, num_images, rows, cols;
        file.read(reinterpret_cast<char*>(&magic), 4);
        file.read(reinterpret_cast<char*>(&num_images), 4);
        file.read(reinterpret_cast<char*>(&rows), 4);
        file.read(reinterpret_cast<char*>(&cols), 4);

        magic = __builtin_bswap32(magic);
        num_images = __builtin_bswap32(num_images);
        rows = __builtin_bswap32(rows);
        cols = __builtin_bswap32(cols);

        if (magic != 2051) throw std::runtime_error("Invalid magic number in MNIST image file");

        if (max_images > 0 && max_images < num_images) {
            num_images = max_images;
        }

        std::vector<std::vector<float>> images;
        images.reserve(num_images);

        for (int i = 0; i < num_images; ++i) {
            std::vector<float> image(rows * cols);
            for (int j = 0; j < rows * cols; ++j) {
                uint8_t pixel;
                file.read(reinterpret_cast<char*>(&pixel), 1);
                image[j] = pixel / 255.0f;
            }
            images.push_back(std::move(image));
        }

        return images;
    }

    static std::vector<std::vector<float>> loadLabels(const std::string& filename, int max_labels = -1) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open MNIST labels file: " + filename);

        int32_t magic, num_labels;
        file.read(reinterpret_cast<char*>(&magic), 4);
        file.read(reinterpret_cast<char*>(&num_labels), 4);

        magic = __builtin_bswap32(magic);
        num_labels = __builtin_bswap32(num_labels);

        if (magic != 2049) throw std::runtime_error("Invalid magic number in MNIST label file");

        if (max_labels > 0 && max_labels < num_labels) {
            num_labels = max_labels;
        }

        std::vector<std::vector<float>> labels;
        labels.reserve(num_labels);

        for (int i = 0; i < num_labels; ++i) {
            uint8_t label;
            file.read(reinterpret_cast<char*>(&label), 1);

            std::vector<float> one_hot(10, 0.0f);
            if (label > 9) throw std::runtime_error("Label out of range");
            one_hot[label] = 1.0f;
            labels.push_back(std::move(one_hot));
        }

        return labels;
    }

    static void displayImage(const std::vector<float>& image, int rows, int cols) {
        const std::string shades = " .:-=+*#%@";
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                float pixel = image[i * cols + j];
                int level = static_cast<int>(pixel * (shades.size() - 1));
                std::cout << shades[level] << shades[level];
            }
            std::cout << '\n';
        }
    }
};

// *********************** Visualización OpenGL ***********************
struct Neuron {
    float x, y, z;
    GLuint textureID;
};

std::vector<Neuron> neurons;
Kohonen3D* kohonenNet = nullptr;
std::vector<std::vector<float>> mnistImages;

float zoom = -15.0f, angleX = 20.0f, angleY = -30.0f;
int lastX, lastY; bool mouseDown = false;

GLuint createTextureFromMNIST(const std::vector<float>& image, int width, int height) {
    // Convertir la imagen MNIST (valores 0-1) a datos de textura (0-255)
    std::vector<unsigned char> pixels(width * height * 3);
    for (int i = 0; i < width * height; ++i) {
        unsigned char val = static_cast<unsigned char>(image[i] * 255);
        pixels[i*3] = val;   // R
        pixels[i*3+1] = val; // G
        pixels[i*3+2] = val; // B
    }

    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    return textureID;
}

void initNeurons() {
    if (!kohonenNet) return;
    
    const auto& weights = kohonenNet->getWeights();
    int sizeX = kohonenNet->getSizeX();
    int sizeY = kohonenNet->getSizeY();
    int sizeZ = kohonenNet->getSizeZ();
    
    for (int x = 0; x < sizeX; ++x) {
        for (int y = 0; y < sizeY; ++y) {
            for (int z = 0; z < sizeZ; ++z) {
                int idx = x * (sizeY * sizeZ) + y * sizeZ + z;
                Neuron n;
                n.x = x * 2.0f; // Espaciado entre neuronas
                n.y = y * 2.0f;
                n.z = z * 2.0f;
                
                // Crear textura desde los pesos de la neurona (que son una imagen MNIST)
                n.textureID = createTextureFromMNIST(weights[idx], 28, 28);
                
                neurons.push_back(n);
            }
        }
    }
}

void drawTexturedQuad(float x, float y, float z, GLuint textureID) {
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, textureID);

    glPushMatrix();
    glTranslatef(x, y, z);
    glBegin(GL_QUADS);
    glColor3f(1, 1, 1); // color blanco base para no oscurecer la textura
    glTexCoord2f(0, 0); glVertex3f(-0.8, -0.8, 0.0);
    glTexCoord2f(1, 0); glVertex3f( 0.8, -0.8, 0.0);
    glTexCoord2f(1, 1); glVertex3f( 0.8,  0.8, 0.0);
    glTexCoord2f(0, 1); glVertex3f(-0.8,  0.8, 0.0);
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
    
    // Centrar la red en el espacio 3D
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

void initGL() {
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
    glClearColor(0.2f, 0.2f, 0.3f, 1.0f);
}

void trainAndVisualize() {
    try {
        std::string dataset_path = "../data/";
        int training_samples = 5000;

        // Cargar imágenes MNIST
        auto train_images = MNISTDataset::loadImages(dataset_path + "train-images.idx3-ubyte", training_samples);
        auto train_labels = MNISTDataset::loadLabels(dataset_path + "train-labels.idx1-ubyte", training_samples);

        std::cout << "Loaded " << train_images.size() << " training images.\n";

        // Crear red Kohonen 3D
        int sizeX = 10, sizeY = 10, sizeZ = 10;
        int input_dim = 28 * 28;
        
        kohonenNet = new Kohonen3D(sizeX, sizeY, sizeZ, input_dim);

        // Entrenar con imágenes MNIST
        kohonenNet->train(train_images, 1, 0.1f, 3.0f);
        std::cout << "Training completed." << std::endl;

        // Inicializar neuronas para visualización
        initNeurons();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        exit(1);
    }
}

/*
int main(int argc, char** argv) {
    // Inicializar GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(1000, 800);
    glutCreateWindow("Kohonen 3D con MNIST");

    initGL();
    
    // Entrenar la red en un hilo separado o antes de iniciar el bucle principal
    trainAndVisualize();

    // Configurar callbacks
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutKeyboardFunc(keyboard);

    glutMainLoop();
    
    // Limpieza
    if (kohonenNet) delete kohonenNet;
    
    return 0;
}
    */