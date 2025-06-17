#include "KohonenNetwork.hpp"
#include <cstdlib>
#include <algorithm>
#include <omp.h>
#include <random>
#include <fstream>
#include <sstream>
#include <iomanip>

Kohonen3D::Kohonen3D(int sizeX, int sizeY, int sizeZ, int input_dim)
    : sizeX_(sizeX), sizeY_(sizeY), sizeZ_(sizeZ), input_dim_(input_dim),
      neuron_distances_3d(sizeX * sizeY * sizeZ, std::vector<float>(sizeX * sizeY * sizeZ))
{
  const int total_neurons = sizeX_ * sizeY_ * sizeZ_;
  weights_.resize(total_neurons, Vector(input_dim_));

  std::mt19937 gen(42); // Semilla diferente por hilo
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  for (int i = 0; i < total_neurons; ++i)
  {
    for (int j = 0; j < input_dim_; ++j)
    {
      weights_[i][j] = dist(gen);
    }
  }

  precomputeNeuronDistances();
}

void Kohonen3D::precomputeNeuronDistances()
{
  const int total_neurons = sizeX_ * sizeY_ * sizeZ_;

#pragma omp parallel for collapse(2)
  for (int i = 0; i < total_neurons; ++i)
  {
    for (int j = 0; j < total_neurons; ++j)
    {
      // Coordenadas 3D para neurona i
      const int x1 = i / (sizeY_ * sizeZ_);
      const int y1 = (i / sizeZ_) % sizeY_;
      const int z1 = i % sizeZ_;

      // Coordenadas 3D para neurona j
      const int x2 = j / (sizeY_ * sizeZ_);
      const int y2 = (j / sizeZ_) % sizeY_;
      const int z2 = j % sizeZ_;

      // Distancia euclidiana al cuadrado
      neuron_distances_3d[i][j] =
          (x1 - x2) * (x1 - x2) +
          (y1 - y2) * (y1 - y2) +
          (z1 - z2) * (z1 - z2);
    }
  }
}


void Kohonen3D::train(const std::vector<Vector> &data, const std::vector<int> &labels,
                      int epochs, float learning_rate_initial, float neighborhood_radius_initial)
{
    const int total_neurons = sizeX_ * sizeY_ * sizeZ_;
    const int input_size = input_dim_;

    std::cout << "========== Entrenamiento Kohonen ==========\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Total épocas: " << epochs << ", Neuronas: " << total_neurons << ", Dimensión entrada: " << input_size << "\n\n";

    std::cout << std::left
              << std::setw(12) << "Epoch"
              << std::setw(15) << "Accuracy (%)"
              << std::setw(15) << "LR"
              << std::setw(15) << "Radius"
              << "\n";

    std::cout << std::string(57, '-') << "\n";

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        const float lr = learning_rate_initial * (1.0f - static_cast<float>(epoch) / epochs);
        const float radius = neighborhood_radius_initial * std::exp(-static_cast<float>(epoch) / (epochs / 2.0f));
        const float radius_sq = radius * radius;

#pragma omp parallel for schedule(dynamic)
        for (int idx = 0; idx < static_cast<int>(data.size()); ++idx)
        {
            const auto &input = data[idx];
            const int winner_idx = findBMU(input);

            for (int i = 0; i < total_neurons; i++)
            {
                const float dist_to_winner_sq = neuron_distances_3d[i][winner_idx];
                if (dist_to_winner_sq <= radius_sq)
                {
                    const float h = std::exp(-dist_to_winner_sq / (2 * radius_sq));
                    for (int j = 0; j < input_size; j++)
                    {
#pragma omp atomic
                        weights_[i][j] += lr * h * (input[j] - weights_[i][j]);
                    }
                }
            }
        }

        float accuracy = computeAccuracy(data, labels);

        std::ostringstream epoch_str;
        epoch_str << "[" << epoch + 1 << "/" << epochs << "]";

        std::cout << std::setw(12) << epoch_str.str()
                  << std::setw(15) << accuracy * 100.0f
                  << std::setw(15) << lr
                  << std::setw(15) << radius
                  << "\n";
    }

    std::cout << std::string(57, '-') << "\n";
    std::cout << "Entrenamiento finalizado. Asignando etiquetas a las neuronas...\n";

    assignLabels(data, labels);
}

int Kohonen3D::findBMU(const Vector &input) const
{
  int total_neurons = sizeX_ * sizeY_ * sizeZ_;
  int best_idx = 0;
  float min_dist = euclideanDistanceVec(input, weights_[0]);

  for (int i = 1; i < total_neurons; ++i)
  {
    float dist = euclideanDistanceVec(input, weights_[i]);
    if (dist < min_dist)
    {
      min_dist = dist;
      best_idx = i;
    }
  }
  return best_idx;
}

const std::vector<Vector> &Kohonen3D::getWeights() const
{
  return weights_;
}

int Kohonen3D::getSizeX() const { return sizeX_; }
int Kohonen3D::getSizeY() const { return sizeY_; }
int Kohonen3D::getSizeZ() const { return sizeZ_; }

float Kohonen3D::euclideanDistanceVec(const Vector &a, const Vector &b) const
{
  float dist = 0.0f;
  for (size_t i = 0; i < a.size(); i++)
  {
    float diff = a[i] - b[i];
    dist += diff * diff;
  }
  return std::sqrt(dist);
}

float Kohonen3D::euclideanDistance3D(int x1, int y1, int z1, int x2, int y2, int z2) const
{
  return std::sqrt((x1 - x2) * (x1 - x2) +
                   (y1 - y2) * (y1 - y2) +
                   (z1 - z2) * (z1 - z2));
}

// Add to Kohonen3D class
void Kohonen3D::assignLabels(const std::vector<Vector> &data, const std::vector<int> &labels) {
    const int total_neurons = sizeX_ * sizeY_ * sizeZ_;
    std::vector<std::map<int, int>> label_counts(total_neurons); // Map of label -> count for each neuron

    // Count how many times each neuron is the BMU for each label
    for (size_t i = 0; i < data.size(); ++i) {
        int bmu = findBMU(data[i]);
        label_counts[bmu][labels[i]]++;
    }

    // Assign the most frequent label to each neuron
    neuron_to_label.clear();
    for (int i = 0; i < total_neurons; ++i) {
        if (!label_counts[i].empty()) {
            auto max_it = std::max_element(
                label_counts[i].begin(), label_counts[i].end(),
                [](const auto &a, const auto &b) { return a.second < b.second; }
            );
            neuron_to_label[i] = max_it->first;
        }
    }
}

std::vector<int> Kohonen3D::labelNeurons(const std::vector<Vector> &data, const std::vector<int> &labels)
{
  std::vector<std::unordered_map<int, int>> labelCounts(sizeX_ * sizeY_ * sizeZ_);

  // Contar etiquetas por neurona
  for (size_t idx = 0; idx < data.size(); ++idx)
  {
    int bmu = findBMU(data[idx]);
    labelCounts[bmu][labels[idx]]++;
  }

  // Asignar etiqueta mayoritaria
  std::vector<int> neuronLabels(sizeX_ * sizeY_ * sizeZ_, -1);
  for (int i = 0; i < neuronLabels.size(); ++i)
  {
    if (!labelCounts[i].empty())
    {
      auto max_label = std::max_element(
          labelCounts[i].begin(), labelCounts[i].end(),
          [](const auto &a, const auto &b)
          { return a.second < b.second; });
      neuronLabels[i] = max_label->first;
    }
  }

  return neuronLabels;
}

int Kohonen3D::classify(const Vector &input, const std::vector<int> &neuronLabels)
{
  int bmu = findBMU(input);
  auto it = neuron_to_label.find(bmu);
  if (it != neuron_to_label.end())
  {
    return it->second; // Return the label from neuron_to_label
  }
  // Fallback to neuronLabels if neuron_to_label is missing
  if (bmu < neuronLabels.size())
  {
    return neuronLabels[bmu];
  }
  std::cerr << "Warning: No label found for BMU " << bmu << std::endl;
  return -1; // Indicate no valid label
}

float Kohonen3D::computeAccuracy(const std::vector<Vector> &data, const std::vector<int> &true_labels)
{
  if (data.size() != true_labels.size())
    return 0.0f;

  // Paso 1: Asignar etiquetas temporales a las neuronas (usando los datos actuales)
  std::vector<int> temp_neuron_labels = labelNeurons(data, true_labels);

  // Paso 2: Contar predicciones correctas
  int correct = 0;
  for (size_t i = 0; i < data.size(); ++i)
  {
    int bmu = findBMU(data[i]);
    if (temp_neuron_labels[bmu] == true_labels[i])
    {
      correct++;
    }
  }

  return static_cast<float>(correct) / data.size();
}
void Kohonen3D::saveModel(const std::string &filename) const
{
  std::ofstream outfile(filename);
  if (!outfile.is_open())
  {
    std::cerr << "Error al abrir el archivo para escritura: " << filename << std::endl;
    return;
  }

  // Save metadata
  outfile << sizeX_ << " " << sizeY_ << " " << sizeZ_ << " " << input_dim_ << "\n";

  // Save weights
  for (const auto &neuron_weights : weights_)
  {
    for (float weight : neuron_weights)
    {
      outfile << weight << " ";
    }
    outfile << "\n";
  }

  // Save labels
  outfile << "LABELS\n";
  if (neuron_to_label.empty())
  {
    std::cerr << "Warning: No neuron labels to save!" << std::endl;
  }
  for (const auto &[neuron_idx, label] : neuron_to_label)
  {
    outfile << neuron_idx << " " << label << "\n";
  }

  outfile.close();
  std::cout << "Modelo guardado en: " << filename << std::endl;
}

bool Kohonen3D::loadModel(const std::string &filename)
{
  std::ifstream infile(filename);
  if (!infile.is_open())
  {
    std::cerr << "Error al abrir el archivo para lectura: " << filename << std::endl;
    return false;
  }

  // Read metadata
  int x, y, z, dim;
  infile >> x >> y >> z >> dim;
  if (x != sizeX_ || y != sizeY_ || z != sizeZ_ || dim != input_dim_)
  {
    std::cerr << "Dimensiones del modelo no coinciden!" << std::endl;
    infile.close();
    return false;
  }

  // Read weights
  for (auto &neuron_weights : weights_)
  {
    for (float &weight : neuron_weights)
    {
      if (!(infile >> weight))
      {
        std::cerr << "Error reading weights!" << std::endl;
        infile.close();
        return false;
      }
    }
  }

  // Read labels
  std::string line;
  std::getline(infile, line); // Clear newline after weights
  neuron_to_label.clear();
  if (std::getline(infile, line) && line == "LABELS")
  {
    int neuron_idx, label;
    while (infile >> neuron_idx >> label)
    {
      neuron_to_label[neuron_idx] = label;
    }
    if (neuron_to_label.empty())
    {
      std::cerr << "Warning: No labels loaded from file!" << std::endl;
    }
  }
  else
  {
    std::cerr << "Warning: No LABELS section found in file!" << std::endl;
  }

  infile.close();
  std::cout << "Modelo cargado desde: " << filename << std::endl;
  return true;
}