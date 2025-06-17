#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <map>

using Vector = std::vector<float>;

class Kohonen3D
{
public:
  Kohonen3D(int sizeX, int sizeY, int sizeZ, int input_dim);

  void train(const std::vector<Vector> &data, const std::vector<int> &labels, int epochs, float learning_rate_initial, float neighborhood_radius_initial);

  const std::vector<Vector> &getWeights() const;
  int getSizeX() const;
  int getSizeY() const;
  int getSizeZ() const;
  int findBMU(const Vector &input) const;
  std::vector<int> labelNeurons(const std::vector<Vector> &data, const std::vector<int> &labels);
  int classify(const Vector &input, const std::vector<int> &neuronLabels);
  float computeAccuracy(const std::vector<Vector> &data, const std::vector<int> &true_labels);

  void saveModel(const std::string &filename) const;
  bool loadModel(const std::string &filename);

  // Obtener etiquetas actuales
  const std::unordered_map<int, int> &getNeuronLabels() const { return neuron_to_label; }
  void assignLabels(const std::vector<Vector> &data, const std::vector<int> &labels);

private:
  float euclideanDistanceVec(const Vector &a, const Vector &b) const;
  float euclideanDistance3D(int x1, int y1, int z1, int x2, int y2, int z2) const;
  int sizeX_, sizeY_, sizeZ_;
  int input_dim_;
  std::vector<Vector> weights_;
  std::unordered_map<int, std::map<int, int>> neuron_label_count;
  std::unordered_map<int, int> neuron_to_label;
  std::vector<std::vector<float>> neuron_distances_3d; // Distancias precalculadas entre neuronas
  void precomputeNeuronDistances();
};
