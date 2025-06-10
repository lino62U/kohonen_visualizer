#include "KohonenNetwork.hpp"
#include <cstdlib>
#include <algorithm>

Kohonen3D::Kohonen3D(int sizeX, int sizeY, int sizeZ, int input_dim)
    : sizeX_(sizeX), sizeY_(sizeY), sizeZ_(sizeZ), input_dim_(input_dim) {
    int total_neurons = sizeX * sizeY * sizeZ;
    weights_.resize(total_neurons, Vector(input_dim));
    for (auto& w : weights_) {
        for (auto& v : w) {
            v = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

void Kohonen3D::train(const std::vector<Vector>& data, int epochs, float learning_rate_initial, float neighborhood_radius_initial) {
    int total_neurons = sizeX_ * sizeY_ * sizeZ_;
    int input_size = input_dim_;

    for (int epoch = 0; epoch < epochs; epoch++) {
        float lr = learning_rate_initial * (1.0f - static_cast<float>(epoch) / epochs);
        float radius = neighborhood_radius_initial * (1.0f - static_cast<float>(epoch) / epochs);

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

const std::vector<Vector>& Kohonen3D::getWeights() const {
    return weights_;
}

int Kohonen3D::getSizeX() const { return sizeX_; }
int Kohonen3D::getSizeY() const { return sizeY_; }
int Kohonen3D::getSizeZ() const { return sizeZ_; }

float Kohonen3D::euclideanDistanceVec(const Vector& a, const Vector& b) {
    float dist = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return std::sqrt(dist);
}

float Kohonen3D::euclideanDistance3D(int x1, int y1, int z1, int x2, int y2, int z2) {
    return std::sqrt((x1 - x2)*(x1 - x2) +
                     (y1 - y2)*(y1 - y2) +
                     (z1 - z2)*(z1 - z2));
}
