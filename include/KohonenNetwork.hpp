#pragma once

#include <vector>
#include <cmath>
#include <iostream>

using Vector = std::vector<float>;

class Kohonen3D {
public:
    Kohonen3D(int sizeX, int sizeY, int sizeZ, int input_dim);

    void train(const std::vector<Vector>& data, int epochs, float learning_rate_initial, float neighborhood_radius_initial);

    const std::vector<Vector>& getWeights() const;
    int getSizeX() const;
    int getSizeY() const;
    int getSizeZ() const;

private:
    float euclideanDistanceVec(const Vector& a, const Vector& b);
    float euclideanDistance3D(int x1, int y1, int z1, int x2, int y2, int z2);

    int sizeX_, sizeY_, sizeZ_;
    int input_dim_;
    std::vector<Vector> weights_;
};
