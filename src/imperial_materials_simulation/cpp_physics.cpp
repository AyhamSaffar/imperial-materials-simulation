#include <array>
#include <iostream>
using namespace std;

template<size_t SIZE>
double get_kinetic_energy(array<double, SIZE>& velocities, double mass) {
    double kinetic_energy = 0.0;
    for (double velocity: velocities) {
        kinetic_energy += velocity * velocity;
    };
    kinetic_energy *= mass / 2;
    return kinetic_energy;
}

int main() {
    array<double, 3> velocities = {1.0, 1.0, 1.0};
    cout << get_kinetic_energy(velocities, 1.0);
    return 0;
}
