#pragma once

#include <Eigen/Dense>

namespace OpenANN {

/**
 * Scale all values to the interval [min, max].
 * @param data matrix that contains e. g. network inputs or outputs
 * @param min minimum value of the output
 * @param max maximum value of the output
 */
void scaleData(Mt& data, fpt min = (fpt) -1, fpt max = (fpt) 1);

}
