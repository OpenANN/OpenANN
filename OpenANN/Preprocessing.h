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

/**
 * Apply a (numerically stable) filter (FIR or IIR) on the input signal.
 * @param x input signal
 * @param y output, filtered signal
 * @param b feedforward filter coefficients
 * @param a feedback filter coefficients
 */
void filter(const Mt& x, Mt& y, const Mt& b, const Mt& a);

/**
 * Downsample an input signal.
 * @param y input signal
 * @param d downsampled signal
 * @param downSamplingFactor downsampling factor
 */
void downsample(const Mt& y, Mt& d, int downSamplingFactor);

}
