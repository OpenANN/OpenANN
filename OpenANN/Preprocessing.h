#ifndef OPENANN_PREPROCESSING_H_
#define OPENANN_PREPROCESSING_H_

/**
 * @file Preprocessing.h
 *
 * Provides some convenience functions to preprocess data.
 */

#include <Eigen/Dense>

namespace OpenANN
{

/**
 * Scale all values to the interval [min, max].
 * @param data matrix that contains e. g. network inputs or outputs
 * @param min minimum value of the output
 * @param max maximum value of the output
 */
void scaleData(Eigen::MatrixXd& data, double min = -1.0, double max = 1.0);

/**
 * Apply a (numerically stable) filter (FIR or IIR) on the input signal.
 * @param x input signal
 * @param y output, filtered signal
 * @param b feedforward filter coefficients
 * @param a feedback filter coefficients
 */
void filter(const Eigen::MatrixXd& x, Eigen::MatrixXd& y,
            const Eigen::MatrixXd& b, const Eigen::MatrixXd& a);

/**
 * Downsample an input signal.
 * @param y input signal
 * @param d downsampled signal
 * @param downSamplingFactor downsampling factor
 */
void downsample(const Eigen::MatrixXd& y, Eigen::MatrixXd& d, int downSamplingFactor);

/**
 * Extract random patches from a images.
 * @param images each row contains an original image
 * @param channels number of color channels, e.g. 3 for RGB, 1 for grayscale
 * @param rows height of the images
 * @param cols width of the images
 * @param samples number of sampled patches for each image
 * @param patchRows height of image patches
 * @param patchCols width of image patches
 * @return sampled patches, each row represents a patch
 */
Eigen::MatrixXd sampleRandomPatches(const Eigen::MatrixXd& images,
                                    int channels, int rows, int cols,
                                    int samples, int patchRows, int patchCols);

} // namespace OpenANN

#endif // OPENANN_PREPROCESSING_H_
