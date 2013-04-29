#ifndef COMPRESSOR_H
#define COMPRESSOR_H

#include <Eigen/Dense>

namespace OpenANN
{

/**
 * @class Compressor
 *
 * Compresses arbitrary one-dimensional data.
 *
 * The compression must have the form \f$ \Phi x = y \f$, where \f$ \Phi \f$
 * can be an arbitrary compression matrix, \f$ x \f$ is the D-dimensional
 * input and \f$ y \f$ is the F-dimensional output (i.e. F < D).
 *
 * The compression can be done on a GPU with CUDA. This sometimes speeds the
 * compression up by a factor of 2.
 */
class Compressor
{
  Eigen::MatrixXd cm;
#ifdef CUDA_AVAILABLE
  double* cmOnDevice;
  double* inputOnDevice;
  double* outputOnDevice;
#endif
public:
  Compressor();
  ~Compressor();
  /**
   * Clear compression matrix.
   */
  void reset();
  /**
   * Set the compression matrix.
   * @param cm compression matrix
   */
  void init(const Eigen::MatrixXd& cm);
  /**
   * Compress a signal.
   * @param instance input signal
   * @return compressed signal
   */
  Eigen::VectorXd compress(const Eigen::VectorXd& instance);
};

}

#endif // COMPRESSOR_H
