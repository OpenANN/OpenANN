#ifndef COMPRESSOR_H
#define COMPRESSOR_H

#include <Eigen/Dense>

namespace OpenANN
{

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
  void reset();
  void init(const Eigen::MatrixXd& cm);
  Eigen::VectorXd compress(const Eigen::VectorXd& instance);
};

}

#endif // COMPRESSOR_H
