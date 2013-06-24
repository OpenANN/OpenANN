#ifndef OPENANN_SPARSE_AUTO_ENCODER_H_
#define OPENANN_SPARSE_AUTO_ENCODER_H_

#include <OpenANN/Learner.h>
#include <OpenANN/Net.h>

namespace OpenANN
{

class SparseAutoEncoder : public Net
{
  int D, H;
  double beta, rho;
public:
  /**
   * Sparse auto-encoder.
   * @param D number of inputs
   * @param H number of outputs
   * @param act activation function of the hidden layer
   * @param beta weight of sparsity
   * @param rho desired mean activation of hidden neurons
   */
  SparseAutoEncoder(int D, int H, ActivationFunction act, double beta,
                    double rho);
  virtual Eigen::VectorXd operator()(const Eigen::VectorXd& x);
  virtual Eigen::MatrixXd operator()(const Eigen::MatrixXd& X);
  virtual void errorGradient(double& value, Eigen::VectorXd& grad);
};

} // OpenANN

#endif // OPENANN_SPARSE_AUTO_ENCODER_H_
