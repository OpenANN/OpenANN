#ifndef OPENANN_SPARSE_AUTO_ENCODER_H_
#define OPENANN_SPARSE_AUTO_ENCODER_H_

#include <OpenANN/Learner.h>
#include <OpenANN/ActivationFunctions.h>

namespace OpenANN
{

class SparseAutoEncoder : public Learner
{
  int D, H;
  double beta, rho, lambda;
  ActivationFunction act;
  Eigen::MatrixXd W1, W2;
  Eigen::VectorXd b1, b2;
  Eigen::MatrixXd A1, Z1, A2, Z2;
  Eigen::VectorXd parameters, grad;
public:
  /**
   * Sparse auto-encoder.
   * @param D number of inputs
   * @param H number of outputs
   * @param act activation function of the hidden layer
   * @param beta weight of sparsity
   * @param rho desired mean activation of hidden neurons
   * @param lambda L2 norm penalty
   */
  SparseAutoEncoder(int D, int H, double beta, double rho, double lambda,
                    ActivationFunction act);
  virtual Eigen::VectorXd operator()(const Eigen::VectorXd& x);
  virtual Eigen::MatrixXd operator()(const Eigen::MatrixXd& X);
  virtual bool providesInitialization();
  virtual void initialize();
  virtual unsigned int dimension();
  virtual void setParameters(const Eigen::VectorXd& parameters);
  virtual const Eigen::VectorXd& currentParameters();
  virtual double error();
  virtual bool providesGradient();
  virtual Eigen::VectorXd gradient();
  virtual void errorGradient(double& value, Eigen::VectorXd& grad);
  Eigen::MatrixXd getInputWeights();
  Eigen::MatrixXd getOutputWeights();
  Eigen::VectorXd reconstruct(const Eigen::VectorXd& x);
private:
  void pack(Eigen::VectorXd& vector, const Eigen::MatrixXd& W1,
            const Eigen::MatrixXd& W2, const Eigen::VectorXd& b1,
            const Eigen::VectorXd& b2);
  void unpack(const Eigen::VectorXd& vector, Eigen::MatrixXd& W1,
              Eigen::MatrixXd& W2, Eigen::VectorXd& b1, Eigen::VectorXd& b2);
};

} // OpenANN

#endif // OPENANN_SPARSE_AUTO_ENCODER_H_
