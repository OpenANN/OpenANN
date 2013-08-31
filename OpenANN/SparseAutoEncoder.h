#ifndef OPENANN_SPARSE_AUTO_ENCODER_H_
#define OPENANN_SPARSE_AUTO_ENCODER_H_

#include <OpenANN/Learner.h>
#include <OpenANN/ActivationFunctions.h>
#include <OpenANN/layers/Layer.h>

namespace OpenANN
{

/**
 * @class SparseAutoEncoder
 *
 * A sparse auto-encoder tries to reconstruct the inputs from a compressed
 * representation. Its objective function includes a penalty term for the
 * distance to the desired mean activation of the hidden nodes as well as the
 * reconstruction error. Sparse auto-encoders (SAEs) can be used to train
 * multiple layers of feature detectors unsupervised.
 */
class SparseAutoEncoder : public Learner, public Layer
{
  int D, H;
  double beta, rho, lambda;
  ActivationFunction act;
  Eigen::MatrixXd X;
  Eigen::MatrixXd W1, W2, W1d, W2d;
  Eigen::VectorXd b1, b2, b1d, b2d;
  Eigen::MatrixXd A1, Z1, G1D, A2, Z2, G2D;
  Eigen::VectorXd parameters, grad;
  Eigen::MatrixXd dEdZ2, dEdZ1;
  Eigen::VectorXd meanActivation;
public:
  /**
   * Sparse auto-encoder.
   * @param D number of inputs
   * @param H number of outputs
   * @param beta weight of sparsity
   * @param rho desired mean activation of hidden neurons
   * @param lambda L2 norm penalty
   * @param act activation function of the hidden layer
   */
  SparseAutoEncoder(int D, int H, double beta, double rho, double lambda,
                    ActivationFunction act);

  // Learner interface
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
  virtual Learner& trainingSet(DataSet& trainingSet);

  // Layer interface
  virtual void forwardPropagate(Eigen::MatrixXd* x, Eigen::MatrixXd*& y,
                                bool dropout);
  virtual void backpropagate(Eigen::MatrixXd* ein, Eigen::MatrixXd*& eout,
                             bool backpropToPrevious, double& error);
  virtual Eigen::MatrixXd& getOutput();
  virtual Eigen::VectorXd getParameters();
  virtual OutputInfo initialize(std::vector<double*>& parameterPointers,
                                std::vector<double*>& parameterDerivativePointers);
  virtual void initializeParameters();
  virtual void updatedParameters() {}

  // SAE interface
  Eigen::MatrixXd getInputWeights();
  Eigen::MatrixXd getOutputWeights();
  Eigen::VectorXd reconstruct(const Eigen::VectorXd& x);
};

} // OpenANN

#endif // OPENANN_SPARSE_AUTO_ENCODER_H_
