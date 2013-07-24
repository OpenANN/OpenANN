#ifndef OPENANN_INTRINSIC_PLASTICITY_H_
#define OPENANN_INTRINSIC_PLASTICITY_H_

#include <OpenANN/Learner.h>
#include <OpenANN/layers/Layer.h>
#include <OpenANN/io/DataSet.h>
#include <Eigen/Dense>

namespace OpenANN
{

/**
 * @class IntrinsicPlasticity
 *
 * Learns the parameters of a logistic sigmoid activation function.
 *
 * Activation functions of the form \f$ y = 1 / (1 + \exp (-s a - b)) \f$
 * with slopes \f$ s \f$ and biases \f$ b \f$ are adapted such that the output
 * distribution is approximately exponential with mean \f$ \mu \f$ and with
 * respect to a input distribution given by a training set. This procedure
 * prevents saturation. Note that changing the incoming weights might require
 * readjustment.
 *
 * [1] Jochen Triesch:
 * A Gradient Rule for the Plasticity of a Neuron’s Intrinsic Excitability,
 * Proceedings of the International Conference on Artificial Neural Networks,
 * pp. 1–7, 2005.
 *
 * [2] Jochen Triesch:
 * Synergies between intrinsic and synaptic plasticity mechanisms,
 * Neural Computation 19, pp. 885-909, 2007.
 */
class IntrinsicPlasticity : public Learner, public Layer
{
  const int nodes;
  const double mu;
  const double stdDev;
  Eigen::VectorXd s;
  Eigen::VectorXd b;
  Eigen::VectorXd parameters;
  Eigen::VectorXd g;
  Eigen::VectorXd y;
  Eigen::MatrixXd Y;
  Eigen::MatrixXd Yd;
  Eigen::MatrixXd e;
public:
  IntrinsicPlasticity(int nodes, double mu, double stdDev = 1.0);

  virtual unsigned int examples();
  virtual unsigned int dimension();
  virtual bool providesInitialization();
  virtual void initialize();
  virtual double error();
  virtual double error(unsigned int n);
  virtual const Eigen::VectorXd& currentParameters();
  virtual void setParameters(const Eigen::VectorXd& parameters);
  virtual bool providesGradient();
  virtual Eigen::VectorXd gradient();
  virtual Eigen::VectorXd gradient(unsigned int n);
  virtual Eigen::VectorXd operator()(const Eigen::VectorXd& a);
  virtual Eigen::MatrixXd operator()(const Eigen::MatrixXd& A);

  virtual OutputInfo initialize(std::vector<double*>& parameterPointers,
                                std::vector<double*>& parameterDerivativePointers);
  virtual void initializeParameters() {}
  virtual void updatedParameters() {}
  virtual void forwardPropagate(Eigen::MatrixXd* x, Eigen::MatrixXd*& y,
                                bool dropout);
  virtual void backpropagate(Eigen::MatrixXd* ein, Eigen::MatrixXd*& eout,
                             bool backpropToPrevious);
  virtual Eigen::MatrixXd& getOutput();
  virtual Eigen::VectorXd getParameters();
};

} // namespace OpenANN

#endif // OPENANN_INTRINSIC_PLASTICITY_H_
