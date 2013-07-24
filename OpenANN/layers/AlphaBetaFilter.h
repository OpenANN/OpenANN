#ifndef OPENANN_LAYERS_ALPHA_BETA_FILTER_H_
#define OPENANN_LAYERS_ALPHA_BETA_FILTER_H_

#include <OpenANN/layers/Layer.h>
#include <OpenANN/ActivationFunctions.h>

namespace OpenANN
{

/**
 * @class AlphaBetaFilter
 *
 * A recurrent layer that can be used to smooth the input and estimate its
 * derivative.
 *
 * In a partially observable Markov decision process (POMDP), we can use an
 * \f$ \alpha-\beta \f$ filter to smooth noisy observations and estimate the
 * derivatives. We can e.g. estimate the velocities from the positions of an
 * object.
 */
class AlphaBetaFilter : public Layer
{
  int I, J;
  double deltaT;
  double stdDev;
  Eigen::VectorXd gamma;
  Eigen::VectorXd gammad;
  Eigen::VectorXd alpha;
  Eigen::VectorXd beta;
  bool first;
  Eigen::MatrixXd* x;
  Eigen::MatrixXd y;

public:
  AlphaBetaFilter(OutputInfo info, double deltaT, double stdDev);
  virtual OutputInfo initialize(std::vector<double*>& parameterPointers,
                                std::vector<double*>& parameterDerivativePointers);
  virtual void initializeParameters();
  virtual void updatedParameters();
  virtual void reset();
  virtual void forwardPropagate(Eigen::MatrixXd* x, Eigen::MatrixXd*& y,
                                bool dropout);
  virtual void backpropagate(Eigen::MatrixXd* ein, Eigen::MatrixXd*& eout,
                             bool backpropToPrevious);
  virtual Eigen::MatrixXd& getOutput();
  virtual Eigen::VectorXd getParameters();
};

} // namespace OpenANN

#endif // OPENANN_LAYERS_ALPHA_BETA_FILTER_H_
