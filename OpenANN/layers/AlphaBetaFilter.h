#pragma once

#include <OpenANN/layers/Layer.h>
#include <OpenANN/ActivationFunctions.h>

namespace OpenANN {

class AlphaBetaFilter : public Layer
{
  int I, J;
  double deltaT;
  bool bias;
  double stdDev;
  Eigen::VectorXd gamma;
  Eigen::VectorXd gammad;
  Eigen::VectorXd alpha;
  Eigen::VectorXd beta;
  bool first;
  Eigen::VectorXd* x;
  Eigen::VectorXd y;

public:
  AlphaBetaFilter(OutputInfo info, double deltaT, bool bias, double stdDev);
  virtual OutputInfo initialize(std::vector<double*>& parameterPointers, std::vector<double*>& parameterDerivativePointers);
  virtual void initializeParameters();
  virtual void updatedParameters();
  virtual void reset();
  virtual void forwardPropagate(Eigen::VectorXd* x, Eigen::VectorXd*& y, bool dropout);
  virtual void backpropagate(Eigen::VectorXd* ein, Eigen::VectorXd*& eout);
  virtual Eigen::VectorXd& getOutput();
};

}
