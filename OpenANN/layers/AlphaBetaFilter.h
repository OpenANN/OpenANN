#pragma once

#include <OpenANN/layers/Layer.h>
#include <OpenANN/ActivationFunctions.h>

namespace OpenANN {

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
  virtual void backpropagate(Eigen::MatrixXd* ein, Eigen::MatrixXd*& eout);
  virtual Eigen::MatrixXd& getOutput();
};

}
