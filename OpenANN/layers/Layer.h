#pragma once

#include <Eigen/Dense>
#include <vector>

namespace OpenANN {

class OutputInfo
{
public:
  bool bias;
  std::vector<int> dimensions;

  int outputs();
};

class Layer
{
public:
  virtual OutputInfo initialize(std::vector<fpt*>& parameterPointers, std::vector<fpt*>& parameterDerivativePointers) = 0;
  virtual void forwardPropagate(Vt* x, Vt*& y) = 0;
  virtual void backpropagate(Vt* ein, Vt*& eout) = 0;
};

}
