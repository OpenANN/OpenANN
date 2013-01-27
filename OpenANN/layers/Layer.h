#pragma once

#include <Eigen/Dense>
#include <list>
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
  virtual OutputInfo initialize(std::list<fpt*>& parameterPointers, std::list<fpt*>& parameterDerivativePointers) = 0;
  virtual void forwardPropagate(Vt* x, Vt*& y) = 0;
  virtual void backpropagate(Vt* ein, Vt*& eout) = 0;
};

}
