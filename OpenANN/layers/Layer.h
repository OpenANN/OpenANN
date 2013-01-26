#pragma once

#include <Eigen/Dense>
#include <list>

namespace OpenANN {

class Layer
{
public:
  virtual void initialize(std::list<fpt*>& parameterPointers, std::list<fpt*>& parameterDerivativePointers) = 0;
  virtual void forwardPropagate(Vt* x, Vt*& y) = 0;
  virtual void accumulate(Vt* e) = 0;
  virtual void gradient() = 0;
  virtual void backpropagate(Vt*& e) = 0;
};

}
