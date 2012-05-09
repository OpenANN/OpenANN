#pragma once

#include <Eigen/Dense>
#include <vector>

namespace OpenANN
{

class ActionSpace
{
public:
  typedef Vt Action;
  typedef std::vector<Action> A;
  virtual int actionSpaceDimension() const = 0;
  virtual bool actionSpaceContinuous() const = 0;
  virtual int actionSpaceElements() const = 0;
  virtual const Action& actionSpaceLowerBound() const = 0;
  virtual const Action& actionSpaceUpperBound() const = 0;
  virtual const A& getDiscreteActionSpace() const = 0;
};

}
