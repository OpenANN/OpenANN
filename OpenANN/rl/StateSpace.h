#pragma once

#include <Eigen/Dense>
#include <vector>

namespace OpenANN
{

class StateSpace
{
public:
  typedef Vt State;
  typedef std::vector<State> S;
  virtual ~StateSpace() {}
  virtual int stateSpaceDimension() const = 0;
  virtual bool stateSpaceContinuous() const = 0;
  virtual int stateSpaceElements() const = 0;
  virtual const State& stateSpaceLowerBound() const = 0;
  virtual const State& stateSpaceUpperBound() const = 0;
  virtual const S& getDiscreteStateSpace() const = 0;
};

}
