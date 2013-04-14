#pragma once

#include <OpenANN/rl/ActionSpace.h>
#include <OpenANN/rl/StateSpace.h>
#include <Eigen/Dense>

namespace OpenANN
{

class Environment : public StateSpace, public ActionSpace
{
public:
  virtual ~Environment() {}
  virtual void restart() = 0;
  virtual const State& getState() const = 0;
  virtual const Action& getAction() const = 0;
  virtual void stateTransition(const Action& action) = 0;
  virtual fpt reward() const = 0;
  virtual bool terminalState() const = 0;
  virtual bool successful() const = 0;
  virtual int stepsInEpisode() const = 0;
  virtual fpt deltaT() const { return 1.0; }
};

}
