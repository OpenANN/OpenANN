#ifndef OPENANN_RL_ENVIRONMENT_H_
#define OPENANN_RL_ENVIRONMENT_H_

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
  virtual double reward() const = 0;
  virtual bool terminalState() const = 0;
  virtual bool successful() const = 0;
  virtual int stepsInEpisode() const = 0;
  virtual double deltaT() const { return 1.0; }
};

} // namespace OpenANN

#endif // OPENANN_RL_ENVIRONMENT_H_
