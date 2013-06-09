#ifndef OPENANN_RL_AGENT_H_
#define OPENANN_RL_AGENT_H_

#include <OpenANN/rl/Environment.h>

namespace OpenANN
{

/**
 * @class Agent
 *
 * A (learning) agent in a reinforcement learning problem.
 */
class Agent
{
public:
  virtual ~Agent() {}
  virtual void abandoneIn(Environment& environment) = 0;
  virtual void chooseAction() = 0;
  virtual void chooseOptimalAction() = 0;
};

} // namespace OpenANN

#endif // OPENANN_RL_AGENT_H_
