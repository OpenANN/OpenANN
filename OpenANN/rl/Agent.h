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
  /**
   * Abandon an agent in an environment.
   *
   * @param environment reinforcement learning environment
   */
  virtual void abandoneIn(Environment& environment) = 0;
  /**
   * Choose an action and execute it in the environment.
   *
   * This action might not be optimal, i.e. the agent is allowed to explore.
   */
  virtual void chooseAction() = 0;
  /**
   * Choose an action and execute it in the environment.
   *
   * The action must be optimal, i.e. the agent must exploit the learned
   * policy.
   */
  virtual void chooseOptimalAction() = 0;
};

} // namespace OpenANN

#endif // OPENANN_RL_AGENT_H_
