#ifndef OPENANN_RL_RANDOM_AGENT_H_
#define OPENANN_RL_RANDOM_AGENT_H_

#include <OpenANN/rl/Agent.h>

namespace OpenANN
{

/**
 * @class RandomAgent
 *
 * Choses actions randomly.
 */
class RandomAgent : public Agent
{
  Environment* environment;
  double accumulatedReward;
public:
  double lastReturn;
  RandomAgent();
  virtual void abandoneIn(Environment& environment);
  virtual void chooseAction();
  virtual void chooseOptimalAction();
};

} // namespace OpenANN

#endif // OPENANN_RL_RANDOM_AGENT_H_
