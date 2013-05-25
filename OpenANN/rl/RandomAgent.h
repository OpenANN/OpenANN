#pragma once

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

}
