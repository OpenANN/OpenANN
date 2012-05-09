#pragma once

#include <rl/Agent.h>

namespace OpenANN
{

class RandomAgent : public Agent
{
  Environment* environment;
  fpt accumulatedReward;
public:
  fpt lastReturn;
  RandomAgent();
  virtual void abandoneIn(Environment& environment);
  virtual void chooseAction();
  virtual void chooseOptimalAction();
};

}
