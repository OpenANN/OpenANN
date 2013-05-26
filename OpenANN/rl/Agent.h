#pragma once

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

}
