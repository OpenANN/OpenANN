#ifndef OPENANN_RL_ENVIRONMENT_H_
#define OPENANN_RL_ENVIRONMENT_H_

#include <OpenANN/rl/ActionSpace.h>
#include <OpenANN/rl/StateSpace.h>
#include <Eigen/Dense>

namespace OpenANN
{

/**
 * @class Environment
 *
 * A reinforcement learning environment.
 */
class Environment : public StateSpace, public ActionSpace
{
public:
  virtual ~Environment() {}
  /**
   * Restart environment.
   */
  virtual void restart() = 0;
  /**
   * Get current state.
   * @return state
   */
  virtual const State& getState() const = 0;
  /**
   * Get current action.
   * @return action
   */
  virtual const Action& getAction() const = 0;
  /**
   * Perform an action.
   * @param action next action
   */
  virtual void stateTransition(const Action& action) = 0;
  /**
   * Get reward.
   * @return reward for the last state transition
   */
  virtual double reward() const = 0;
  /**
   * Check for terminal state.
   * @return is the environment in a terminal state?
   */
  virtual bool terminalState() const = 0;
  /**
   * Check if the agent was successful.
   * @return was the agent successful?
   */
  virtual bool successful() const = 0;
  /**
   * Number of steps during the episode.
   */
  virtual int stepsInEpisode() const = 0;
  /**
   * Time between two simulation steps.
   */
  virtual double deltaT() const { return 1.0; }
};

} // namespace OpenANN

#endif // OPENANN_RL_ENVIRONMENT_H_
