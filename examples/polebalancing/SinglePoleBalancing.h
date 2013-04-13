#pragma once

#include <OpenANN/rl/Environment.h>
#include <Eigen/Dense>

/**
 * @class SinglePoleBalancing
 *
 * In this environment the agent has to control a cart such that the pole
 * mounted at the top of the cart does not pass a defined angle threshold. A
 * more difficult version of this benchmark is DoublePoleBalancing.
 */
class SinglePoleBalancing : public OpenANN::Environment
{
  //! With velocities?
  bool fullyObservable;
  //! The gravity force. Benchmark default "-9.8".
  double gravity;
  //! The mass of the cart. Benchmark default "1.0".
  double cartMass;
  //! The time step between two commands of the agent. Benchmark default "0.02".
  double tau;
  //! The mass of pole 1. Benchmark default "0.1".
  double pole1Mass;
  //! The length of pole 1. Benchmark default "0.5".
  double pole1Length;
  //! Coefficient of friction of the poles' hinges. Benchmark default "0.000002".
  double mup;
  //! Coefficient that controls friction. Benchmark default "0.0005".
  double muc;
  //! Initial angle of pole 1. Benchmark default "4.0*pi/180".
  double initialPoleAngularPosition1;
  /**
   * The maximal distance the cart is allowed to move away from its start
   * position. Benchmark default "2.4"
   */
  double maxCartPosition;
  //! Maximal angle pole 1 is allowed to take on. Benchmark default "36.0*pi/180".
  double maxPoleAngularPosition1;
  //! The number of steps the agent must balance the poles. Benchmark default "100000".
  int maxSteps;
  //! Maximal applicable force.
  double maxForce;

  fpt massLength1;
  fpt massLength2;

  Environment::Action actionSpaceLo;
  Environment::Action actionSpaceHi;
  Environment::State stateSpaceLo;
  Environment::State StateSpaceHi;
  //! The vector used for normalization of the state for the agent.
  Environment::State stateNormalizationVector;

  Environment::Action action;
  Environment::State state;
  Environment::State normalizedState;
  int step;
public:
  SinglePoleBalancing(bool fullyObservable = true);
  virtual bool actionSpaceContinuous() const;
  virtual int actionSpaceDimension() const;
  virtual int actionSpaceElements() const;
  virtual const Action& actionSpaceLowerBound() const;
  virtual const Action& actionSpaceUpperBound() const;
  virtual fpt deltaT() const;
  virtual const Action& getAction() const;
  virtual const ActionSpace::A& getDiscreteActionSpace() const;
  virtual const StateSpace::S& getDiscreteStateSpace() const;
  virtual const State& getState() const;
  virtual void restart();
  virtual fpt reward() const;
  virtual bool stateSpaceContinuous() const;
  virtual int stateSpaceDimension() const;
  virtual int stateSpaceElements() const;
  virtual const State& stateSpaceLowerBound() const;
  virtual const State& stateSpaceUpperBound() const;
  virtual void stateTransition(const Action& action);
  virtual int stepsInEpisode() const;
  virtual bool successful() const;
  virtual bool terminalState() const;
private:
  void normalizeState();
  State derivative(const State& state, fpt force);
  State rk4(const State& state, fpt force, const State& derivative);
};
