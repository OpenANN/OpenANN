#include "SinglePoleBalancing.h"
#include <OpenANN/util/AssertionMacros.h>
#include <cmath>

SinglePoleBalancing::SinglePoleBalancing(bool fullyObservable)
  : fullyObservable(fullyObservable),
    gravity(-9.8),
    cartMass(1.0),
    tau(0.02),
    pole1Mass(0.1),
    pole1Length(0.5),
    mup(0.000002),
    muc(0.0005),
    initialPoleAngularPosition1(0.07),
    maxCartPosition(2.4),
    maxPoleAngularPosition1(0.62),
    maxSteps(100000),
    maxForce(10.0),

    state(4),
    normalizedState(fullyObservable ? 4 : 2),
    step(0)
{
  massLength1 = pole1Mass * pole1Length;

  actionSpaceLo.resize(1);
  actionSpaceLo << -maxForce;
  actionSpaceHi.resize(1);
  actionSpaceHi << maxForce;
  stateSpaceLo.resize(4);
  stateSpaceLo << -1.0, -0.1, -1.0, -0.5;
  StateSpaceHi.resize(4);
  StateSpaceHi << 1.0, 0.1, 1.0, 0.5;
  stateNormalizationVector.resize(4);
  stateNormalizationVector << 1.0 / maxCartPosition, 1.0 / maxForce,
                           1.0 / maxPoleAngularPosition1, 1.0 / 5.0;

  action = actionSpaceLo;
}

bool SinglePoleBalancing::actionSpaceContinuous() const
{
  return true;
}

int SinglePoleBalancing::actionSpaceDimension() const
{
  return actionSpaceLo.rows();
}

int SinglePoleBalancing::actionSpaceElements() const
{
  return std::numeric_limits<double>::infinity();
}

const OpenANN::Environment::Action& SinglePoleBalancing::actionSpaceLowerBound() const
{
  return actionSpaceLo;
}

const OpenANN::Environment::Action& SinglePoleBalancing::actionSpaceUpperBound() const
{
  return actionSpaceHi;
}

double SinglePoleBalancing::deltaT() const
{
  return tau;
}

const OpenANN::Environment::Action& SinglePoleBalancing::getAction() const
{
  return action;
}

const OpenANN::ActionSpace::A& SinglePoleBalancing::getDiscreteActionSpace() const
{
  OPENANN_CHECK(false);
  static ActionSpace::A dummy;
  return dummy;
}

const OpenANN::StateSpace::S& SinglePoleBalancing::getDiscreteStateSpace() const
{
  OPENANN_CHECK(false);
  static StateSpace::S dummy;
  return dummy;
}

const OpenANN::Environment::State& SinglePoleBalancing::getState() const
{
  return normalizedState;
}

void SinglePoleBalancing::restart()
{
  step = 0;
  state << 0.0, 0.0, initialPoleAngularPosition1, 0.0;
  normalizeState();
}

double SinglePoleBalancing::reward() const
{
  return 1.0;
}

bool SinglePoleBalancing::stateSpaceContinuous() const
{
  return true;
}

int SinglePoleBalancing::stateSpaceDimension() const
{
  return normalizedState.rows();
}

int SinglePoleBalancing::stateSpaceElements() const
{
  return std::numeric_limits<double>::infinity();
}

const OpenANN::Environment::State& SinglePoleBalancing::stateSpaceLowerBound() const
{
  return stateSpaceLo;
}

const OpenANN::Environment::State& SinglePoleBalancing::stateSpaceUpperBound() const
{
  return StateSpaceHi;
}

void SinglePoleBalancing::stateTransition(const Action& action)
{
  step++;

  this->action = action;
  double force = action(0, 0);
  if(fabs(force) > maxForce)
    force = force / fabs(force) * maxForce;

  State s = state;
  for(int i = 0; i < 2; i++)
  {
    State der = derivative(s, force);
    s = rk4(s, force, der);
  }
  state = s;
  normalizeState();
}

int SinglePoleBalancing::stepsInEpisode() const
{
  return step;
}

bool SinglePoleBalancing::successful() const
{
  return step >= maxSteps;
}

bool SinglePoleBalancing::terminalState() const
{
  return successful() || std::fabs(state(0)) > maxCartPosition
         || std::fabs(state(2)) > maxPoleAngularPosition1;
}

OpenANN::Environment::State SinglePoleBalancing::derivative(const State& s, double force)
{
  double costheta1 = std::cos(s(2));
  double sintheta1 = std::sin(s(2));
  double gsintheta1 = gravity * sintheta1;

  double temp1 = mup * s(3) / massLength1;

  double fi1 = massLength1 * std::pow(s(3), 2.0) * sintheta1
               + 0.75 * pole1Mass * costheta1 * (temp1 + gsintheta1);

  double mi1 = pole1Mass * (1.0 - 0.75 * costheta1 * gsintheta1);

  double cartVelocityDot = (force - muc * (s(1) == 0.0 ? 0.0 : s(1) / std::fabs(s(1))) + fi1)
                           / (mi1 + cartMass);
  double poleAngularVelocity1Dot = -0.75 * (cartVelocityDot * costheta1 + gsintheta1 + temp1) / pole1Length;

  State derivative(4);
  derivative(0) = s(1);
  derivative(1) = cartVelocityDot;
  derivative(2) = s(3);
  derivative(3) = poleAngularVelocity1Dot;
  return derivative;
}

OpenANN::Environment::State SinglePoleBalancing::rk4(const State& s, double force, const State& der)
{
  const double TIME_DELTA = 0.01;
  State result(4);

  double hh = TIME_DELTA * 0.5, h6 = TIME_DELTA / 6.0;
  State dym(4);
  State dyt(4);
  State yt(4);

  yt = s + hh * der;
  dyt = derivative(yt, force);
  dyt(0) = yt(1);
  dyt(2) = yt(3);
  yt = s + hh * dyt;
  dym = derivative(yt, force);
  dym(0) = yt(1);
  dym(2) = yt(3);
  yt = s + TIME_DELTA * dym;
  dym += dyt;
  dyt = derivative(yt, force);
  dyt(0) = yt(1);
  dyt(2) = yt(3);
  result = s + h6 * (der + dyt + 2.0 * dym);

  return result;
}

void SinglePoleBalancing::normalizeState()
{
  if(fullyObservable)
    normalizedState = state.cwiseProduct(stateNormalizationVector);
  else
  {
    normalizedState(0) = state(0) * stateNormalizationVector(0);
    normalizedState(1) = state(2) * stateNormalizationVector(2);
  }
}
