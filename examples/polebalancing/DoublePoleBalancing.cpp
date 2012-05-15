#include "DoublePoleBalancing.h"
#include <AssertionMacros.h>
#include <EigenWrapper.h>
#include <cmath>

DoublePoleBalancing::DoublePoleBalancing(bool fullyObservable)
  : fullyObservable(fullyObservable),
    gravity(-9.8),
    cartMass(1.0),
    tau(0.02),
    pole1Mass(0.1),
    pole2Mass(0.01),
    pole1Length(0.5),
    pole2Length(0.05),
    mup(0.000002),
    muc(0.0005),
    initialPoleAngularPosition1(0.07),
    maxCartPosition(2.4),
    maxPoleAngularPosition1(0.62),
    maxPoleAngularPosition2(0.62),
    maxSteps(100000),
    maxForce(10.0),

    step(0)
{
  massLength1 = pole1Mass * pole1Length;
  massLength2 = pole2Mass * pole2Length;

  actionSpaceLo.resize(1);
  actionSpaceLo << -maxForce;
  actionSpaceHi.resize(1);
  actionSpaceHi << maxForce;
  stateSpaceLo.resize(fullyObservable ? 6 : 3);
  StateSpaceHi.resize(fullyObservable ? 6 : 3);
  if(fullyObservable)
  {
    stateSpaceLo << -1.0, -0.1, -1.0, -0.5, -1.0, -0.5;
    StateSpaceHi << 1.0, 0.1, 1.0, 0.5, 1.0, 0.5;
  }
  else
  {
    stateSpaceLo << -1.0, -1.0, -1.0;
    StateSpaceHi << 1.0, 1.0, 1.0;
  }
  stateNormalizationVector.resize(6);
  stateNormalizationVector << 1.0 / maxCartPosition, 1.0 / maxForce,
                              1.0 / maxPoleAngularPosition1, 1.0 / 5.0,
                              1.0 / maxPoleAngularPosition2, 1.0 / 5.0;

  action = actionSpaceLo;
  state.resize(6);
  normalizedState.resize(fullyObservable ? 6 : 3);
}

bool DoublePoleBalancing::actionSpaceContinuous() const
{
  return true;
}

int DoublePoleBalancing::actionSpaceDimension() const
{
  return actionSpaceLo.rows();
}

int DoublePoleBalancing::actionSpaceElements() const
{
  return std::numeric_limits<double>::infinity();
}

const OpenANN::Environment::Action& DoublePoleBalancing::actionSpaceLowerBound() const
{
  return actionSpaceLo;
}

const OpenANN::Environment::Action& DoublePoleBalancing::actionSpaceUpperBound() const
{
  return actionSpaceHi;
}

fpt DoublePoleBalancing::deltaT() const
{
  return tau;
}

const OpenANN::Environment::Action& DoublePoleBalancing::getAction() const
{
  return action;
}

const OpenANN::ActionSpace::A& DoublePoleBalancing::getDiscreteActionSpace() const
{
  OPENANN_CHECK(false);
  static ActionSpace::A dummy;
  return dummy;
}

const OpenANN::StateSpace::S& DoublePoleBalancing::getDiscreteStateSpace() const
{
  OPENANN_CHECK(false);
  static StateSpace::S dummy;
  return dummy;
}

const OpenANN::Environment::State& DoublePoleBalancing::getState() const
{
  return normalizedState;
}

void DoublePoleBalancing::restart()
{
  step = 0;
  state << 0.0, 0.0, initialPoleAngularPosition1, 0.0, 0.0, 0.0;
  normalizeState();
}

fpt DoublePoleBalancing::reward() const
{
  return 1.0;
}

bool DoublePoleBalancing::stateSpaceContinuous() const
{
  return true;
}

int DoublePoleBalancing::stateSpaceDimension() const
{
  return normalizedState.rows();
}

int DoublePoleBalancing::stateSpaceElements() const
{
  return std::numeric_limits<double>::infinity();
}

const OpenANN::Environment::State& DoublePoleBalancing::stateSpaceLowerBound() const
{
  return stateSpaceLo;
}

const OpenANN::Environment::State& DoublePoleBalancing::stateSpaceUpperBound() const
{
  return StateSpaceHi;
}

void DoublePoleBalancing::stateTransition(const Action& action)
{
  OPENANN_CHECK_MATRIX_BROKEN(action);
  step++;

  this->action = action;
  double force = action(0,0);
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

int DoublePoleBalancing::stepsInEpisode() const
{
  return step;
}

bool DoublePoleBalancing::successful() const
{
  return step >= maxSteps;
}

bool DoublePoleBalancing::terminalState() const
{
  return successful() || std::fabs(state(0)) > maxCartPosition
      || std::fabs(state(2)) > maxPoleAngularPosition1
      || std::fabs(state(4)) > maxPoleAngularPosition2;
}

OpenANN::Environment::State DoublePoleBalancing::derivative(const State& s, fpt force)
{
  fpt costheta1 = std::cos(s(2));
  fpt sintheta1 = std::sin(s(2));
  fpt gsintheta1 = gravity * sintheta1;
  fpt costheta2 = std::cos(s(4));
  fpt sintheta2 = std::sin(s(4));
  fpt gsintheta2 = gravity * sintheta2;

  fpt temp1 = mup * s(3) / massLength1;
  fpt temp2 = mup * s(5) / massLength2;

  fpt fi1 = massLength1 * std::pow(s(3), 2.0) * sintheta1
      + 0.75 * pole1Mass * costheta1 * (temp1+gsintheta1);
  fpt fi2 = massLength2 * std::pow(s(5), 2.0) * sintheta2
      + 0.75 * pole2Mass * costheta2 * (temp2+gsintheta2);

  fpt mi1 = pole1Mass * (1.0 - 0.75 * costheta1 * gsintheta1);
  fpt mi2 = pole2Mass * (1.0 - 0.75 * costheta2 * gsintheta2);

  fpt cartVelocityDot = (force - muc * (s(1) == 0.0 ? 0.0 : s(1) / std::fabs(s(1))) + fi1 + fi2)
      / (mi1 + mi2 + cartMass);
  fpt poleAngularVelocity1Dot = -0.75 * (cartVelocityDot * costheta1 + gsintheta1 + temp1) / pole1Length;
  fpt poleAngularVelocity2Dot = -0.75 * (cartVelocityDot * costheta2 + gsintheta2 + temp2) / pole2Length;

  State derivative(6);
  derivative(0) = s(1);
  derivative(1) = cartVelocityDot;
  derivative(2) = s(3);
  derivative(3) = poleAngularVelocity1Dot;
  derivative(4) = s(5);
  derivative(5) = poleAngularVelocity2Dot;
  return derivative;
}

OpenANN::Environment::State DoublePoleBalancing::rk4(const State& s, fpt force, const State& der)
{
  const double TIME_DELTA = 0.01;
  State result(6);

  double hh = TIME_DELTA * 0.5, h6 = TIME_DELTA / 6.0;
  State dym(6);
  State dyt(6);
  State yt(6);

  yt = s + hh * der;
  dyt = derivative(yt, force);
  dyt(0) = yt(1);
  dyt(2) = yt(3);
  dyt(4) = yt(5);
  yt = s + hh * dyt;
  dym = derivative(yt, force);
  dym(0) = yt(1);
  dym(2) = yt(3);
  dym(4) = yt(5);
  yt = s + TIME_DELTA * dym;
  dym += dyt;
  dyt = derivative(yt, force);
  dyt(0) = yt(1);
  dyt(2) = yt(3);
  dyt(4) = yt(5);
  result = s + h6 * (der + dyt + 2.0 * dym);

  return result;
}

void DoublePoleBalancing::normalizeState()
{
  if(fullyObservable)
    normalizedState = state.cwiseProduct(stateNormalizationVector);
  else
  {
    normalizedState(0) = state(0) * stateNormalizationVector(0);
    normalizedState(1) = state(2) * stateNormalizationVector(2);
    normalizedState(2) = state(4) * stateNormalizationVector(4);
  }
}
