#pragma once

#include <Eigen/Dense>

class AlphaBetaFilter
{
  fpt gamma;
  fpt alpha;
  fpt beta;
  fpt deltaT;

  fpt estimatedPosition;
  fpt smoothedPosition;
  fpt smoothedVelocity;
  bool first;
public:
  AlphaBetaFilter()
    : gamma(0.0), alpha(0.0), beta(0.0), deltaT(0.0), first(true)
  {
  }

  void setDeltaT(fpt deltaT)
  {
    this->deltaT = deltaT;
  }

  void restart();

  void random();

  void setParameter(fpt gamma);

  fpt currentParameter() const
  {
    return gamma;
  }

  Vt operator()(fpt in);
};