#include "AlphaBetaFilter.h"
#include <Random.h>
#include <AssertionMacros.h>
#include <EigenWrapper.h>

void AlphaBetaFilter::restart()
{
  estimatedPosition = 0.0;
  smoothedVelocity = 0.0;
  first = true;
}

void AlphaBetaFilter::random()
{
  OpenANN::RandomNumberGenerator rng;
  setParameter(rng.sampleNormalDistribution<fpt>()*5.0);
}

void AlphaBetaFilter::setParameter(fpt g)
{
  OPENANN_CHECK_INF_AND_NAN(g);
  gamma = fabs(g);
  OPENANN_CHECK_INF_AND_NAN(gamma);
  const fpt r = (4.0 + gamma - sqrt(8.0 * gamma + gamma * gamma)) / 4.0;
  OPENANN_CHECK_INF_AND_NAN(r);
  alpha = 1.0 - r*r;
  const fpt rr = 1.0 - r;
  beta = 2.0 * rr * rr;
  OPENANN_CHECK_INF_AND_NAN(alpha);
  OPENANN_CHECK_INF_AND_NAN(beta);
}

Vt AlphaBetaFilter::operator()(fpt in)
{
  if(first)
  {
    estimatedPosition = in;
    first = false;
  }
  const fpt activation = in - estimatedPosition;
  OPENANN_CHECK_INF_AND_NAN(activation);

  smoothedPosition = alpha * activation + estimatedPosition;
  OPENANN_CHECK_INF_AND_NAN(smoothedPosition);
  smoothedVelocity += beta / deltaT * activation;
  OPENANN_CHECK_INF_AND_NAN(smoothedVelocity);

  estimatedPosition = smoothedPosition + deltaT * smoothedVelocity;
  OPENANN_CHECK_INF_AND_NAN(estimatedPosition);
  Vt out(2);
  out << estimatedPosition, smoothedVelocity;
  return out;
}
