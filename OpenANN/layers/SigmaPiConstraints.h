#pragma once

#include <OpenANN/util/AssertionMacros.h>
#include <OpenANN/layers/SigmaPi.h>

namespace OpenANN {

/**
 * Common constraint for encoding translation invariances into a SigmaPi layer.
 */
struct DistanceConstraint : public SigmaPi::Constraint
{
  DistanceConstraint(size_t width, size_t height)
      : width(width), height(height)
  {}

  virtual double operator() (int p1, int p2) const
  {
    double x1 = p1 % width;
    double y1 = p1 / width;
    double x2 = p2 % width;
    double y2 = p2 / width;

    OPENANN_CHECK(y1 < height);
    OPENANN_CHECK(y2 < height);

    return std::sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
  }

private:
  size_t width;
  size_t height;
};


/**
 * Common constraint for encoding translation and scale invariances into a SigmaPi layer.
 */
struct SlopeConstraint : public SigmaPi::Constraint
{
  SlopeConstraint(size_t width, size_t height)
      : width(width), height(height)
  {}

  virtual double operator() (int p1, int p2) const
  {
    double x1 = p1 % width;
    double y1 = p1 / width;
    double x2 = p2 % width;
    double y2 = p2 / width;

    OPENANN_CHECK(y1 < height);
    OPENANN_CHECK(y2 < height);

    return (x1 == x2) ? (M_PI / 2) : std::atan((y2 - y1) / (x2 - x1));
  }

private:
  size_t width;
  size_t height;
};



}
