#ifndef OPENANN_RANDOM_H
#define OPENANN_RANDOM_H

#include <AssertionMacros.h>
#include <cmath>
#include <cstdlib>

namespace OpenANN
{

class RandomNumberGenerator
{
public:
  RandomNumberGenerator();
  int generateInt(int min, int range) const;
  size_t generateIndex(size_t size) const;

  template<class T>
  T generate(T min, T range) const
  {
    OPENANN_CHECK(range >= T());
    if(range == T())
      return min;
    else
      return (T) rand() / (T) RAND_MAX * range + min;
  }

  /**
   * Box–Muller transform.
   * Draws a sample from a normal distribution with zero mean and variance 1.
   * @see http://en.wikipedia.org/wiki/Box-Muller_transform
   */
  template<class T>
  T sampleNormalDistribution() const
  {
    return sqrt(T(-2) * log(generate(T(), T(1)))) * cos(T(2) * T(M_PI) * generate(T(), T(1)));
  }
};

}

#endif // OPENANN_RANDOM_H
