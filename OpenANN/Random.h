#pragma once

#include <AssertionMacros.h>
#include <cmath>
#include <ctime>
#include <cstdlib>

class RandomNumberGenerator
{
public:
  RandomNumberGenerator()
  {
    static bool seedInitialized = false;
    if(!seedInitialized)
    {
      srand(time(0));
      seedInitialized = true;
    }
  }

  int generateInt(int min, int range) const
  {
    OPENANN_CHECK(range >= 0);
    if(range == 0)
      return min;
    else
      return rand() % range + min;
  }

  size_t generateIndex(size_t size) const
  {
    return size_t(generateInt(0, size));
  }

  template<class T>
  T generate(T min, T range) const
  {
    OPENANN_CHECK(range >= T());
    if(range == T(0))
      return min;
    else
      return (T(rand()) / T(RAND_MAX)) * range + min;
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
