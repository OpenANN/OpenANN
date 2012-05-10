/**
 * @file Random.h
 * Random number generator.
 */

#ifndef OPENANN_RANDOM_H
#define OPENANN_RANDOM_H

#include <AssertionMacros.h>
#include <cmath>
#include <cstdlib>

namespace OpenANN
{

/**
 * @class RandomNumberGenerator
 * A utility class that simplifies the generation of random numbers.
 */
class RandomNumberGenerator
{
public:
  /**
   * Initialize the seed.
   */
  RandomNumberGenerator();
  /**
   * Draw an integer from a uniform distribution.
   * @param min minimal value
   * @param range range of the interval, must be greater than 0
   * @return random number from the interval [min, range)
   */ 
  int generateInt(int min, int range) const;
  /**
   * Draw an index from a uniform distribution.
   * @param size range of the index, must be greater than 0
   * @return random number from the interval [0, size)
   */ 
  size_t generateIndex(size_t size) const;

  /**
   * Draw a number from a uniform distribution.
   * @tparam T number type
   * @param min minimal value
   * @param range range of the interval, must be greater than 0
   * @return random number from the interval [min, range)
   */
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
   * Draw a sample from a normal distribution with zero mean and variance 1.
   * We use the <a href="http://en.wikipedia.org/wiki/Box-Muller_transform"
   * target=_blank>Box–Muller transform</a>. In order to draw a random number
   * from the distribution \f$ \mathcal{N}(\mu, \sigma) \f$, you have to
   * shift and scale the output of this function:
   * \code
RandomNumberGenerator rng;
double mu = ...
double sigma = ...
double rn = mu + sigma*rng.sampleNormalDistribution<double>();
     \endcode
   * @tparam T number type
   * @return standard normal distributed random number
   */
  template<class T>
  T sampleNormalDistribution() const
  {
    return std::sqrt(T(-2) * std::log(generate(T(), T(1)))) * std::cos(T(2) * T(M_PI) * generate(T(), T(1)));
  }
};

}

#endif // OPENANN_RANDOM_H
