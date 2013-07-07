/**
 * @file Random.h
 * Random number generator.
 */

#ifndef OPENANN_UTIL_RANDOM_H_
#define OPENANN_UTIL_RANDOM_H_

#include <OpenANN/util/AssertionMacros.h>
#include <cmath>
#include <cstdlib>
#include <algorithm>

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
   * Set the seed to ensure repeatability.
   *
   * Note that the seed is set globally, i.e. it might also be overwritten by
   * another part of you program.
   *
   * @param seed initial parameter for random number generator
   */
  void seed(unsigned int seed);
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
    return std::sqrt(T(-2) * std::log(generate(T(), T(1)))) *
           std::cos(T(2) * T(M_PI) * generate(T(), T(1)));
  }

  /**
   * Generate a random sequence of indices.
   * @tparam container type of result (must support push_back())
   * @param n number of indices
   * @param result result container, must be empty if initialized = false
   * @param initialized does the container already contain indices?
   */
  template<class C>
  void generateIndices(int n, C& result, bool initialized = false)
  {
    if(!initialized)
    {
      OPENANN_CHECK_EQUALS(result.size(), 0);
      for(int i = 0; i < n; i++)
        result.push_back(i);
    }
    else
    {
      OPENANN_CHECK_EQUALS(result.size(), (size_t) n);
    }
    std::random_shuffle(result.begin(), result.end());
  }
};

} // namespace OpenANN

#endif // OPENANN_UTIL_RANDOM_H_
