#include "RandomTestCase.h"
#include <Random.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

void RandomTestCase::run()
{
  RUN(RandomTestCase, generateInt);
  RUN(RandomTestCase, generateIndex);
  RUN(RandomTestCase, generate);
  RUN(RandomTestCase, sampleNormalDistribution);
}

void RandomTestCase::generateInt()
{
  const int N = 1000;
  std::vector<int> random(N);
  OpenANN::RandomNumberGenerator rng;
  for(int i = 0; i < N; i++)
    random[i] = rng.generateInt(-10, 21);
  int sum = std::accumulate(random.begin(), random.end(), 0);
  ASSERT_WITHIN((double) sum / (double) N, -1.0, 1.0);
  int max = *std::max_element(random.begin(), random.end());
  ASSERT(max <= 10);
  int min = *std::min_element(random.begin(), random.end());
  ASSERT(min >= -10);
}

void RandomTestCase::generateIndex()
{
  const int N = 1000;
  std::vector<size_t> random(N);
  OpenANN::RandomNumberGenerator rng;
  for(int i = 0; i < N; i++)
    random[i] = rng.generateIndex(10);
  size_t sum = std::accumulate(random.begin(), random.end(), 0);
  ASSERT_WITHIN((double) sum / (double) N, 3.5, 5.5);
  size_t max = *std::max_element(random.begin(), random.end());
  ASSERT(max <= 10);
  size_t min = *std::min_element(random.begin(), random.end());
  ASSERT(min >= 0);
}

void RandomTestCase::generate()
{
  const int N = 1000;
  std::vector<double> random(N);
  OpenANN::RandomNumberGenerator rng;
  for(int i = 0; i < N; i++)
    random[i] = rng.generate(-10.0, 20.0);
  double sum = std::accumulate(random.begin(), random.end(), 0);
  ASSERT_WITHIN(sum / (double) N, -1.0, 1.0);
  double max = *std::max_element(random.begin(), random.end());
  ASSERT(max <= 10.0);
  double min = *std::min_element(random.begin(), random.end());
  ASSERT(min >= -10.0);
}

void RandomTestCase::sampleNormalDistribution()
{
  const int N = 1000;
  std::vector<double> random(N);
  OpenANN::RandomNumberGenerator rng;
  for(int i = 0; i < N; i++)
    random[i] = rng.sampleNormalDistribution<double>();
  double sum = std::accumulate(random.begin(), random.end(), 0);
  double mean = sum / (double) N;
  ASSERT_WITHIN(mean, -0.1, 0.1);
  double variance = 0.0;
  for(int i = 0; i < N; i++)
    variance += std::pow(random[i] - mean, 2.0);
  variance /= (double) N;
  ASSERT_WITHIN(variance, 0.8, 1.2);
}
