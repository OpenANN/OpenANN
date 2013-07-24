#include <OpenANN/OpenANN>
#include <SinglePoleBalancing.h>
#include <DoublePoleBalancing.h>
#include <NeuroEvolutionAgent.h>
#include <Test/Stopwatch.h>
#include <numeric>
#include <vector>

/**
 * \page PoleBalancingBenchmark Pole Balancing
 *
 * This benchmark is based on the example \ref PB.
 *
 * We compare the number of episodes that is needed to learn a successful
 * policy. We use a Single Layer Perceptron (SLP) to represent the policy
 * \f$ \pi : S \rightarrow A \f$. In case of a partially observable
 * environment, we estimate the velocities either with \f$ \alpha - \beta \f$
 * filters or by double exponential smoothing. We do 1000 runs per
 * configuration. The output of the program could be
 * \verbatim
SPB, MDP, uncompressed
....................................................................................................
0/1000 failed
episodes:       33.088+-20.8315
range:          [1,142]
median:         28
time:           118.6 ms

SPB, MDP, compressed (1)
....................................................................................................
0/1000 failed
episodes:       2.476+-2.4391
range:          [1,40]
median:         2
time:           104.3 ms

DPB, MDP, uncompressed
....................................................................................................
0/1000 failed
episodes:       261.146+-174.2955
range:          [28,1410]
median:         224
time:           210.9 ms

DPB, MDP, compressed (5)
....................................................................................................
0/1000 failed
episodes:       201.384+-229.8003
range:          [10,1336]
median:         139
time:           159.6 ms

SPB, POMDP (ABF), uncompressed
....................................................................................................
0/1000 failed
episodes:       31.381+-15.4108
range:          [1,102]
median:         30
time:           117.2 ms

SPB, POMDP (ABF), compressed (3)
....................................................................................................
0/1000 failed
episodes:       14.318+-9.5876
range:          [1,57]
median:         12
time:           114.2 ms

DPB, POMDP (ABF), uncompressed
....................................................................................................
0/1000 failed
episodes:       425.499+-220.8568
range:          [3,1714]
median:         388
time:           228.7 ms

DPB, POMDP (ABF), compressed (5)
....................................................................................................
0/1000 failed
episodes:       434.321+-318.3513
range:          [25,1909]
median:         352
time:           195.4 ms

SPB, POMDP (DES), uncompressed
....................................................................................................
0/1000 failed
episodes:       25.485+-15.8919
range:          [1,97]
median:         22
time:           209.3 ms

SPB, POMDP (DES), compressed (3)
....................................................................................................
0/1000 failed
episodes:       12.169+-7.9433
range:          [1,56]
median:         11
time:           149.2 ms

DPB, POMDP (DES), uncompressed
....................................................................................................
0/1000 failed
episodes:       225.166+-196.6204
range:          [27,1532]
median:         173
time:           584.4 ms

DPB, POMDP (DES), compressed (5)
....................................................................................................
0/1000 failed
episodes:       203.143+-241.2319
range:          [7,1331]
median:         133
time:           322.9 ms
\endverbatim
 * Here SPB means Single Pole Balancing, DPB Double Pole Balancing, MDP
 * (Fully Observable) Markov Decision Process, POMDP Partially Observable
 * Markov Decision Process, ABF \f$ \alpha - \beta \f$ Filters, DES Double
 * Exponential Smoothing (with \f$ \alpha = 0.9, \beta = 0.9 \f$). The number
 * of compressed SLPs' parameters are given in brackets.
 */

struct Result
{
  bool success;
  int episodes;
  unsigned long time;
};

struct Results
{
  int runs;
  int failures;
  int median, min, max;
  double mean, stdDev, time;
  Results()
    : runs(0), failures(0), median(0), min(0), max(0), mean(0), stdDev(0), time(0)
  {
  }
};

Result benchmarkSingleRun(OpenANN::Environment& environment, OpenANN::Agent& agent)
{
  Result result;
  int maximalEpisodes = 100000;
  int requiredSteps = 100000;
  agent.abandoneIn(environment);

  result.success = false;
  Stopwatch sw;
  for(int i = 1; i <= maximalEpisodes; i++)
  {
    environment.restart();
    while(!environment.terminalState())
      agent.chooseAction();
    if(environment.stepsInEpisode() >= requiredSteps)
    {
      result.success = true;
      result.episodes = i;
      break;
    }
  }
  result.time = sw.stop(Stopwatch::MILLISECOND);

  return result;
}

Results benchmarkConfiguration(bool doublePole, bool fullyObservable,
                               bool alphaBetaFilter, bool doubleExponentialSmoothing, int parameters,
                               int runs, double sigma0)
{
  OpenANN::Environment* env;
  if(doublePole)
    env = new DoublePoleBalancing(fullyObservable);
  else
    env = new SinglePoleBalancing(fullyObservable);

  Results results;
  results.runs = runs;
  std::vector<double> episodes;

  OpenANN::Logger progressLogger(Logger::CONSOLE);
  for(int run = 0; run < runs; run++)
  {
    NeuroEvolutionAgent agent(0, false, "linear", parameters > 0, parameters,
                              fullyObservable, alphaBetaFilter, doubleExponentialSmoothing);
    agent.setSigma0(sigma0);
    Result result = benchmarkSingleRun(*env, agent);
    if(run % 10 == 0)
      progressLogger << ".";
    if(!result.success)
      results.failures++;
    episodes.push_back(result.episodes);
    results.time += result.time;
    results.mean += result.episodes;
  }
  progressLogger << "\n";
  results.mean /= (double) runs;
  results.time /= (double) runs;
  results.min = (int) * std::min_element(episodes.begin(), episodes.end());
  results.max = (int) * std::max_element(episodes.begin(), episodes.end());
  std::sort(episodes.begin(), episodes.end());
  results.median = (int) episodes[episodes.size() / 2];
  for(int run = 0; run < runs; run++)
  {
    episodes[run] -= results.mean;
    episodes[run] *= episodes[run];
  }
  results.stdDev = std::sqrt(std::accumulate(episodes.begin(), episodes.end(), 0.0) / (double) runs);

  delete env;
  return results;
}

void printResults(const Results& results)
{
  typedef OpenANN::FloatingPointFormatter fmt;
  OpenANN::Logger resultLogger(OpenANN::Logger::CONSOLE);
  resultLogger << results.failures << "/" << results.runs
               << " failed\nepisodes:\t" << fmt(results.mean, 3) << "+-"
               << fmt(results.stdDev, 4) << "\nrange:\t\t[" << results.min << ","
               << results.max << "]\nmedian:\t\t" << results.median << "\ntime:\t\t"
               << results.time << " ms\n\n";
}

int main(int argc, char** argv)
{
  OpenANN::useAllCores();

  OpenANN::Logger configLogger(OpenANN::Logger::CONSOLE);
  int runs = 1000;

  configLogger << "SPB, MDP, uncompressed\n";
  Results results = benchmarkConfiguration(false, true, true, false, -1, runs, 10.0);
  printResults(results);
  configLogger << "SPB, MDP, compressed (1)\n";
  results = benchmarkConfiguration(false, true, false, false, 1, runs, 100.0);
  printResults(results);
  configLogger << "DPB, MDP, uncompressed\n";
  results = benchmarkConfiguration(true, true, false, false, -1, runs, 10.0);
  printResults(results);
  configLogger << "DPB, MDP, compressed (5)\n";
  results = benchmarkConfiguration(true, true, false, false, 5, runs, 10.0);
  printResults(results);
  configLogger << "SPB, POMDP (ABF), uncompressed\n";
  results = benchmarkConfiguration(false, false, true, false, -1, runs, 10.0);
  printResults(results);
  configLogger << "SPB, POMDP (ABF), compressed (3)\n";
  results = benchmarkConfiguration(false, false, true, false, 3, runs, 10.0);
  printResults(results);
  configLogger << "DPB, POMDP (ABF), uncompressed\n";
  results = benchmarkConfiguration(true, false, true, false, -1, runs, 10.0);
  printResults(results);
  configLogger << "DPB, POMDP (ABF), compressed (5)\n";
  results = benchmarkConfiguration(true, false, true, false, 5, runs, 10.0);
  printResults(results);
  configLogger << "SPB, POMDP (DES), uncompressed\n";
  results = benchmarkConfiguration(false, false, false, true, -1, runs, 10.0);
  printResults(results);
  configLogger << "SPB, POMDP (DES), compressed (3)\n";
  results = benchmarkConfiguration(false, false, false, true, 3, runs, 10.0);
  printResults(results);
  configLogger << "DPB, POMDP (DES), uncompressed\n";
  results = benchmarkConfiguration(true, false, false, true, -1, runs, 10.0);
  printResults(results);
  configLogger << "DPB, POMDP (DES), compressed (5)\n";
  results = benchmarkConfiguration(true, false, false, true, 5, runs, 10.0);
  printResults(results);
}
