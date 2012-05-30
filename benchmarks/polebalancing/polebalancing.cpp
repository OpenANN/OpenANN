#include <OpenANN>
#include <SinglePoleBalancing.h>
#include <DoublePoleBalancing.h>
#include <NeuroEvolutionAgent.h>
#include <Test/Stopwatch.h>
#include <numeric>
#include <vector>
#ifdef PARALLEL_CORES
#include <omp.h>
#endif

/**
 * \page PoleBalancingBenchmark Pole Balancing
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
  fpt mean, stdDev, time;
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
    bool alphaBetaFilter, int parameters, int runs, fpt sigma0)
{
  OpenANN::Environment* env;
  if(doublePole)
    env = new DoublePoleBalancing(fullyObservable);
  else
    env = new SinglePoleBalancing(fullyObservable);

  Results results;
  results.runs = runs;
  std::vector<fpt> episodes;

  OpenANN::Logger progressLogger(Logger::CONSOLE);
  for(int run = 0; run < runs; run++)
  {
    NeuroEvolutionAgent agent(0, false, "linear", true, parameters,
        fullyObservable, alphaBetaFilter);
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
  results.mean /= (fpt) runs;
  results.time /= (fpt) runs;
  results.min = (int) *std::min_element(episodes.begin(), episodes.end());
  results.max = (int) *std::max_element(episodes.begin(), episodes.end());
  std::sort(episodes.begin(), episodes.end());
  results.median = (int) episodes[episodes.size()/2];
  for(int run = 0; run < runs; run++)
  {
    episodes[run] -= results.mean;
    episodes[run] *= episodes[run];
  }
  results.stdDev = std::sqrt(std::accumulate(episodes.begin(), episodes.end(), (fpt) 0) / (fpt) runs);

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
#ifdef PARALLEL_CORES
  omp_set_num_threads(PARALLEL_CORES);
#endif

  OpenANN::Logger configLogger(OpenANN::Logger::CONSOLE);
  int runs = 1000;

  configLogger << "SPB, MDP, uncompressed\n";
  Results results = benchmarkConfiguration(false, true, false, -1, runs, 10.0);
  printResults(results);
  configLogger << "SPB, MDP, compressed (1)\n";
  results = benchmarkConfiguration(false, true, false, 1, runs, 100.0);
  printResults(results);
  configLogger << "DPB, MDP, uncompressed\n";
  results = benchmarkConfiguration(true, true, false, -1, runs, 10.0);
  printResults(results);
  configLogger << "DPB, MDP, compressed (5)\n";
  results = benchmarkConfiguration(true, true, false, 5, runs, 10.0);
  printResults(results);
  configLogger << "SPB, POMDP, uncompressed\n";
  results = benchmarkConfiguration(false, false, true, -1, runs, 10.0);
  printResults(results);
  configLogger << "SPB, POMDP, compressed (3)\n";
  results = benchmarkConfiguration(false, false, true, 3, runs, 10.0);
  printResults(results);
  configLogger << "DPB, POMDP, uncompressed\n";
  results = benchmarkConfiguration(true, false, true, -1, runs, 10.0);
  printResults(results);
  configLogger << "DPB, POMDP, compressed (5)\n";
  results = benchmarkConfiguration(true, false, true, 5, runs, 10.0);
  printResults(results);
}
