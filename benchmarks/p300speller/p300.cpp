#include "BCIDataSet.h"
#include <OpenANN>
#include <CompressionMatrixFactory.h>
#include <io/Logger.h>
#include <Test/Stopwatch.h>
#ifdef PARALLEL_CORES
#include <omp.h>
#endif

/**
 * \page P300Speller P300 Speller
 *
 * This program demonstrates how neural networks can be used to classify
 * electroencephalography (EEG) data. In this example we train a single layer
 * perceptron (SLP) to recognize P300 potentials. This is needed in order to
 * spell characters with brain-computer interfaces (BCI).
 *
 * You can download the data set from http://www.bbci.de/competition/iii. Note
 * that you have to register first. You need the ASCII format.
 */

struct Result
{
  fpt duration, iterations, correct5, correct15;

  Result() { reset(); }

  void reset()
  {
    duration = 0;
    iterations = 0;
    correct5 = 0;
    correct15 = 0;
  }
};

void runTest(Result& result, BCIDataSet& trainingSet, BCIDataSet& testSet,
    int runs, OpenANN::StopCriteria stop, int csDimension, bool filter,
    int subsamplingFactor = 1)
{
  trainingSet.reset();
  testSet.reset();
  if(filter)
  {
    trainingSet.decimate(subsamplingFactor);
    testSet.decimate(subsamplingFactor);
  }

  OpenANN::MLP mlp(OpenANN::Logger::NONE, OpenANN::Logger::NONE);
  mlp.input(csDimension > 0 ? csDimension : trainingSet.inputs())
    .output(trainingSet.outputs())
    .training(OpenANN::MLP::BATCH_LMA)
    .trainingSet(trainingSet)
    .testSet(testSet);

  OpenANN::Logger progressLogger(OpenANN::Logger::CONSOLE);
  for(int run = 0; run < runs; run++)
  {

    Stopwatch sw;
    mlp.fit(stop);
    result.duration += sw.stop(Stopwatch::SECONDS);
    result.iterations += trainingSet.iteration;
    result.correct5 += testSet.evaluate(mlp, 5);
    result.correct15 += testSet.evaluate(mlp, 15);
    progressLogger << ".";
  }
  progressLogger << "\n";
}

void printResult(Result& result, int runs)
{
  typedef OpenANN::FloatingPointFormatter fmt;
  OpenANN::Logger resultLogger(OpenANN::Logger::CONSOLE);
  resultLogger << fmt(result.iterations / (fpt) runs, 2) << "\t"
      << fmt(result.duration / (fpt) runs, 2) << "\t"
      << fmt(result.correct5 / (fpt) runs, 2) << "\t\t"
      << fmt(result.correct15 / (fpt) runs, 2) << "\n";
  result.reset();
}

int main(int argc, char** argv)
{
#ifdef PARALLEL_CORES
  omp_set_num_threads(PARALLEL_CORES);
#endif

  OpenANN::Logger interfaceLogger(OpenANN::Logger::CONSOLE);

  std::string directory = ".";
  if(argc > 1)
    directory = std::string(argv[1]);

  BCIDataSet trainingSetA(directory, "A", "training", false);
  BCIDataSet testSetA(directory, "A", "test", false);
  BCIDataSet trainingSetB(directory, "B", "training", false);
  BCIDataSet testSetB(directory, "B", "test", false);

  Stopwatch sw;
  trainingSetA.load();
  testSetA.load();
  interfaceLogger << "Loaded data set A in " << sw.stop(Stopwatch::SECONDS) << " s.\n";
  sw.start();
  trainingSetB.load();
  testSetB.load();
  interfaceLogger << "Loaded data set B in " << sw.stop(Stopwatch::SECONDS) << " s.\n";

  OpenANN::StopCriteria stop;
  stop.maximalIterations = 20;
  stop.minimalValueDifferences = 0.001;

  int runs = 10;
  interfaceLogger << "Iter.\tTime\t5 trials\t15 trials\t(average of " << runs << " runs, 2 data sets)\n";
  Result result;
  interfaceLogger << "decimation, 1344 parameters\n";
  runTest(result, trainingSetA, testSetA, runs, stop, -1, true, 11);
  runTest(result, trainingSetB, testSetB, runs, stop, -1, true, 11);
  printResult(result, 2*runs);
  interfaceLogger << "decimation, compression, 800 parameters\n";
  runTest(result, trainingSetA, testSetA, runs, stop, 800, true, 11);
  runTest(result, trainingSetB, testSetB, runs, stop, 800, true, 11);
  printResult(result, 2*runs);
  interfaceLogger << "lowpass filter, compression, 800 parameters\n";
  runTest(result, trainingSetA, testSetA, runs, stop, 800, true, 1);
  runTest(result, trainingSetB, testSetB, runs, stop, 800, true, 1);
  printResult(result, 2*runs);
  interfaceLogger << "compression, 1200 parameters\n";
  runTest(result, trainingSetA, testSetA, runs, stop, 1200, false);
  runTest(result, trainingSetB, testSetB, runs, stop, 1200, false);
  printResult(result, 2*runs);

  return EXIT_SUCCESS;
}
