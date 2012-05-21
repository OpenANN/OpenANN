#include "BCIDataSet.h"
#include <MLP.h>
#include <CompressionMatrixFactory.h>
#include <io/Logger.h>
#include <Test/Stopwatch.h>
#include <vector>
#include <OpenANN>
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
  fpt duration, iterations, correctTraining, correctTest;

  Result()
  {
    reset();
  }

  void reset()
  {
    duration = 0;
    iterations = 0;
    correctTraining = 0;
    correctTest = 0;
  }
};

void runTest(Result& result, BCIDataSet& trainingSet, BCIDataSet& testSet,
    int runs, StopCriteria stop, int trials, int csDimension, bool filter,
    int subsamplingFactor = -1)
{
  OpenANN::Logger setupLogger(OpenANN::Logger::CONSOLE);

  if(filter)
  {
    trainingSet.decimate(subsamplingFactor);
    testSet.decimate(subsamplingFactor);
  }
  if(csDimension > 0)
  {
    Mt compressionMatrix;
    CompressionMatrixFactory cmf(trainingSet.inputs(), csDimension,
        CompressionMatrixFactory::SPARSE_RANDOM);
    cmf.createCompressionMatrix(compressionMatrix);
    trainingSet.compress(compressionMatrix);
    testSet.compress(compressionMatrix);
  }
  trainingSet.trials = trials;
  testSet.trials = trials;

  setupLogger << (filter ? "lowpass filter" : "no filter") << ", ";
  if(subsamplingFactor > 0)
    setupLogger << "subsampling factor " << subsamplingFactor << ", ";
  if(csDimension > 0)
    setupLogger << "random compression, ";
  setupLogger << trainingSet.inputs() << " parameters, ";
  setupLogger << trials << " trials\n";

  MLP mlp(Logger::NONE, Logger::NONE);
  mlp.input(trainingSet.inputs())
    .output(trainingSet.outputs())
    .training(OpenANN::MLP::BATCH_LMA)
    .trainingSet(trainingSet)
    .testSet(testSet);

  for(int run = 0; run < runs; run++)
  {
    Stopwatch sw;
    mlp.fit(stop);
    result.duration += sw.stop(Stopwatch::SECONDS);
    result.iterations += trainingSet.iteration;
    result.correctTraining += trainingSet.correct;
    result.correctTest += testSet.correct;
  }
}

void printResult(Result& result, int runs)
{
  typedef OpenANN::FloatingPointFormatter fmt;
  OpenANN::Logger resultLogger(OpenANN::Logger::CONSOLE);
  resultLogger << fmt(result.iterations / (fpt) runs, 2) << "\t"
      << fmt(result.duration / (fpt) runs, 2) << "\t"
      << fmt(result.correctTraining / (fpt) runs, 2) << "\t"
      << fmt(result.correctTest / (fpt) runs, 2) << "\n";
  result.reset();
}

int main(int argc, char** argv)
{
#ifdef PARALLEL_CORES
  omp_set_num_threads(PARALLEL_CORES);
#endif

  Logger interfaceLogger(Logger::CONSOLE);

  std::string directory = ".";
  if(argc > 1)
    directory = std::string(argv[1]);

  BCIDataSet trainingSetA(directory, "A", "training", false);
  BCIDataSet testSetA(directory, "A", "test", false);
  BCIDataSet trainingSetB(directory, "B", "training", false);
  BCIDataSet testSetB(directory, "B", "test", false);

  Stopwatch sw;
  trainingSetA.load();
  interfaceLogger << "Loaded training A set in " << sw.stop(Stopwatch::SECONDS) << " s.\n";
  sw.start();
  testSetA.load();
  interfaceLogger << "Loaded test set A in " << sw.stop(Stopwatch::SECONDS) << " s.\n";
  sw.start();
  trainingSetB.load();
  interfaceLogger << "Loaded training set B in " << sw.stop(Stopwatch::SECONDS) << " s.\n";
  sw.start();
  testSetB.load();
  interfaceLogger << "Loaded test set B in " << sw.stop(Stopwatch::SECONDS) << " s.\n";

  StopCriteria stop;
  stop.maximalIterations = 20;
  stop.minimalValueDifferences = 0.001;

  int runs = 10;
  interfaceLogger << "Iter.\tTime\tTrain.\tTest\n";
  Result result;
  runTest(result, trainingSetA, testSetA, runs, stop, 15, 800, true, 11);
  runTest(result, trainingSetB, testSetB, runs, stop, 15, 800, true, 11);
  printResult(result, 2*runs);
  runTest(result, trainingSetA, testSetA, runs, stop, 5, 800, true, 11);
  runTest(result, trainingSetB, testSetB, runs, stop, 5, 800, true, 11);
  printResult(result, 2*runs);
  runTest(result, trainingSetA, testSetA, runs, stop, 15, 800, true, 1);
  runTest(result, trainingSetB, testSetB, runs, stop, 15, 800, true, 1);
  printResult(result, 2*runs);
  runTest(result, trainingSetA, testSetA, runs, stop, 5, 800, true, 1);
  runTest(result, trainingSetB, testSetB, runs, stop, 5, 800, true, 1);
  printResult(result, 2*runs);
  runTest(result, trainingSetA, testSetA, runs, stop, 15, -1, true, 11);
  runTest(result, trainingSetB, testSetB, runs, stop, 15, -1, true, 11);
  printResult(result, 2*runs);
  runTest(result, trainingSetA, testSetA, runs, stop, 5, -1, true, 11);
  runTest(result, trainingSetB, testSetB, runs, stop, 5, -1, true, 11);
  printResult(result, 2*runs);
  runTest(result, trainingSetA, testSetA, runs, stop, 15, 1200, false);
  runTest(result, trainingSetB, testSetB, runs, stop, 15, 1200, false);
  printResult(result, 2*runs);
  runTest(result, trainingSetA, testSetA, runs, stop, 5, 1200, false);
  runTest(result, trainingSetB, testSetB, runs, stop, 5, 1200, false);
  printResult(result, 2*runs);

  return EXIT_SUCCESS;
}
