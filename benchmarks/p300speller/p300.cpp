#include "BCIDataSet.h"
#include <OpenANN>
#include <CompressionMatrixFactory.h>
#include <io/Logger.h>
#include <Test/Stopwatch.h>
#ifdef PARALLEL_CORES
#include <omp.h>
#endif

/**
 * \page P300SpellerBenchmark P300 Speller
 *
 * This program demonstrates how neural networks can be used to classify
 * electroencephalography (EEG) data. In this example we train a single layer
 * perceptron (SLP) to recognize P300 potentials. This is needed in order to
 * spell characters with brain-computer interfaces (BCI).
 *
 * The benchmarks can be accelerated with CUDA. We need at least 6 GByte RAM.
 *
 * Here we use the data set II from the BCI competition III. You can download
 * the data set from http://www.bbci.de/competition/iii. Note that you have to
 * register first. You need the ASCII format. The downloaded files will be
 *
 * - Subject_A_Train_Flashing.txt
 * - Subject_A_Train_Signal.txt
 * - Subject_A_Train_StimulusCode.txt
 * - Subject_A_Train_StimulusType.txt
 * - Subject_A_Train_TargetChar.txt
 * - Subject_A_Test_Flashing.txt
 * - Subject_A_Test_Signal.txt
 * - Subject_A_Test_StimulusCode.txt
 * - Subject_B_Train_Flashing.txt
 * - Subject_B_Train_Signal.txt
 * - Subject_B_Train_StimulusCode.txt
 * - Subject_B_Train_StimulusType.txt
 * - Subject_B_Train_TargetChar.txt
 * - Subject_B_Test_Flashing.txt
 * - Subject_B_Test_Signal.txt
 * - Subject_B_Test_StimulusCode.txt
 *
 * In order to test the performance on the test set, we have to download the
 * target characters of the test set separately and we must generate the
 * expected targets for the classifier. You find the target characters at
 *
 * - http://www.bbci.de/competition/iii/results/albany/true_labels_a.txt
 * - http://www.bbci.de/competition/iii/results/albany/true_labels_b.txt
 * 
 * You can generate the files
 *
 * - Subject_A_Test_StimulusCode.txt
 * - Subject_A_Test_TargetChar.txt
 * - Subject_B_Test_StimulusCode.txt
 * - Subject_B_Test_TargetChar.txt
 *
 * with the ruby script "targets":
 *
 * \code
 * ruby targets /path/to/dataset-directory/
 * \endcode
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
  OpenANN::MLP mlp(OpenANN::Logger::NONE, OpenANN::Logger::NONE);
  mlp.input(csDimension > 0 ? csDimension : trainingSet.inputs())
    .output(trainingSet.outputs())
    .training(OpenANN::MLP::BATCH_LMA)
    .trainingSet(trainingSet)
    .testSet(testSet);

  OpenANN::Logger progressLogger(OpenANN::Logger::CONSOLE);
  for(int run = 0; run < runs; run++)
  {
    trainingSet.reset();
    testSet.reset();
    if(filter)
    {
      trainingSet.decimate(subsamplingFactor);
      testSet.decimate(subsamplingFactor);
    }
    if(csDimension > 0)
    {
      Mt compressionMatrix;
      OpenANN::CompressionMatrixFactory cmf(trainingSet.inputs(), csDimension,
          OpenANN::CompressionMatrixFactory::SPARSE_RANDOM);
      cmf.createCompressionMatrix(compressionMatrix);
      trainingSet.compress(compressionMatrix);
      testSet.compress(compressionMatrix);
    }

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
