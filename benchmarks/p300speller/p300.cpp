#include "BCIDataSet.h"
#include <OpenANN/OpenANN>
#include <OpenANN/CompressionMatrixFactory.h>
#include <OpenANN/io/Logger.h>
#include <Test/Stopwatch.h>

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
 * register first. You need the files in ASCII format. The downloaded files
 * will be
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
 * expected targets for the classifier. This will be done by a script.
 *
 * To execute the benchmark you can run the Python script:
\code
python benchmark.py directory [download] [run]
\endcode
 * download will download the targets for test sets and run will start the
 * benchmark. The directory should point to the location where the datasets
 * are stored.
 *
 * The output could be
 * \verbatim
$ P300Speller /path/to/dataset-directory/
Loaded data set A in 33 s.
Loaded data set B in 33 s.
Iter.   Time    5 trials        15 trials       (average of 10 runs, 2 data sets)
decimation, 1344 parameters
..........
..........
16.60   145.45  64.10           93.50
decimation, compression, 800 parameters
..........
..........
12.35   59.60   63.75           93.90
lowpass filter, compression, 800 parameters
..........
..........
16.10   85.30   64.15           94.05
compression, 1200 parameters
..........
..........
16.10   143.75  55.30           87.80
\endverbatim
 * Here we tested 4 configurations with different preprocessing and
 * compression methods. We performed 10 runs for each configuration and
 * calculated the average number of iterations (Iter.), training time (Time),
 * accuracy with 5 trials and accuracy with 15 trials.
 */

struct Result
{
  double duration, iterations, correct5, correct15;

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
             int runs, OpenANN::StoppingCriteria stop, int csDimension, bool filter,
             int subsamplingFactor = 1)
{
  OpenANN::Net net;
  net.inputLayer(csDimension > 0 ? csDimension : trainingSet.inputs())
  .outputLayer(trainingSet.outputs(), OpenANN::TANH)
  .validationSet(testSet)
  .trainingSet(trainingSet);

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
      Eigen::MatrixXd compressionMatrix;
      OpenANN::CompressionMatrixFactory cmf(trainingSet.inputs(), csDimension,
                                            OpenANN::CompressionMatrixFactory::SPARSE_RANDOM);
      cmf.createCompressionMatrix(compressionMatrix);
      trainingSet.compress(compressionMatrix);
      testSet.compress(compressionMatrix);
    }

    Stopwatch sw;
    train(net, "LMA", OpenANN::MSE, stop);
    result.duration += sw.stop(Stopwatch::SECONDS);
    result.iterations += trainingSet.iteration;
    result.correct5 += testSet.evaluate(net, 5);
    result.correct15 += testSet.evaluate(net, 15);
    progressLogger << ".";
  }
  progressLogger << "\n";
}

void printResult(Result& result, int runs)
{
  typedef OpenANN::FloatingPointFormatter fmt;
  OpenANN::Logger resultLogger(OpenANN::Logger::CONSOLE);
  resultLogger << fmt(result.iterations / (double) runs, 2) << "\t"
               << fmt(result.duration / (double) runs, 2) << "\t"
               << fmt(result.correct5 / (double) runs, 2) << "\t\t"
               << fmt(result.correct15 / (double) runs, 2) << "\n";
  result.reset();
}

int main(int argc, char** argv)
{
  OpenANN::useAllCores();

  OpenANN::Log::getLevel() = OpenANN::Log::INFO;
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

  OpenANN::StoppingCriteria stop;
  stop.maximalIterations = 20;
  stop.minimalValueDifferences = 0.001;

  int runs = 10;
  interfaceLogger << "Iter.\tTime\t5 trials\t15 trials\t(average of " << runs << " runs, 2 data sets)\n";
  Result result;
  interfaceLogger << "decimation, 1344 parameters\n";
  runTest(result, trainingSetA, testSetA, runs, stop, -1, true, 11);
  runTest(result, trainingSetB, testSetB, runs, stop, -1, true, 11);
  printResult(result, 2 * runs);
  interfaceLogger << "decimation, compression, 800 parameters\n";
  runTest(result, trainingSetA, testSetA, runs, stop, 800, true, 11);
  runTest(result, trainingSetB, testSetB, runs, stop, 800, true, 11);
  printResult(result, 2 * runs);
  interfaceLogger << "lowpass filter, compression, 800 parameters\n";
  runTest(result, trainingSetA, testSetA, runs, stop, 800, true, 1);
  runTest(result, trainingSetB, testSetB, runs, stop, 800, true, 1);
  printResult(result, 2 * runs);
  interfaceLogger << "compression, 1200 parameters\n";
  runTest(result, trainingSetA, testSetA, runs, stop, 1200, false);
  runTest(result, trainingSetB, testSetB, runs, stop, 1200, false);
  printResult(result, 2 * runs);

  return EXIT_SUCCESS;
}
