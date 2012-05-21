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
 * \image html eeg-flashing.png
 */

int main(int argc, char** argv)
{
#ifdef PARALLEL_CORES
  omp_set_num_threads(PARALLEL_CORES);
#endif

  Logger errorLogger(Logger::CONSOLE);
  Logger interfaceLogger(Logger::CONSOLE);

  std::string directory = ".";
  if(argc > 1)
    directory = std::string(argv[1]);

  std::string subject = "A";
  int csDimension = -1;
  bool randomCompression = true;
  bool decimate = false;
  int downSamplingFactor = 1;
  bool writeCompressionMatrix = false;
  int trials = 15;

  bool acceptSubject = false;
  bool acceptCSDimension = false;
  bool acceptDownSamplingFactor = false;
  bool acceptTrials = false;

  errorLogger << "Loader parser usage: [-s SUBJECT] [-dec downSamplingFactor] [-wcm] [-csr D | -csd D] [-trials T]\n";
  for(int arg = 1; arg < argc; arg++)
  {
    std::string argument(argv[arg]);
    if(argument == "-s")
      acceptSubject = true;
    else if(acceptSubject)
    {
      subject = argument;
      acceptSubject = false;
    }
    else if(argument == "-csd")
    {
      acceptCSDimension = true;
      randomCompression = false;
    }
    else if(argument == "-csr")
    {
      acceptCSDimension = true;
      randomCompression = true;
    }
    else if(acceptCSDimension)
    {
      csDimension = atoi(argv[arg]);
      acceptCSDimension = false;
    }
    else if(argument == "-dec")
    {
      decimate = true;
      acceptDownSamplingFactor = true;
    }
    else if(acceptDownSamplingFactor)
    {
      downSamplingFactor = atoi(argv[arg]);
      acceptDownSamplingFactor = false;
    }
    else if(argument == "-wcm")
      writeCompressionMatrix = true;
    else if(argument == "-trials")
      acceptTrials = true;
    else if(acceptTrials)
    {
      trials = atoi(argv[arg]);
      acceptTrials = false;
    }
  }

  interfaceLogger << "Subject: " << subject << "\n";
  BCIDataSet trainingSet(directory, subject, "training", false);
  BCIDataSet testSet(directory, subject, "test", false);
  if(decimate)
  {
    trainingSet.decimate(downSamplingFactor);
    testSet.decimate(downSamplingFactor);
  }

  Mt compressionMatrix;
  if(csDimension > 0)
  {
    CompressionMatrixFactory cmf(trainingSet.inputs(), csDimension,
        randomCompression ? CompressionMatrixFactory::GAUSSIAN : CompressionMatrixFactory::DCT);
    cmf.createCompressionMatrix(compressionMatrix);
    trainingSet.compress(compressionMatrix);
    testSet.compress(compressionMatrix);
  }

  if(writeCompressionMatrix && csDimension > 0 && randomCompression)
  {
    Logger logger(Logger::FILE, "compression-matrix");
    logger << compressionMatrix << "\n";
  }

  Stopwatch sw;
  trainingSet.load();
  trainingSet.trials = trials;
  interfaceLogger << "Loaded training set in " << sw.stop(Stopwatch::MILLISECOND) << " ms.\n";
  sw.start();
  testSet.load();
  testSet.trials = trials;
  interfaceLogger << "Loaded test set in " << sw.stop(Stopwatch::MILLISECOND) << " ms.\n";

  StopCriteria stop;
  stop.maximalIterations = 20;
  stop.minimalValueDifferences = 0.001;

  for(int i = 0; i < 1; i++)
  {
    MLP mlp(Logger::FILE, Logger::FILE);
    mlp.input(trainingSet.inputs())
      .output(trainingSet.outputs())
      .training(OpenANN::MLP::BATCH_LMA)
      .trainingSet(trainingSet)
      .testSet(testSet);
    interfaceLogger << "Created MLP.\n" << "D = " << trainingSet.inputs() << ", F = " << trainingSet.outputs() << ", L = " << mlp.dimension() << "\n";
    mlp.fit(stop);
  }

  return EXIT_SUCCESS;
}
