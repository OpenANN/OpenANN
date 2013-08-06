#include <OpenANN/OpenANN>
#include <OpenANN/optimization/MBSGD.h>
#include <OpenANN/Evaluator.h>
#include <OpenANN/io/DataStream.h>
#include <OpenANN/io/DirectStorageDataSet.h>
#include "IDXLoader.h"

/**
 * \page MNISTBenchmark MNIST
 *
 * Here, we use a CNN that is similar to Yann LeCun's LeNet 5 to learn
 * handwritten digit  recognition. Download the MNIST data set from
 * <a href="http://yann.lecun.com/exdb/mnist/" target=_blank>THE MNIST
 * DATABASE of handwritten digits</a>. You need all four files. Create the
 * directory "mnist" in your working directory, move the data set to this
 * directory and execute the benchmark or pass the directory of the MNIST
 * data set as argument to the program. Some information about the
 * classification of the test set will be logged in the file
 * "evaluation-*.log", where '*' is the starting time.
 *
 * To execute the benchmark you can run the Python script:
\code
python benchmark.py [download] [run] [evaluate]
\endcode
 * download will download the dataset, run will start the benchmark and
 * evaluate will plot the result. You can of course modify the script or do
 * each step manually.
 */

int main(int argc, char** argv)
{
  OpenANN::useAllCores();

  std::string directory = "./";
  bool distortions = false;
  if(argc > 1)
    directory = std::string(argv[1]);
  if(argc > 2)
    distortions = true;

  IDXLoader loader(28, 28, 60000, 10000, directory);
  Distorter distorter;

  OpenANN::Net net;
  net.inputLayer(1, loader.padToX, loader.padToY);
  net.setRegularization(0.0, 0.0, 15.0);
  if(distortions)
  {
    // High model complexity
    net.convolutionalLayer(20, 5, 5, OpenANN::RECTIFIER, 0.05)
    .maxPoolingLayer(2, 2)
    .convolutionalLayer(40, 5, 5, OpenANN::RECTIFIER, 0.05)
    .maxPoolingLayer(2, 2)
    .fullyConnectedLayer(150, OpenANN::RECTIFIER, 0.05);
  }
  else
  {
    // Smaller network
    net.dropoutLayer(0.2)
    .convolutionalLayer(30, 5, 5, OpenANN::RECTIFIER, 0.05)
    .dropoutLayer(0.2)
    .maxPoolingLayer(2, 2)
    .convolutionalLayer(30, 5, 5, OpenANN::RECTIFIER, 0.05)
    .dropoutLayer(0.2)
    .maxPoolingLayer(2, 2)
    .fullyConnectedLayer(200, OpenANN::RECTIFIER, 0.05)
    .dropoutLayer(0.4)
    .fullyConnectedLayer(150, OpenANN::RECTIFIER, 0.05)
    .dropoutLayer(0.4);
  }
  net.outputLayer(loader.F, OpenANN::LINEAR, 0.05);
  net.useDropout(true);
  OpenANN::MulticlassEvaluator evaluator(OpenANN::Logger::FILE);
  OpenANN::DirectStorageDataSet testSet(&loader.testInput, &loader.testOutput,
                                        &evaluator);
  net.validationSet(testSet);
  net.setErrorFunction(OpenANN::CE);
  OPENANN_INFO << "Created MLP.";
  OPENANN_INFO << "D = " << loader.D << ", F = " << loader.F
               << ", N = " << loader.trainingN << ", L = " << net.dimension();
  OPENANN_INFO << "Press CTRL+C to stop optimization after the next"
               " iteration is finished.";

  OpenANN::MBSGD optimizer(0.05, 0.0, 1, true, 0.9998, 0.001, 0.0, 1.0, 1.0, 1.0);
  OpenANN::DataStream stream = OpenANN::DataStream(loader.trainingN)
      .setLearner(net).setOptimizer(optimizer);

  Eigen::VectorXd x, t;
  if(distortions)
  {
    // Generate more training data with distortions
    for(int it = 0; it < 200; it++)
    {
      for(int n = 0; n < loader.trainingN; n++)
      {
        x = loader.trainingInput.row(n);
        t = loader.trainingOutput.row(n);
        distorter.applyDistortion(x, loader.padToX, loader.padToY);
        stream.addSample(&x, &t);
      }
    }
  }
  else
  {
    for(int it = 0; it < 100; it++)
    {
      for(int n = 0; n < loader.trainingN; n++)
      {
        x = loader.trainingInput.row(n);
        t = loader.trainingOutput.row(n);
        stream.addSample(&x, &t);
      }
    }
  }

  // Set best parameter vector
  optimizer.result();
  OPENANN_INFO << "Error = " << net.error();
  OPENANN_INFO << "Wrote data to evaluation-*.log.";

  net.save("mnist.net");

  return EXIT_SUCCESS;
}
