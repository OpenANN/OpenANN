#include <OpenANN/OpenANN>
#include <OpenANN/optimization/MBSGD.h>
#include <OpenANN/io/DirectStorageDataSet.h>
#include <OpenANN/Evaluator.h>
#include "CIFARLoader.h"

/**
 * \page CIFAR10Benchmark CIFAR-10
 *
 * The dataset is available at:
 * http://www.cs.toronto.edu/~kriz/cifar.html
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

  std::string directory = ".";
  if(argc > 1)
    directory = std::string(argv[1]);

  CIFARLoader loader(directory);

  OpenANN::Net net;                                             // Nodes per layer:
  net.inputLayer(loader.C, loader.X, loader.Y)                  //   3 x 32 x 32
  .convolutionalLayer(50, 5, 5, OpenANN::RECTIFIER, 0.05)       //  50 x 28 x 28
  .maxPoolingLayer(2, 2)                                        //  50 x 14 x 14
  .convolutionalLayer(30, 3, 3, OpenANN::RECTIFIER, 0.05)       //  30 x 12 x 12
  .maxPoolingLayer(2, 2)                                        //  30 x  6 x  6
  .convolutionalLayer(20, 3, 3, OpenANN::RECTIFIER, 0.05)       //  20 x  4 x  4
  .maxPoolingLayer(2, 2)                                        //  20 x  2 x  2
  .fullyConnectedLayer(100, OpenANN::RECTIFIER, 0.05, true)     // 100
  .fullyConnectedLayer(50, OpenANN::RECTIFIER, 0.05, true)      //  50
  .outputLayer(loader.F, OpenANN::SOFTMAX, 0.05)                //  10
  .trainingSet(loader.trainingInput, loader.trainingOutput);
  OpenANN::MulticlassEvaluator evaluator(1, OpenANN::Logger::FILE);
  OpenANN::DirectStorageDataSet testSet(&loader.testInput, &loader.testOutput,
                                        &evaluator);
  net.validationSet(testSet);
  net.setErrorFunction(OpenANN::CE);
  OPENANN_INFO << "Created MLP.";
  OPENANN_INFO << "D = " << loader.D << ", F = " << loader.F << ", N = "
               << loader.trainingN << ", L = " << net.dimension();

  OpenANN::StoppingCriteria stop;
  stop.maximalIterations = 100;
  OpenANN::MBSGD optimizer(0.01, 0.6, 10, false, 1.0, 0.0, 0.0, 1.0, 0.01, 100.0);
  optimizer.setOptimizable(net);
  optimizer.setStopCriteria(stop);
  optimizer.optimize();

  OPENANN_INFO << "Error = " << net.error();
  OPENANN_INFO << "Wrote data to evaluation-*.log.";

  return EXIT_SUCCESS;
}
