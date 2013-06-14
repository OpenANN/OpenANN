#include <OpenANN/OpenANN>
#include <OpenANN/optimization/MBSGD.h>
#include <OpenANN/io/DirectStorageDataSet.h>
#include <OpenANN/Evaluator.h>
#include "CIFARLoader.h"
#ifdef PARALLEL_CORES
#include <omp.h>
#endif

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
#ifdef PARALLEL_CORES
  omp_set_num_threads(PARALLEL_CORES);
#endif
  bool bigNet = false;
  std::string directory = ".";
  if(argc > 1)
    directory = std::string(argv[1]);
  if(argc > 2)
    bigNet = true;

  CIFARLoader loader(directory);

  OpenANN::Net net;                                                  // Nodes per layer:
  net.inputLayer(loader.C, loader.X, loader.Y);                      //   3 x 32 x 32
  if(bigNet)
  {
    net.convolutionalLayer(200, 5, 5, OpenANN::RECTIFIER, 0.05)      // 200 x 28 x 28
    .maxPoolingLayer(2, 2)                                           // 200 x 14 x 14
    .convolutionalLayer(150, 3, 3, OpenANN::RECTIFIER, 0.05)         // 150 x 12 x 12
    .maxPoolingLayer(2, 2)                                           // 150 x  6 x  6
    .convolutionalLayer(100, 3, 3, OpenANN::RECTIFIER, 0.05)         // 100 x  4 x  4
    .maxPoolingLayer(2, 2)                                           // 100 x  2 x  2
    .fullyConnectedLayer(300, OpenANN::RECTIFIER, 0.05, true)        // 300
    .fullyConnectedLayer(100, OpenANN::RECTIFIER, 0.05, true);       // 100
  }
  else
  {
    net.convolutionalLayer(50, 5, 5, OpenANN::RECTIFIER, 0.05)       //  50 x 28 x 28
    .maxPoolingLayer(2, 2)                                           //  50 x 14 x 14
    .convolutionalLayer(30, 3, 3, OpenANN::RECTIFIER, 0.05)          //  30 x 12 x 12
    .maxPoolingLayer(2, 2)                                           //  30 x  6 x  6
    .convolutionalLayer(20, 3, 3, OpenANN::RECTIFIER, 0.05)          //  20 x  4 x  4
    .maxPoolingLayer(2, 2)                                           //  20 x  2 x  2
    .fullyConnectedLayer(100, OpenANN::RECTIFIER, 0.05, true)        // 100
    .fullyConnectedLayer(50, OpenANN::RECTIFIER, 0.05, true);        //  50
  }
  net.outputLayer(loader.F, OpenANN::LINEAR, 0.05)                   //  10
  .trainingSet(loader.trainingInput, loader.trainingOutput);
  OpenANN::MulticlassEvaluator evaluator(OpenANN::Logger::FILE);
  OpenANN::DirectStorageDataSet testSet(&loader.testInput, &loader.testOutput,
                                        &evaluator);
  net.validationSet(testSet);
  net.setErrorFunction(OpenANN::CE);
  OPENANN_INFO << "Created MLP.";
  OPENANN_INFO << "D = " << loader.D << ", F = " << loader.F << ", N = "
               << loader.trainingN << ", L = " << net.dimension();

  OpenANN::StoppingCriteria stop;
  stop.maximalIterations = 100;
  OpenANN::MBSGD optimizer(0.01, 0.6, 10, 1.0, 0.0, 0.0, 1.0, 0.01, 100.0);
  optimizer.setOptimizable(net);
  optimizer.setStopCriteria(stop);
  optimizer.optimize();

  OPENANN_INFO << "Error = " << net.error();
  OPENANN_INFO << "Wrote data to evaluation-*.log.";

  return EXIT_SUCCESS;
}
