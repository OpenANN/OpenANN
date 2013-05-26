#include <OpenANN/OpenANN>
#include <OpenANN/optimization/MBSGD.h>
#include "IDXLoader.h"
#include <OpenANN/io/DirectStorageDataSet.h>
#ifdef PARALLEL_CORES
#include <omp.h>
#endif

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
 * classification of the test set will be logged in the file "dataset-*.log",
 * where '*' is the starting time.
 *
 * To execute the benchmark you can run the Python script:
\code
python benchmark.py [download] [run] [evaluate]
\endcode
 * download will download the dataset, run will start the benchmark and
 * evaluate will plot the result. You can of course modify the script or do
 * the each step manually.
 */

int main(int argc, char** argv)
{
#ifdef PARALLEL_CORES
  omp_set_num_threads(PARALLEL_CORES);
#endif
  std::string directory = "./";
  if(argc > 1)
    directory = std::string(argv[1]);

  IDXLoader loader(28, 28, 60000, 10000, directory);

  OpenANN::Net net;                                               // Nodes per layer:
  net.inputLayer(1, loader.padToX, loader.padToY)                 //   1 x 28 x 28
     .convolutionalLayer(10, 5, 5, OpenANN::RECTIFIER, 0.05)      //  20 x 24 x 24
     .maxPoolingLayer(2, 2)                                       //  20 x 12 x 12
     .convolutionalLayer(16, 5, 5, OpenANN::RECTIFIER, 0.05)      //  20 x  8 x  8
     .maxPoolingLayer(2, 2)                                       //  20 x  4 x  4
     .fullyConnectedLayer(120, OpenANN::RECTIFIER, 0.05)          // 150
     .fullyConnectedLayer(84, OpenANN::RECTIFIER, 0.05)          // 100
     .outputLayer(loader.F, OpenANN::LINEAR, 0.05)                //  10
     .trainingSet(loader.trainingInput, loader.trainingOutput);
  OpenANN::DirectStorageDataSet testSet(&loader.testInput, &loader.testOutput,
                                        OpenANN::DirectStorageDataSet::MULTICLASS,
                                        OpenANN::Logger::FILE);
  net.validationSet(testSet);
  net.setErrorFunction(OpenANN::CE);
  OPENANN_INFO << "Created MLP.";
  OPENANN_INFO << "D = " << loader.D << ", F = " << loader.F
               << ", N = " << loader.trainingN << ", L = " << net.dimension();
  OPENANN_INFO << "Press CTRL+C to stop optimization after the next"
      " iteration is finished.";

  OpenANN::StoppingCriteria stop;
  stop.maximalIterations = 100;
  OpenANN::MBSGD optimizer(0.01, 0.6, 16, 0.0, 1.0, 0.0, 0.0, 1.0, 0.01, 100.0);
  optimizer.setOptimizable(net);
  optimizer.setStopCriteria(stop);
  optimizer.optimize();

  OPENANN_INFO << "Error = " << net.error();
  OPENANN_INFO << "Wrote data to dataset-*.log.";

  OpenANN::Logger resultLogger(OpenANN::Logger::APPEND_FILE, "weights");
  resultLogger << optimizer.result();

  return EXIT_SUCCESS;
}
