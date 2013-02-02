#include <DeepNetwork.h>
#include "IDXLoader.h"
#include <io/DirectStorageDataSet.h>
#ifdef PARALLEL_CORES
#include <omp.h>
#endif

/**
 * \page MNISTBenchmark MNIST
 *
 * Here, we use an MLP with the architecture 784-200-100-10 to learn
 * handwritten digit recognition. Download the MNIST data set from
 * <a href="http://yann.lecun.com/exdb/mnist/" target=_blank>THE MNIST
 * DATABASE of handwritten digits</a>. You need all four files. Create the
 * directory "mnist" in your working directory, move the data set to this
 * directory and execute the benchmark or pass the directory of the MNIST
 * data set as argument to the program. The sum of squared errors on training
 * and test set, the correct and wrong predictions on training and test set
 * and the training time will be recorded during the training and saved in the
 * file "dataset.log".
 *
 * You can display the accuracy on training set and test set during the
 * training with this Gnuplot script:
 * \code
 * reset
 * set title "MNIST Data Set (MLP 784-200-100-10)"
 * set key bottom
 * set xlabel "Training time / min"
 * set ylabel "Accuracy / %"
 * plot "dataset.log" u ($16/60000):($3/600) t "Training Set" w l, \
 *     "dataset.log" u ($16/60000):($10/100) t "Test Set" w l
 * \endcode
 */

int main(int argc, char** argv)
{
#ifdef PARALLEL_CORES
  omp_set_num_threads(PARALLEL_CORES);
#endif
  OpenANN::Logger interfaceLogger(OpenANN::Logger::CONSOLE);

  std::string directory = "mnist/";
  if(argc > 1)
    directory = std::string(argv[1]);

  IDXLoader loader(32, 32, 60000, 10000, directory);

  OpenANN::DeepNetwork net(OpenANN::DeepNetwork::CE);
  net.inputLayer(1, loader.padToX, loader.padToY)
     .convolutionalLayer(6, 5, 5, OpenANN::TANH)
     .maxPoolingLayer(2, 2)
     .convolutionalLayer(16, 5, 5, OpenANN::TANH)
     .maxPoolingLayer(2, 2)
     .fullyConnectedLayer(120, OpenANN::TANH)
     .fullyConnectedLayer(84, OpenANN::TANH)
     .outputLayer(loader.F, OpenANN::LINEAR)
     .trainingSet(loader.trainingInput, loader.trainingOutput);
  OpenANN::DirectStorageDataSet testSet(loader.testInput, loader.testOutput,
                                        OpenANN::DirectStorageDataSet::MULTICLASS,
                                        OpenANN::Logger::APPEND_FILE);
  net.testSet(testSet);
  OpenANN::StopCriteria stop;
  stop.maximalIterations = 15;
  interfaceLogger << "Created MLP.\n" << "D = " << loader.D << ", F = "
      << loader.F << ", N = " << loader.trainingN << ", L = " << net.dimension() << "\n";
  net.train(OpenANN::DeepNetwork::BATCH_SGD, stop);
  interfaceLogger << "Error = " << net.error() << "\n\n";
  interfaceLogger << "Wrote data to mlp-error.log.\n";

  return EXIT_SUCCESS;
}
