#include <DeepNetwork.h>
#include "IDXLoader.h"
#include <io/DirectStorageDataSet.h>
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
 * classification of the test set will be logged in the file "dataset*.log".
 *
 * You can display the accuracy on test set during the training with this
 * Gnuplot script:
 * \code
 * reset
 * set title "MNIST Data Set"
 * set key bottom
 * set xlabel "Training time / min"
 * set ylabel "Error / %"
 * plot "dataset*.log" u ($5/1000):($4/100) t "Test Set" w l
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

  IDXLoader loader(28, 28, 60000, 10000, directory);

  OpenANN::DeepNetwork net(OpenANN::DeepNetwork::CE);             // Nodes per layer:
  net.inputLayer(1, loader.padToX, loader.padToY, true)           //   1 x 28 x 28
     .convolutionalLayer(10, 5, 5, OpenANN::RECTIFIER, 0.05)      //  10 x 24 x 24
     .maxPoolingLayer(2, 2)                                       //  10 x 12 x 12
     .convolutionalLayer(16, 5, 5, OpenANN::RECTIFIER, 0.05)      //  16 x 10 x 10
     .maxPoolingLayer(2, 2)                                       //  16 x  5 x  5
     .compressedLayer(240, 100, OpenANN::RECTIFIER, "sparse")     // 120
     .fullyConnectedLayer(84, OpenANN::RECTIFIER)                 //  84
     .outputLayer(loader.F, OpenANN::LINEAR, 0.05)                //  10
     .trainingSet(loader.trainingInput, loader.trainingOutput);
  OpenANN::DirectStorageDataSet testSet(loader.testInput, loader.testOutput,
                                        OpenANN::DirectStorageDataSet::MULTICLASS,
                                        OpenANN::Logger::FILE);
  net.testSet(testSet);
  OpenANN::StopCriteria stop;
  stop.maximalIterations = 100;
  interfaceLogger << "Created MLP.\n" << "D = " << loader.D << ", F = "
      << loader.F << ", N = " << loader.trainingN << ", L = " << net.dimension() << "\n";
  net.train(OpenANN::DeepNetwork::MINIBATCH_SGD, stop, true, true);
  interfaceLogger << "Error = " << net.error() << "\n\n";
  interfaceLogger << "Wrote data to mlp-error.log.\n";

  return EXIT_SUCCESS;
}
