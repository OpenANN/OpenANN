#include <OpenANN>
#include "IDXLoader.h"
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
 * directory and execute the benchmark. The sum of squared errors on training
 * and test set, the correct and wrong predictions on training and test set
 * and the training time will be recorded during the training and saved in the
 * file "mlp-error.log".
 *
 * You can display the accuracy on training set and test set during the
 * training with this Gnuplot script:
 * \code
 * reset
 * set title "MNIST Data Set (MLP 784-200-100-10)"
 * set key bottom
 * set xlabel "Training time / min"
 * set ylabel "Accuracy / %"
 * plot "mlp-error.log" u ($16/60000):($3/600) t "Training Set" w l, \
 *     "mlp-error.log" u ($16/60000):($10/100) t "Test Set" w l
 * \endcode
 */

int main(int argc, char** argv)
{
#ifdef PARALLEL_CORES
  omp_set_num_threads(PARALLEL_CORES);
#endif
  OpenANN::Logger interfaceLogger(OpenANN::Logger::CONSOLE);
  IDXLoader loader(28, 28, 60000, 10000);

  OpenANN::MLP mlp(OpenANN::Logger::APPEND_FILE, OpenANN::Logger::NONE);
  mlp.input(loader.D)
    .fullyConnectedHiddenLayer(200, OpenANN::MLP::TANH)
    .fullyConnectedHiddenLayer(100, OpenANN::MLP::TANH)
    .output(loader.F, OpenANN::MLP::CE, OpenANN::MLP::SM)
    .trainingSet(loader.trainingInput, loader.trainingOutput)
    .testSet(loader.testInput, loader.testOutput)
    .training(OpenANN::MLP::BATCH_SGD);
  OpenANN::StopCriteria stop;
  stop.maximalIterations = 15;
  interfaceLogger << "Created MLP.\n" << "D = " << loader.D << ", F = "
      << loader.F << ", N = " << loader.trainingN << ", L = " << mlp.dimension() << "\n";
  mlp.fit(stop);
  interfaceLogger << "Error = " << mlp.error() << "\n\n";
  interfaceLogger << "Wrote data to mlp-error.log.\n";

  return EXIT_SUCCESS;
}
