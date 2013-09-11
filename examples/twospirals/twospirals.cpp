#include "TwoSpiralsVisualization.h"
#include <QApplication>
#include <OpenANN/OpenANN>
#include "CreateTwoSpiralsDataSet.h"
#ifdef PARALLEL_CORES
#include <omp.h>
#endif

/**
 * \page TwoSpirals Two Spirals
 *
 * \section DataSet Data Set
 *
 * This is a classification benchmark which is particularly difficult for
 * gradient descent optimization.
 *
 * The two spirals data set was developed by Lang and Witbrock [1]. The data
 * set we use here is taken from the
 * <a href="http://www.cs.cmu.edu/afs/cs/project/ai-repository/ai/areas/neural/bench/0.html" target=_blank>
 * CMU Neural Networks Benchmarks</a>. It consists of 193 training instances
 * and 193 test instances located on a 2D surface. They belong to one of two
 * spirals:
 *
 * \image html TwoSpirals-DataSet.png
 *
 * \section NetworkArchitecture Network Architecture
 *
 * It is possible to solve this problem with one or two hidden layers. But
 * architectures with two hidden layers need less connections and can learn
 * faster. The architecture we chose here is 2-20-20-1 with bias. You can solve
 * this problem with different compression setups. the smallest number of
 * parameters that reliably solves this problem is 3-6-1 with orthogonal cosine
 * functions. All activation functions are tangens hyperbolicus (MLP::TANH). The
 * best optimization algorithm is Levenberq-Marquardt (MLP::BATCH_LMA).
 *
 * \section UI User Interface
 *
 * You can use the following keys to control the program:
 *  - Q: Toggle display of training set
 *  - W: Toggle display of test set
 *  - E: Toggle display of predicted classes for the whole surface
 *  - R: Toggle display of smooth prediction transitions (black+white or grey)
 *  - A: Start visible training
 *  - Escape: Quit application.
 *
 * A trained model could make these predictions:
 *
 * \image html TwoSpirals-Model.png
 *
 * \section References
 *
 * [1] Kevin J. Lang and Michael J. Witbrock: Learning to Tell Two Spirals
 *    Apart. In: Proceedings of the 1988 Connectionist Models Summer School,
 *    1988. ISBN: 0-55860-015-9
 *
 * \section Code
 *
 * Note that you should not use this as an example for good GUI code because it
 * mixes a lot of logic and visualization.
 *
 * \include "twospirals/twospirals.cpp"
 * \include "twospirals/TwoSpiralsVisualization.h"
 * \include "twospirals/TwoSpiralsVisualization.cpp"
 */

int main(int argc, char** argv)
{
  useAllCores(); // Enable all cores (must be specified during build)
  OpenANNLibraryInfo::print();
  QApplication app(argc, argv);

  Eigen::MatrixXd Xtr, Ytr, Xte, Yte;
  createTwoSpiralsDataSet(2, 1.0, Xtr, Ytr, Xte, Yte);
  TwoSpiralsVisualization visual(Xtr, Ytr, Xte, Yte);
  visual.show();
  visual.resize(500, 500);

  return app.exec();
}
