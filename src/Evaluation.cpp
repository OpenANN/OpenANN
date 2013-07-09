#include <OpenANN/Evaluation.h>
#include <OpenANN/Learner.h>
#include <OpenANN/ErrorFunctions.h>
#include <OpenANN/io/DataSet.h>
#include <OpenANN/io/DataSetView.h>
#include <OpenANN/io/Logger.h>
#include <OpenANN/optimization/StoppingInterrupt.h>
#include <OpenANN/optimization/Optimizer.h>
#include <cmath>
#include <vector>

namespace OpenANN
{

double sse(Learner& learner, DataSet& dataSet)
{
  const int N = dataSet.samples();
  double sse = 0.0;
  for(int n = 0; n < N; n++)
    sse += (learner(dataSet.getInstance(n)) - dataSet.getTarget(n)).squaredNorm();
  return sse;
}

double mse(Learner& learner, DataSet& dataSet)
{
  return sse(learner, dataSet) / (double) dataSet.samples();
}

double rmse(Learner& learner, DataSet& dataSet)
{
  return std::sqrt(mse(learner, dataSet));
}

double ce(Learner& learner, DataSet& dataSet)
{
  const int N = dataSet.samples();
  double ce = 0.0;
  for(int n = 0; n < N; n++)
    ce += crossEntropy(learner(dataSet.getInstance(n)).transpose(),
                       dataSet.getTarget(n).transpose());
  return ce;
}


double accuracy(Learner& learner, DataSet& dataSet)
{
  const int N = dataSet.samples();
  double accuracy = 0.0;
  for(int n = 0; n < N; n++)
    accuracy += (double)(oneOfCDecoding(learner(dataSet.getInstance(n)))
                         == oneOfCDecoding(dataSet.getTarget(n)));
  return accuracy / (double) N;
}

Eigen::MatrixXi confusionMatrix(Learner& learner, DataSet& dataSet)
{
  const int N = dataSet.samples();
  Eigen::MatrixXi confusionMatrix(dataSet.outputs(), dataSet.outputs());
  confusionMatrix.setZero();
  for(int n = 0; n < N; n++)
    confusionMatrix(oneOfCDecoding(dataSet.getTarget(n)),
                    oneOfCDecoding(learner(dataSet.getInstance(n))))++;
  return confusionMatrix;
}


int oneOfCDecoding(const Eigen::VectorXd& target)
{
  int i;
  target.maxCoeff(&i);
  return i;
}

int classificationHits(Learner& learner, DataSet& dataSet)
{
  int hits = 0;

  for(int i = 0; i < dataSet.samples(); ++i)
  {
    Eigen::VectorXd& output = dataSet.getTarget(i);
    Eigen::VectorXd& input = dataSet.getInstance(i);

    int klass, result;

    if(dataSet.outputs() >= 2)
    {
      output.maxCoeff(&klass);
      learner(input).maxCoeff(&result);
    }
    else
    {
      klass = std::floor(output.x() + 0.5);
      result = std::floor(learner(input).x() + 0.5);
    }

    if(klass == result)
      hits++;
  }

  return hits;
}


void crossValidation(int folds, Learner& learner, DataSet& dataSet, Optimizer& opt)
{
  std::vector<DataSetView> splits;

  split(splits, dataSet, folds);

  OPENANN_INFO << "Run " << folds << "-fold cross-validation";

  for(int i = 0; i < folds; ++i)
  {
    // generate training set from splits (remove validation set)
    std::vector<DataSetView> training_splits = splits;
    training_splits.erase(training_splits.begin() + i);

    // generate validation and training set
    DataSetView& test = splits.at(i);
    DataSetView training(dataSet);
    merge(training, training_splits);

    learner.trainingSet(training);
    learner.initialize();

    opt.setOptimizable(learner);

    OpenANN::StoppingInterrupt interrupt;
    int iteration = 0;

    int training_hits = 0;
    int test_hits = 0;

    while(opt.step() && !interrupt.isSignaled())
    {
      std::stringstream ss;

      training_hits = classificationHits(learner, training);
      test_hits = classificationHits(learner, test);

      ss << "iteration " << ++iteration;
      ss << ", training sse = " << FloatingPointFormatter(sse(learner, training), 4);
      ss << ", training accuracy = " << FloatingPointFormatter((100.0 * training_hits) / training.samples(), 2) << "%";
      ss << ", test accuracy = " << FloatingPointFormatter((100.0 * test_hits) / test.samples(), 2) << "%";

      OPENANN_DEBUG << ss.str();
    }

    OPENANN_INFO
        << "Fold [" << i + 1 << "] "
        << "training result = "
        << OpenANN::FloatingPointFormatter(100.0 * ((double) training_hits / training.samples()), 2)
        << "% (" << training_hits << "/" << training.samples() << "), "
        << "test result = "
        << OpenANN::FloatingPointFormatter(100.0 * ((double) test_hits / test.samples()), 2)
        << "% (" << test_hits << "/" << test.samples() << ")  [classification]";
  }
}

}
