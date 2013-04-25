#include <OpenANN/Evaluation.h>
#include <OpenANN/Learner.h>
#include <OpenANN/io/DataSet.h>
#include <OpenANN/io/DataSetView.h>
#include <OpenANN/io/Logger.h>
#include <OpenANN/optimization/Optimizer.h>
#include <cmath>
#include <vector>

namespace OpenANN {

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
    ce -= (dataSet.getTarget(n).array() *
        (learner(dataSet.getInstance(n)).array() + 1e-10).log()).sum();
  return ce;
}


double accuracy(Learner& learner, DataSet& dataSet)
{
  // TODO implement
}

Eigen::MatrixXd confusionMatrix(Learner& learner, DataSet& dataSet)
{
  // TODO implement
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

  for(int i = 0; i < dataSet.samples(); ++i) {
    Eigen::VectorXd& output = dataSet.getTarget(i);
    Eigen::VectorXd& input = dataSet.getInstance(i);

    int klass, result;

    if(dataSet.outputs() > 2) {
      klass = output.maxCoeff();
      result = learner(input).maxCoeff();
    } else {
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

  for(int i = 0; i < folds; ++i) {
    // generate training set from splits (remove validation set)
    std::vector<DataSetView> training_splits = splits;
    training_splits.erase(training_splits.begin() + i);

    // generate validation and training set
    DataSetView& validation = splits.at(i);
    DataSetView training(dataSet);
    merge(training, training_splits);
  
    learner.trainingSet(training);
    learner.initialize();

    opt.setOptimizable(learner);
    opt.optimize();

    int training_hits = classificationHits(learner, training);
    int validation_hits = classificationHits(learner, validation);

    OPENANN_INFO
      << "Fold [" << i + 1 << "] "
      << "training result = " 
      << OpenANN::FloatingPointFormatter(100.0 * ((double) training_hits / training.samples()), 2) 
      << "% (" << training_hits << "/" << training.samples() << "), "
      << "test result = " 
      << OpenANN::FloatingPointFormatter(100.0 * ((double) validation_hits / validation.samples()), 2) 
      << "% (" << validation_hits << "/" << validation.samples() << ")  [classification]";
  }
}

}
