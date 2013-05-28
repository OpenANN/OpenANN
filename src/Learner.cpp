#include <OpenANN/Learner.h>
#include <OpenANN/io/DirectStorageDataSet.h>

namespace OpenANN
{

Learner::Learner()
    : trainSet(0), validSet(0), deleteTrainSet(false), deleteValidSet(false),
      N(0)
{
}

Learner::~Learner()
{
  if(deleteTrainSet && trainSet)
    delete trainSet;
  if(deleteValidSet && validSet)
    delete validSet;
}

Learner& Learner::trainingSet(Eigen::MatrixXd& input,
                              Eigen::MatrixXd& output)
{
  if(deleteTrainSet && trainSet)
    delete trainSet;
  trainSet = new DirectStorageDataSet(&input, &output);
  deleteTrainSet = true;
  N = trainSet->samples();
  return *this;
}

Learner& Learner::trainingSet(DataSet& trainingSet)
{
  if(deleteTrainSet && trainSet)
    delete trainSet;
  trainSet = &trainingSet;
  deleteTrainSet = false;
  N = trainSet->samples();
  return *this;
}

Learner& Learner::removeTrainingSet()
{
  if(deleteTrainSet && trainSet)
    delete trainSet;
  deleteTrainSet = false;
  trainSet = 0;
  return *this;
}

Learner& Learner::validationSet(Eigen::MatrixXd& input,
                                Eigen::MatrixXd& output)
{
  if(deleteValidSet && validSet)
    delete validSet;
  validSet = new DirectStorageDataSet(&input, &output);
  deleteValidSet = true;
  return *this;
}

Learner& Learner::validationSet(DataSet& validationSet)
{
  if(deleteValidSet && validSet)
    delete validSet;
  validSet = &validationSet;
  deleteValidSet = false;
  return *this;
}

Learner& Learner::removeValidationSet()
{
  if(deleteValidSet && validSet)
    delete validSet;
  deleteValidSet = false;
  validSet = 0;
  return *this;
}

}
