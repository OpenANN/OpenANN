#pragma once

#include <OpenANN/io/DataSet.h>
#include <OpenANN/optimization/Optimizable.h>

namespace OpenANN
{

/**
 * @class Learner
 *
 * Common base class of all learning algorithms.
 *
 * A learner combines a model and a training set so that an Optimizer can
 * minimize the error function on the training set.
 */
class Learner : public Optimizable
{
protected:
  DataSet* trainSet;
  DataSet* validSet;
  bool deleteTrainSet, deleteValidSet;
  int N;
public:
  Learner();
  virtual ~Learner();
  /**
   * Make a prediction.
   * @param x Input vector.
   * @return Prediction.
   */
  virtual Eigen::VectorXd operator()(const Eigen::VectorXd& x) = 0;
  /**
   * Make predictions.
   * @param X Each row represents an input vector.
   * @return Each row represents a prediction.
   */
  virtual Eigen::MatrixXd operator()(const Eigen::MatrixXd& X) = 0;
  /**
   * Set training set.
   * @param input input vectors, each instance should be in a row
   * @param output output vectors, each instance should be in a row
   * @return this for chaining
   */
  virtual Learner& trainingSet(Eigen::MatrixXd& input,
                               Eigen::MatrixXd& output);
  /**
   * Set training set.
   * @param trainingSet training set
   * @return this for chaining
   */
  virtual Learner& trainingSet(DataSet& trainingSet);
  /**
   * Remove the training set from the learner.
   * @return this for chaining
   */
  virtual Learner& removeTrainingSet();
  /**
   * Set validation set.
   * @param input input vectors, each instance should be in a row
   * @param output output vectors, each instance should be in a row
   * @return this for chaining
   */
  virtual Learner& validationSet(Eigen::MatrixXd& input,
                                 Eigen::MatrixXd& output);
  /**
   * Set validation set.
   * @param validationSet validation set
   * @return this for chaining
   */
  virtual Learner& validationSet(DataSet& validationSet);
  /**
   * Remove the validation set from the learner.
   * @return this for chaining
   */
  virtual Learner& removeValidationSet();
};

}