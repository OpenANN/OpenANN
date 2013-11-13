#ifndef OPENANN_IO_DATA_SET_H_
#define OPENANN_IO_DATA_SET_H_

#include <Eigen/Core>

namespace OpenANN
{

class Learner;

/**
 * @class DataSet
 *
 * Data set interface.
 *
 * A data set can be either a training, validation or test set. Supervised
 * data sets provide instances and targets. Unsupervised data sets might
 * provide only instances.
 *
 * See \ref CreateDataSet.
 */
class DataSet
{
public:
  virtual ~DataSet() {}
  /**
   * Number of instances. Assumes that the data set has a fixed size, at least
   * for one iteration of the optimization algorithm.
   * @return number of examples
   */
  virtual int samples() = 0;
  /**
   * Input dimensions of instances.
   * @return number of inputs
   */
  virtual int inputs() = 0;
  /**
   * Output dimensions of instances.
   * @return number of outputs
   */
  virtual int outputs() = 0;
  /**
   * Get the input of the ith instance.
   * @param n number of instance
   * @return input
   */
  virtual Eigen::VectorXd& getInstance(int n) = 0;
  /**
   * Get the output of the ith instance.
   * @param n number of instance
   * @return output
   */
  virtual Eigen::VectorXd& getTarget(int n) = 0;
  /**
   * This function is called after an iteration of the optimization algorithm.
   * It could log results, modify or extend the data set or whatever.
   * @param learner learned model
   */
  virtual void finishIteration(Learner& learner) = 0;
};

} // namespace OpenANN

#endif // OPENANN_IO_DATA_SET_H_
