#ifndef OPENANN_IO_DATA_STREAM_H_
#define OPENANN_IO_DATA_STREAM_H_

#include <Eigen/Dense>

namespace OpenANN
{

class DirectStorageDataSet;
class Optimizer;
class Learner;

/**
 * @class DataStream
 *
 * Streams training data for online training.
 *
 * A DataStream combines a learner and an optimizer. It is able to cache
 * training data and will start an optimization epoch if the cache is full.
 * Training instances will be added incrementally with
 * DataStream::addSample(). The optimization algorithm that will be passed
 * with DataStream::setOptimizer() should work for online learning like e.g.
 * MBSGD.
 */
class DataStream
{
  int cacheSize, collected;
  Eigen::MatrixXd X, T;
  DirectStorageDataSet* cache;
  Optimizer* opt; //!< Do not delete this!
  Learner* learner; //!< Do not delete this!
public:
  /**
   * Create a data stream.
   * @param cacheSize size of the internal data cache
   */
  DataStream(int cacheSize);
  ~DataStream();
  /**
   * Add a learner.
   * @param learner model
   * @return this for chaining
   */
  DataStream& setLearner(Learner& learner);
  /**
   * Add an optimizer.
   * @param opt online optimization algorithm
   * @return this for chaining
   */
  DataStream& setOptimizer(Optimizer& opt);
  /**
   * Add a sample of the data distribution. This will start an optimization
   * epoch if the cache is full.
   * @param x input
   * @param t target (can be 0 for unsupervised learning)
   */
  void addSample(Eigen::VectorXd* x, Eigen::VectorXd* t = 0);
private:
  void initialize(int inputs, int outputs);
};

} // namespace OpenANN

#endif // OPENANN_IO_DATA_STREAM_H_