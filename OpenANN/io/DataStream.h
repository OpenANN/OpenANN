#ifndef OPENANN_IO_DATA_STREAM_H_
#define OPENANN_IO_DATA_STREAM_H_

#include <Eigen/Dense>

namespace OpenANN
{

class DirectStorageDataSet;
class Optimizer;
class Learner;

class DataStream
{
  int cacheSize, collected;
  Eigen::MatrixXd X, T;
  DirectStorageDataSet* cache;
  Optimizer* opt; //!< Do not delete this!
  Learner* learner; //!< Do not delete this!
public:
  DataStream(int cacheSize);
  ~DataStream();
  void setLearner(Learner& learner);
  void setOptimizer(Optimizer& opt);
  void addSample(Eigen::VectorXd* x, Eigen::VectorXd* t = 0);
private:
  void initialize(int inputs, int outputs);
};

} // namespace OpenANN

#endif // OPENANN_IO_DATA_STREAM_H_