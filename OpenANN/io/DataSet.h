#pragma once

#include <Eigen/Dense>

namespace OpenANN {

class MLP;

class DataSet
{
public:
  virtual ~DataSet() {}
  virtual int samples() = 0;
  virtual int inputs() = 0;
  virtual int outputs() = 0;
  virtual Vt& getInstance(int i) = 0;
  virtual Vt& getTarget(int i) = 0;
  virtual void finishIteration(MLP& mlp) = 0;
};

}
