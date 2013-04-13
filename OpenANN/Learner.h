#pragma once

#include <OpenANN/io/DataSet.h>

namespace OpenANN {

/**
 * @class Learner
 * Common base class of all learning algorithms.
 */
class Learner
{
public:
  /**
   * Set the current training set.
   * @param trainingInput input vectors, each instance should be in a new column
   * @param trainingOutput output vectors, each instance should be in a new
   *                       column
   */
  virtual Learner& trainingSet(Mt& trainingInput, Mt& trainingOutput) = 0;
  /**
   * Set the current training set.
   * @param trainingSet custom training set
   */
  virtual Learner& trainingSet(DataSet& trainingSet) = 0;
  /**
   * Make a prediction.
   * @param x Input vector.
   */
  virtual Vt operator()(const Vt& x) = 0;
};

}