#pragma once

#include <OpenANN/Learner.h>
#include <OpenANN/io/DataSet.h>
#include <OpenANN/optimization/Optimizable.h>
#include <Eigen/Dense>

namespace OpenANN
{

/**
 * @class IntrinsicPlasticity
 *
 * Learns the parameters of a logistic sigmoid activation function.
 *
 * Activation functions of the form \f$ y = 1 / (1 + \exp (-s a - b)) \f$
 * with slopes \f$ s \f$ and biases \f$ b \f$ are adapted such that the output
 * distribution is approximately exponential with mean \f$ \mu \f$ and with
 * respect to a input distribution given by a training set. This procedure
 * prevents saturation. Note that changing the incoming weights might require
 * readjustment.
 *
 * [1] Jochen Triesch:
 * A Gradient Rule for the Plasticity of a Neuron’s Intrinsic Excitability,
 * Proceedings of the International Conference on Artificial Neural Networks,
 * pp. 1–7, 2005.
 *
 * [2] Jochen Triesch:
 * Synergies between intrinsic and synaptic plasticity mechanisms,
 * Neural Computation 19, pp. 885-909, 2007.
 */
class IntrinsicPlasticity : public Optimizable, public Learner
{
  const int nodes;
  const fpt mu;
  const fpt stdDev;
  Vt s;
  Vt b;
  Vt parameters;
  Vt g;
  DataSet* dataSet;
  bool deleteDataSet;
  Vt y;
public:
  IntrinsicPlasticity(int nodes, fpt mu, fpt stdDev = 1.0);
  virtual ~IntrinsicPlasticity();

  virtual unsigned int examples();
  virtual unsigned int dimension();
  virtual bool providesInitialization();
  virtual void initialize();
  virtual fpt error();
  virtual fpt error(unsigned int n);
  virtual Vt currentParameters();
  virtual void setParameters(const Vt& parameters);
  virtual bool providesGradient();
  virtual Vt gradient();
  virtual Vt gradient(unsigned int n);
  virtual bool providesHessian();
  virtual Mt hessian();
  virtual Learner& trainingSet(Mt& trainingInput, Mt& trainingOutput);
  virtual Learner& trainingSet(DataSet& trainingSet);
  virtual Vt operator()(const Vt& a);
};

}
