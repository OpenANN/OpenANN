#include "LayerAdapter.h"
#include <OpenANN/io/DataSet.h>
#include <OpenANN/util/AssertionMacros.h>

LayerAdapter::LayerAdapter(OpenANN::Layer& layer,
                                   OpenANN::OutputInfo inputs)
  : layer(layer)
{
  info = layer.initialize(parameters, derivatives);
  input = Eigen::VectorXd::Random(inputs.outputs()).transpose();
  desired = Eigen::VectorXd::Random(info.outputs()).transpose();
}

unsigned int LayerAdapter::dimension()
{
  return parameters.size();
}

unsigned int LayerAdapter::examples()
{
  return input.rows();
}

const Eigen::VectorXd& LayerAdapter::currentParameters()
{
  params.conservativeResize(dimension());
  std::vector<double*>::const_iterator it = parameters.begin();
  for(int i = 0; i < dimension(); i++, it++)
    params(i) = **it;
  return params;
}

void LayerAdapter::setParameters(const Eigen::VectorXd& parameters)
{
  std::vector<double*>::const_iterator it = this->parameters.begin();
  for(int i = 0; i < dimension(); i++, it++)
    **it = parameters(i);
  layer.updatedParameters();
}

double LayerAdapter::error()
{
  Eigen::MatrixXd* output;
  layer.forwardPropagate(&input, output, false);
  Eigen::MatrixXd diff = (*output) - desired;
  return (diff * diff.transpose()).diagonal().sum() / 2.0;
}

double LayerAdapter::error(unsigned int n)
{
  Eigen::MatrixXd* output;
  layer.forwardPropagate(&input, output, false);
  Eigen::MatrixXd diff = ((*output) - desired).row(n);
  return (diff * diff.transpose()).sum() / 2.0;
}

Eigen::VectorXd LayerAdapter::error(std::vector<int>::const_iterator startN,
                                    std::vector<int>::const_iterator endN)
{
  // Assumes that we want to comput the gradient of the whole training set
  OPENANN_CHECK_EQUALS(*startN, 0);
  OPENANN_CHECK_EQUALS(endN-startN, input.rows());
  Eigen::MatrixXd* output;
  layer.forwardPropagate(&input, output, false);
  Eigen::MatrixXd diff = ((*output) - desired);
  return (diff * diff.transpose()).diagonal() / 2.0;
}

Eigen::VectorXd LayerAdapter::gradient()
{
  Eigen::MatrixXd* output;
  layer.forwardPropagate(&input, output, false);
  Eigen::MatrixXd diff = *output - desired;
  double error = 0;
  Eigen::MatrixXd* e = &diff;
  layer.backpropagate(e, e, true, error);
  Eigen::VectorXd derivs(dimension());
  std::vector<double*>::const_iterator it = derivatives.begin();
  for(int i = 0; i < dimension(); i++, it++)
    derivs(i) = **it;
  return derivs;
}

Eigen::VectorXd LayerAdapter::gradient(unsigned int n)
{
  Eigen::MatrixXd in = input.row(n);
  Eigen::MatrixXd out = desired.row(n);

  Eigen::MatrixXd* output;
  layer.forwardPropagate(&in, output, false);
  Eigen::MatrixXd diff = *output - out;
  double error = 0;
  Eigen::MatrixXd* e = &diff;
  layer.backpropagate(e, e, true, error);
  Eigen::VectorXd derivs(dimension());
  std::vector<double*>::const_iterator it = derivatives.begin();
  for(int i = 0; i < dimension(); i++, it++)
    derivs(i) = **it;
  return derivs;
}

Eigen::VectorXd LayerAdapter::gradient(std::vector<int>::const_iterator startN,
                                       std::vector<int>::const_iterator endN)
{
  // Assumes that we want to comput the gradient of the whole training set
  OPENANN_CHECK_EQUALS(*startN, 0);
  OPENANN_CHECK_EQUALS(endN-startN, input.rows());
  Eigen::MatrixXd* output;
  layer.forwardPropagate(&input, output, false);
  Eigen::MatrixXd diff = *output - desired;
  double error = 0;
  Eigen::MatrixXd* e = &diff;
  layer.backpropagate(e, e, true, error);
  Eigen::VectorXd derivs(dimension());
  std::vector<double*>::const_iterator it = derivatives.begin();
  for(int i = 0; i < dimension(); i++, it++)
    derivs(i) = **it;
  return derivs;
}

Eigen::MatrixXd LayerAdapter::inputGradient()
{
  Eigen::MatrixXd* output;
  layer.forwardPropagate(&input, output, false);
  Eigen::MatrixXd diff = *output - desired;
  double error = 0;
  Eigen::MatrixXd* e = &diff;
  layer.backpropagate(e, e, true, error);
  return *e;
}

Eigen::VectorXd LayerAdapter::operator()(const Eigen::VectorXd& x)
{
  this->input = x;
  Eigen::MatrixXd* output;
  layer.forwardPropagate(&input, output, false);
  return *output;
}

Eigen::MatrixXd LayerAdapter::operator()(const Eigen::MatrixXd& X)
{
  this->input = X;
  Eigen::MatrixXd* output;
  layer.forwardPropagate(&input, output, false);
  return *output;
}

OpenANN::Learner& LayerAdapter::trainingSet(Eigen::MatrixXd& trainingInput,
                                            Eigen::MatrixXd& trainingOutput)
{
  input = trainingInput;
  desired = trainingOutput;
  return *this;
}

OpenANN::Learner& LayerAdapter::trainingSet(OpenANN::DataSet& trainingSet)
{
  throw OpenANN::OpenANNException("trainingSet is not implemented in "
                                  "LayerAdapter");
}
