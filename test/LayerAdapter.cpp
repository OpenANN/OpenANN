#include "LayerAdapter.h"
#include <OpenANN/io/DataSet.h>

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

Eigen::VectorXd LayerAdapter::currentParameters()
{
  Eigen::VectorXd params(dimension());
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
  return (diff * diff.transpose()).eval()(0, 0) / 2.0;
}

Eigen::VectorXd LayerAdapter::gradient()
{
  Eigen::MatrixXd* output;
  layer.forwardPropagate(&input, output, false);
  Eigen::MatrixXd diff = *output - desired;
  Eigen::MatrixXd* e = &diff;
  layer.backpropagate(e, e);
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
  Eigen::MatrixXd* e = &diff;
  layer.backpropagate(e, e);
  return *e;
}

Eigen::MatrixXd LayerAdapter::hessian()
{
  return Eigen::MatrixXd::Random(dimension(), dimension());
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
  input = trainingInput.row(0);
  desired = trainingOutput.row(0);
}

OpenANN::Learner& LayerAdapter::trainingSet(OpenANN::DataSet& trainingSet)
{
  throw OpenANN::OpenANNException("trainingSet is not implemented in "
                                  "LayerAdapter");
}
