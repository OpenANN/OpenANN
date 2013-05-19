#pragma once

#include <OpenANN/Learner.h>
#include <OpenANN/util/OpenANNException.h>
#include <OpenANN/layers/Layer.h>
#include <Eigen/Dense>

class LayerAdapter : public OpenANN::Learner
{
  OpenANN::Layer& layer;
  std::vector<double*> parameters;
  std::vector<double*> derivatives;
  OpenANN::OutputInfo info;
  Eigen::MatrixXd input;
  Eigen::MatrixXd desired;
public:
  LayerAdapter(OpenANN::Layer& layer, OpenANN::OutputInfo inputs);
  virtual unsigned int dimension();
  virtual Eigen::VectorXd currentParameters();
  virtual void setParameters(const Eigen::VectorXd& parameters);
  virtual double error();
  virtual Eigen::VectorXd gradient();
  Eigen::MatrixXd inputGradient();
  virtual Eigen::MatrixXd hessian();
  virtual void initialize() {}
  virtual bool providesGradient() { return true; }
  virtual bool providesHessian() { return false; }
  virtual bool providesInitialization() { return true; }
  virtual Eigen::VectorXd operator()(const Eigen::VectorXd& x);
  virtual Eigen::MatrixXd operator()(const Eigen::MatrixXd& X);
  virtual Learner& trainingSet(Eigen::MatrixXd& trainingInput,
                               Eigen::MatrixXd& trainingOutput);
  virtual Learner& trainingSet(OpenANN::DataSet& trainingSet);
};
