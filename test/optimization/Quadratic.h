#include <OpenANN/optimization/Optimizable.h>
#include <OpenANN/util/AssertionMacros.h>

template<int N>
class Quadratic : public OpenANN::Optimizable
{
  Eigen::VectorXd x;
public:
  Quadratic()
      : x(N, 1)
  {
  }

  virtual bool providesInitialization()
  {
    return false;
  }

  virtual void initialize()
  {
    OPENANN_CHECK(false && "Quadratic does not provide an initialization.");
  }

  virtual unsigned dimension()
  {
    return N;
  }

  virtual Eigen::VectorXd currentParameters()
  {
    return x;
  }

  virtual void setParameters(const Eigen::VectorXd& parameters)
  {
    x = parameters;
  }

  virtual double error()
  {
    return x.cwiseProduct(x).sum();
  }

  virtual bool providesGradient()
  {
    return true;
  }

  virtual Eigen::VectorXd gradient()
  {
    return 2 * x;
  }

  virtual bool providesHessian()
  {
    return true;
  }

  virtual Eigen::MatrixXd hessian()
  {
    Eigen::MatrixXd hessian(N, N);
    hessian.setIdentity();
    return hessian * 2;
  }
};
