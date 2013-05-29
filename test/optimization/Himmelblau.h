#include <OpenANN/optimization/Optimizable.h>
#include <OpenANN/util/AssertionMacros.h>

class Himmelblau : public OpenANN::Optimizable
{
  Eigen::VectorXd x;
public:
  Himmelblau()
    : x(2)
  {
  }

  virtual bool providesInitialization()
  {
    return false;
  }

  virtual void initialize()
  {
    OPENANN_CHECK(false && "Himmelblau does not provide an initialization.");
  }

  virtual unsigned dimension()
  {
    return 2;
  }

  virtual const Eigen::VectorXd& currentParameters()
  {
    return x;
  }

  virtual void setParameters(const Eigen::VectorXd& parameters)
  {
    x(0) = parameters(0);
    x(1) = parameters(1);
  }

  virtual double error()
  {
    return pow(x(0) * x(0) + x(1) - 11., 2) + pow(x(0) + x(1) * x(1) - 7., 2);
  }

  virtual bool providesGradient()
  {
    return false;
  }

  virtual Eigen::VectorXd gradient()
  {
    OPENANN_CHECK(false && "Himmelblau does not provide a gradient.");
    return Eigen::VectorXd();
  }
};
