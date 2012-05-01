#include <Optimizable.h>
#include <AssertionMacros.h>

class Himmelblau : public OpenANN::Optimizable
{
  Vt x;
public:
  Himmelblau()
      : x(2, 1)
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

  virtual Vt currentParameters()
  {
    return x;
  }

  virtual void setParameters(const Vt& parameters)
  {
    x(0, 0) = parameters(0, 0);
    x(1, 0) = parameters(1, 0);
  }

  virtual fpt error()
  {
    return pow(x(0,0)*x(0,0)+x(1,0)-11., 2) + pow(x(0,0)+x(1,0)*x(1,0) - 7., 2);
  }

  virtual bool providesGradient()
  {
    return false;
  }

  virtual Vt gradient()
  {
    OPENANN_CHECK(false && "Himmelblau does not provide a gradient.");
    return Vt();
  }

  virtual bool providesHessian()
  {
    return false;
  }

  virtual Mt hessian()
  {
    OPENANN_CHECK(false && "Himmelblau does not provide an hessian.");
    return Mt();
  }
};
