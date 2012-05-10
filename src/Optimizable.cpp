#include <Optimizable.h>
#include <AssertionMacros.h>

namespace OpenANN {

void Optimizable::VJ(Vt& values, Mt& jacobian)
{
  OPENANN_CHECK_EQUALS(values.rows(), (int) examples());
  OPENANN_CHECK_EQUALS(jacobian.rows(), (int) examples());
  OPENANN_CHECK_EQUALS(jacobian.cols(), (int) dimension());
  for(unsigned n = 0; n < examples(); n++)
  {
    values(n) = error(n);
    jacobian.row(n) = gradient(n);
  }
}

Vt Optimizable::singleGradientFD(int n, const fpt eps)
{
  Vt gradient(dimension(), 1);
  gradient.fill(0.0);
  Vt params = currentParameters();
  Vt modifiedParams = params;
  for(unsigned i = 0; i < dimension(); i++)
  {
    modifiedParams(i, 0) += eps;
    setParameters(modifiedParams);
    fpt errorPlusEps = error(n);
    modifiedParams = params;

    modifiedParams(i, 0) -= eps;
    setParameters(modifiedParams);
    fpt errorMinusEps = error(n);
    modifiedParams = params;

    gradient(i, 0) = (errorPlusEps - errorMinusEps) / (2.0 * eps);
  }
  setParameters(params);
  return gradient;
}

Vt Optimizable::gradientFD(const fpt eps)
{
  Vt gradient(dimension(), 1);
  gradient.fill(0.0);
  Vt params = currentParameters();
  Vt modifiedParams = params;
  for(unsigned i = 0; i < dimension(); i++)
  {
    modifiedParams(i, 0) += eps;
    setParameters(modifiedParams);
    fpt errorPlusEps = error();
    modifiedParams = params;

    modifiedParams(i, 0) -= eps;
    setParameters(modifiedParams);
    fpt errorMinusEps = error();
    modifiedParams = params;

    gradient(i, 0) = (errorPlusEps - errorMinusEps) / (2.0 * eps);
  }
  setParameters(params);
  return gradient;
}

Mt Optimizable::hessianFD(const fpt eps)
{
  Mt hessian(dimension(), dimension());
  hessian.fill(0.0);
  Vt params = currentParameters();
  Vt modifiedParams = params;
  for(unsigned i = 0; i < dimension(); i++)
  {
    for(unsigned j = 0; j <= i; j++)
    {
      modifiedParams(i, 0) += eps;
      modifiedParams(j, 0) += eps;
      setParameters(modifiedParams);
      fpt plusplus = error();
      modifiedParams = params;

      modifiedParams(i, 0) += eps;
      modifiedParams(j, 0) -= eps;
      setParameters(modifiedParams);
      fpt plusminus = error();
      modifiedParams = params;

      modifiedParams(i, 0) -= eps;
      modifiedParams(j, 0) += eps;
      setParameters(modifiedParams);
      fpt minusplus = error();
      modifiedParams = params;

      modifiedParams(i, 0) -= eps;
      modifiedParams(j, 0) -= eps;
      setParameters(modifiedParams);
      fpt minusminus = error();
      modifiedParams = params;

      hessian(i, j) = (plusplus - plusminus - minusplus + minusminus) / (4.0 * eps*eps);
      if(i != j)
        hessian(j, i) = hessian(i, j);
    }
  }
  setParameters(params);
  return hessian;
}

}
