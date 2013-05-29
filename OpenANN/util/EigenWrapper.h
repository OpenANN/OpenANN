#pragma once

#include <OpenANN/util/AssertionMacros.h>
#include <Eigen/Dense>
#include <cmath>

template<typename T, int M, int N>
bool equals(const Eigen::Matrix<T, M, N>& a, const Eigen::Matrix<T, M, N>& b, T delta)
{
  Eigen::Matrix<T, M, N> d = a - b;
  for(int i = 0; i < d.rows(); i++) for(int j = 0; j < d.cols(); j++)
    {
      if(std::fabs(d(i, j)) > delta)
        return false;
    }
  return true;
}

#define OPENANN_CHECK_MATRIX_NAN(matrix) \
  OPENANN_CHECK(!isMatrixNan(matrix));

template<typename T, int M, int N>
bool isMatrixNan(const Eigen::Matrix<T, M, N> m)
{
  for(int i = 0; i < m.rows(); i++) for(int j = 0; j < m.cols(); j++)
    {
      if(std::isnan(m(i, j)))
        return true;
    }
  return false;
}

#define OPENANN_CHECK_MATRIX_INF(matrix) \
  OPENANN_CHECK(!isMatrixInf(matrix));

template<typename T, int M, int N>
bool isMatrixInf(const Eigen::Matrix<T, M, N> m)
{
  for(int i = 0; i < m.rows(); i++) for(int j = 0; j < m.cols(); j++)
    {
      if(std::isinf(m(i, j)))
        return true;
    }
  return false;
}

#define OPENANN_CHECK_MATRIX_BROKEN(matrix) \
  OPENANN_CHECK(!isMatrixBroken(matrix));

template<typename T, int M, int N>
bool isMatrixBroken(const Eigen::Matrix<T, M, N> m)
{
  return isMatrixNan(m) || isMatrixInf(m);
}
