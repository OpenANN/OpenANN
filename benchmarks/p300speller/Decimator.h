#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <complex>

#include <iostream>
class Decimator
{
public:
  int downSamplingFactor;
  Vt a, b;

  Decimator(int downSamplingFactor)
    : downSamplingFactor(downSamplingFactor)
  {
  }

  /**
   * Downsample the signal by an integer factor, using a filter. An order 8
   * Chebyshev type I filter is used.
   */
  Mt decimate(const Mt& x)
  {
    // Low pass filter
    // scipy:
    //  signal.firwin(30+1, 10.0/fs, window='hamming')
    Mt y1(x.rows(), x.cols());
    Vt b1(31);
    b1 << 0.00256293,  0.0032317 ,  0.00475108,  0.00727558,  0.01088579,
        0.01557498,  0.02124304,  0.02769838,  0.03466804,  0.04181521,
        0.04876293,  0.05512206,  0.06052093,  0.06463443,  0.06720971,
        0.06808644,  0.06720971,  0.06463443,  0.06052093,  0.05512206,
        0.04876293,  0.04181521,  0.03466804,  0.02769838,  0.02124304,
        0.01557498,  0.01088579,  0.00727558,  0.00475108,  0.0032317 ,
        0.00256293;
    Vt a1(1);
    a1 << 1.;
    filter(x, y1, b1, a1);

    Mt d(x.rows(), x.cols()/downSamplingFactor);
    downsample(y1, d);
    return d;
  }

  /**
   * Apply a filter on the input signal.
   * See scipy, example:
   *    >>> import scipy.signal as signal
   *        fs = 240
   *        signal.cheby1(8, 0.05, [0.1/(fs/2), 10.0/(fs/2)],btype='band')
   */
  void filter(const Mt& x, Mt& y, const Mt& b, const Mt& a)
  {
    y.fill(0.0);

    for(int c = 0; c < x.rows(); c++)
    {
      for(int t = 0; t < x.cols(); t++)
      {
        const int maxPQ = std::max(b.rows(), a.rows());
        for(int pq = 0; pq < maxPQ; pq++)
        {
          const double tSource = t-pq;
          if(pq < b.rows())
          {
            if(tSource >= 0)
              y(c, t) += b(pq) * x(c, tSource);
            else
              y(c, t) += b(pq) * x(c, -tSource);
          }
          if(pq > 0 && pq < a.rows())
          {
            if(tSource >= 0)
              y(c, t) += a(pq) * x(c, tSource);
            else
              y(c, t) += a(pq) * x(c, -tSource);
          }
        }
        y(c, t) /= a(0);
      }
    }
  }

  void downsample(const Mt& y, Mt& d)
  {
    for(int c = 0; c < y.rows(); c++)
      for(int target = 0, source = 0; target < d.cols();
          target++, source+=downSamplingFactor)
        d(c, target) = y(c, source);
  }
};