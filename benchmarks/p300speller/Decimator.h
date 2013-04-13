#ifndef DECIMATOR_H
#define DECIMATOR_H

#include <OpenANN/Preprocessing.h>
#include <Eigen/Dense>
#include <cmath>

class Decimator
{
  int downSamplingFactor;
  Vt a, b;

public:
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
    OpenANN::filter(x, y1, b1, a1);

    Mt d(x.rows(), x.cols()/downSamplingFactor);
    OpenANN::downsample(y1, d, downSamplingFactor);
    return d;
  }
};

#endif // DECIMATOR_H
