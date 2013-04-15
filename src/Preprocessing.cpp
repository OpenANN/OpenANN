#include <OpenANN/Preprocessing.h>
#include <OpenANN/util/OpenANNException.h>

namespace OpenANN {

void scaleData(Mt& data, fpt min, fpt max)
{
  if(min >= max)
    throw OpenANNException("Scaling failed: max has to be greater than min!");
  const fpt minData = data.minCoeff();
  const fpt maxData = data.maxCoeff();
  const fpt dataRange = maxData - minData;
  const fpt range = max - min;
  for(int i = 0; i < data.rows(); i++)
    for(int j = 0; j < data.cols(); j++)
      data(i, j) = (data(i, j)-minData) / dataRange * range + min;
}

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

void downsample(const Mt& y, Mt& d, int downSamplingFactor)
{
  for(int c = 0; c < y.rows(); c++)
    for(int target = 0, source = 0; target < d.cols();
        target++, source+=downSamplingFactor)
      d(c, target) = y(c, source);
}

}


