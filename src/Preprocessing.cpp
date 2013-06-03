#include <OpenANN/Preprocessing.h>
#include <OpenANN/util/OpenANNException.h>

namespace OpenANN
{

void scaleData(Eigen::MatrixXd& data, double min, double max)
{
  if(min >= max)
    throw OpenANNException("Scaling failed: max has to be greater than min!");
  const double minData = data.minCoeff();
  const double maxData = data.maxCoeff();
  const double dataRange = maxData - minData;
  const double desiredRange = max - min;
  const double scaling = desiredRange / dataRange;
  data = data.array() * scaling + (min - minData * scaling);
}

void filter(const Eigen::MatrixXd& x, Eigen::MatrixXd& y, const Eigen::MatrixXd& b, const Eigen::MatrixXd& a)
{
  y.setZero();

  for(int c = 0; c < x.rows(); c++)
  {
    for(int t = 0; t < x.cols(); t++)
    {
      const int maxPQ = std::max(b.rows(), a.rows());
      for(int pq = 0; pq < maxPQ; pq++)
      {
        const double tSource = t - pq;
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

void downsample(const Eigen::MatrixXd& y, Eigen::MatrixXd& d, int downSamplingFactor)
{
  for(int c = 0; c < y.rows(); c++)
    for(int target = 0, source = 0; target < d.cols();
        target++, source += downSamplingFactor)
      d(c, target) = y(c, source);
}

}


