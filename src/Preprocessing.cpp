#include <OpenANN/Preprocessing.h>
#include <OpenANN/util/OpenANNException.h>
#include <OpenANN/util/Random.h>
#include <vector>

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

Eigen::MatrixXd sampleRandomPatches(const Eigen::MatrixXd& images,
                                    int channels, int rows, int cols,
                                    int samples, int patchRows, int patchCols)
{
  RandomNumberGenerator rng;
  const int channelSize = rows * cols;
  const int patchSize = channels * patchRows * patchCols;
  const int totalPatches = images.rows() * samples;
  Eigen::MatrixXd patches(totalPatches, patchSize);

  std::vector<std::pair<int, int> > patchIndices;
  patchIndices.reserve(samples);
  for(int n = 0; n < totalPatches; n++)
    patchIndices.push_back(std::make_pair<int>(
      rng.generateInt(0, rows - patchRows + 1),
      rng.generateInt(0, cols - patchCols + 1)));

#pragma omp parallel for
  for(int m = 0; m < images.rows(); ++m)
  {
    for(int n = 0; n < samples; ++n)
    {
      const int patchIdx = m*samples+n;
      const int rowStart = patchIndices[patchIdx].first;
      const int colStart = patchIndices[patchIdx].second;
      for(int chan = 0, channelOffset = 0, pxIdx = 0; chan < channels;
          ++chan, channelOffset += channelSize)
      {
        for(int row = 0; row < patchRows; ++row)
          for(int col = 0; col < patchCols; ++col)
            patches(patchIdx, pxIdx++) =
                images(m, channelOffset+(rowStart+row)*cols+colStart+col);
      }
    }
  }
  return patches;
}

}
