#include <Preprocessing.h>
#include <OpenANNException.h>

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

}


