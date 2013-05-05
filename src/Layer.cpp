#include <OpenANN/layers/Layer.h>
#include <numeric>

namespace OpenANN
{

int OutputInfo::outputs()
{
  return std::accumulate(dimensions.begin(), dimensions.end(), 1,
                         std::multiplies<int>());;
}

}
