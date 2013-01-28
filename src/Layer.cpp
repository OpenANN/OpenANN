#include <layers/Layer.h>

namespace OpenANN
{

int OutputInfo::outputs()
{
  int prod = 1;
  for(std::vector<int>::const_iterator it = dimensions.begin();
      it != dimensions.end(); it++)
    prod *= *it;
  return prod + bias;
}

}
