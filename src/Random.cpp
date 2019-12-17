#include <OpenANN/util/Random.h>
#include <ctime>

namespace OpenANN
{

RandomNumberGenerator::RandomNumberGenerator()
{
  m_generator.seed(std::time(0));
}

void RandomNumberGenerator::seed(unsigned int seed)
{
	m_generator.seed(seed);
}

int RandomNumberGenerator::generateInt(int min, int range) const
{
  OPENANN_CHECK(range >= 0);
  if(range == 0)
    return min;
  else
    return m_distribution(m_generator) % range + min;
}

size_t RandomNumberGenerator::generateIndex(size_t size) const
{
  return (size_t) generateInt(0, size);
}

}
