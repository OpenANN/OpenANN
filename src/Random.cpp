#include <OpenANN/util/Random.h>
#include <ctime>

namespace OpenANN
{

RandomNumberGenerator::RandomNumberGenerator()
{
#if __cplusplus < 201300L
  static bool seedInitialized = false;
  if(!seedInitialized)
  {
    srand(std::time(0));
    seedInitialized = true;
  }
#else	
	m_generator.seed(std::time(0));
#endif
}

void RandomNumberGenerator::seed(unsigned int seed)
{
#if __cplusplus < 201300L
  srand(seed);
#else	
	m_generator.seed(seed);
#endif
}

int RandomNumberGenerator::generateInt(int min, int range) const
{
  OPENANN_CHECK(range >= 0);
  if(range == 0)
    return min;
  else
#if __cplusplus < 201300L
    return rand() % range + min;
#else 
		return m_distribution(m_generator) % range + min;
#endif		
}

size_t RandomNumberGenerator::generateIndex(size_t size) const
{
  return (size_t) generateInt(0, size);
}

}
