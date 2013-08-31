#include <OpenANN/util/EigenWrapper.h>
#include <cstring>

void pack(Eigen::VectorXd& vec, int components, ...)
{
  std::va_list pointers;
  va_start(pointers, components);

  int offset = 0;
  for(int n = 0; n < components; n++)
  {
    int size = va_arg(pointers, int);
    void* p = va_arg(pointers, void*);
    std::memcpy(vec.data()+offset, p, size*sizeof(double));
    offset += size;
  }

  va_end(pointers);
}

void unpack(const Eigen::VectorXd& vec, int components, ...)
{
  std::va_list pointers;
  va_start(pointers, components);

  int offset = 0;
  for(int n = 0; n < components; n++)
  {
    int size = va_arg(pointers, int);
    void* p = va_arg(pointers, void*);
    std::memcpy(p, vec.data()+offset, size*sizeof(double));
    offset += size;
  }

  va_end(pointers);
}
