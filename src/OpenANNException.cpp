#include <OpenANN/util/OpenANNException.h>

namespace OpenANN {

OpenANNException::OpenANNException(const std::string& msg)
    : std::logic_error(msg)
{
}

}
