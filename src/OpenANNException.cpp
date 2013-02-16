#include <OpenANNException.h>

OpenANNException::OpenANNException(const std::string& msg)
    : std::logic_error(msg)
{
}