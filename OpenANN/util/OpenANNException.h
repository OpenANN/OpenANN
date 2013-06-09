#ifndef OPENANN_UTIL_OPENANN_EXCEPTION_H_
#define OPENANN_UTIL_OPENANN_EXCEPTION_H_

#include <stdexcept>

namespace OpenANN
{

/**
 * @class OpenANNException
 *
 * This exception is thrown for all logical errors that occur in OpenANN API
 * calls that are not time critical.
 */
class OpenANNException : public std::logic_error
{
public:
  OpenANNException(const std::string& msg);
};

} // namespace OpenANN

#endif // OPENANN_UTIL_OPENANN_EXCEPTION_H_
