#pragma once

#include <stdexcept>

namespace OpenANN {

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

}
