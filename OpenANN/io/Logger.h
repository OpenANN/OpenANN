#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>

#ifndef NDEBUG

#define OPENANN_OUTPUT(msg) std::cout << __FILE__ << "(" << __LINE__ << "): " << msg << std::endl;
#define OPENANN_TRACE(msg) std::cerr << __FILE__ << "(" << __LINE__ << "): " << msg << std::endl;

#else // NDEBUG

#define OPENANN_OUTPUT(msg) std::cout << msg << std::endl;
#define OPENANN_TRACE(msg)

#endif // NDEBUG

namespace OpenANN
{

class Logger
{
public:
  static bool deactivate;
  enum Target
  {
    NONE, CONSOLE, FILE, APPEND_FILE
  } target;
  std::string name;
  std::ofstream file;

  Logger(Target target, std::string name = "Logger");
  ~Logger();
  bool isActive();
};

struct FloatingPointFormatter
{
  fpt value;
  int precision;
  FloatingPointFormatter(fpt value, int precision);
};

Logger& operator<<(Logger& logger, const FloatingPointFormatter& t);

template<typename T>
Logger& operator<<(Logger& logger, const T& t)
{
  switch(logger.target)
  {
    case Logger::CONSOLE:
      std::cout << t << std::flush;
      break;
    case Logger::APPEND_FILE:
    case Logger::FILE:
      logger.file << t << std::flush;
      break;
    default: // do not log
      break;
  }
  return logger;
}

}
