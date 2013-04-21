#pragma once

#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>

#ifndef NDEBUG

#ifndef OPENNANN_LOGLEVEL 
  #define OPENANN_LOGLEVEL OpenANN::Log::DEBUG 
#endif // OPENANN_LOGLEVEL

#define OPENANN_OUTPUT(msg) std::cout << __FILE__ << "(" << __LINE__ << "): " << msg << std::endl;
#define OPENANN_TRACE(msg) std::cerr << __FILE__ << "(" << __LINE__ << "): " << msg << std::endl;

#else // NDEBUG

#ifndef OPENNANN_LOGLEVEL 
  #define OPENANN_LOGLEVEL OpenANN::Log::INFO 
#endif // OPENANN_LOGLEVEL

#define OPENANN_OUTPUT(msg) std::cout << msg << std::endl;
#define OPENANN_TRACE(msg)

#endif // NDEBUG

#ifndef OPENANN_LOG_NAMESPACE
  #define OPENANN_LOG_NAMESPACE NULL
#endif

#define OPENANN_LOG(level) \
    if(level > OPENANN_LOGLEVEL) ; \
    else if(level > OpenANN::Log::getLevel()) ; \
    else OpenANN::Log().get(level, OPENANN_LOG_NAMESPACE)

#define OPENANN_DEBUG OPENANN_LOG(OpenANN::Log::DEBUG)
#define OPENANN_INFO OPENANN_LOG(OpenANN::Log::INFO)
#define OPENANN_ERROR OPENANN_LOG(OpenANN::Log::ERROR)

namespace OpenANN
{

struct FloatingPointFormatter
{
  double value;
  int precision;
  FloatingPointFormatter(double value, int precision);
};

std::ostream& operator<<(std::ostream& os, const FloatingPointFormatter& t);

class Log 
{
public:
  enum LogLevel {
      DISABLED = 0,
      ERROR,
      INFO, 
      DEBUG
  };

  Log();
  virtual ~Log();

  std::ostream& get(LogLevel level, const char* name_space);

  static std::ostream& getStream();
  static LogLevel& getLevel();

private:
  std::ostringstream message;
  LogLevel level;
};


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
