#ifndef OPENANN_IO_LOGGER_H_
#define OPENANN_IO_LOGGER_H_

#include <string>
#include <iostream>
#include <sstream>
#include <fstream>

#ifndef NDEBUG

#define OPENANN_OUTPUT(msg) std::cout << __FILE__ << "(" << __LINE__ << "): " << msg << std::endl;
#define OPENANN_TRACE(msg) std::cerr << __FILE__ << "(" << __LINE__ << "): " << msg << std::endl;

#else // NDEBUG

#define OPENANN_OUTPUT(msg) std::cout << msg << std::endl;
#define OPENANN_TRACE(msg)

#endif // NDEBUG

#ifndef OPENANN_LOG_NAMESPACE
/**
 * Namespace for global logging.
 */
#define OPENANN_LOG_NAMESPACE NULL
#endif // OPENANN_LOG_NAMESPACE

#ifndef OPENANN_LOGLEVEL
/**
 * Log level for global logging.
 * \sa OpenANN::Log::LogLevel
 */
#define OPENANN_LOGLEVEL OpenANN::Log::DEBUG
#endif // OPENANN_LOGLEVEL

/**
 * Log a "level" message.
 * @param level log level
 */
#define OPENANN_LOG(level) \
  if(level > OPENANN_LOGLEVEL) ; \
  else if(level > OpenANN::Log::getLevel()) ; \
  else OpenANN::Log().get(level, OPENANN_LOG_NAMESPACE)

/**
 * Log a debug message.
 */
#define OPENANN_DEBUG OPENANN_LOG(OpenANN::Log::DEBUG)
/**
 * Log an info message.
 */
#define OPENANN_INFO OPENANN_LOG(OpenANN::Log::INFO)
/**
 * Log an error message.
 */
#define OPENANN_ERROR OPENANN_LOG(OpenANN::Log::ERROR)

namespace OpenANN
{

/**
 * @struct FloatingPointFormatter
 *
 * Wraps a value and its precision for logging.
 */
struct FloatingPointFormatter
{
  /**
   * Floating point number that will be logged.
   */
  double value;
  /**
   * Number of digits after decimal point.
   */
  int precision;

  FloatingPointFormatter(double value, int precision)
    : value(value), precision(precision)
  {}
};

/**
 * @class Log
 *
 * Global logger.
 */
class Log
{
public:
  enum LogLevel
  {
    DISABLED = 0, //!< Disable logging completely.
    ERROR,        //!< Only errors will be logged.
    INFO,         //!< Interesting runtime events.
    DEBUG         //!< Detailed information.
  };

  Log();
  virtual ~Log();

  std::ostream& get(LogLevel level, const char* name_space);

  static std::ostream& getStream();
  static LogLevel& getLevel();

  static void setDisabled();
  static void setError();
  static void setInfo();
  static void setDebug();
private:
  std::ostringstream message;
  LogLevel level;
};


/**
 * @class Logger
 *
 * A local logger that can redirect messages to several targets.
 */
class Logger
{
public:
  /**
   * Deactivate all Logger objects.
   */
  static bool deactivate;
  /**
   * Target of the Logger.
   */
  enum Target
  {
    NONE,       //!< Log nothing.
    CONSOLE,    //!< Log to stdout.
    FILE,       //!< Log to file "name".log.
    APPEND_FILE //!< Log to file "name"-"time".log.
  } target;

  std::string name;
  std::ofstream file;

  Logger(Target target, std::string name = "Logger");
  ~Logger();
  /**
   * Check if the logger redirects to an existing target.
   * @return is the logger activated?
   */
  bool isActive();
};


std::ostream& operator<<(std::ostream& os, const FloatingPointFormatter& t);
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

} // namespace OpenANN

#endif // OPENANN_IO_LOGGER_H_
