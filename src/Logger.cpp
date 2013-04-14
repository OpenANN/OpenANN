#include <OpenANN/io/Logger.h>
#include <OpenANN/util/AssertionMacros.h>
#include <ctime>

namespace OpenANN
{

const char* LevelToString[] = {
    "DISABLED",
    "ERROR",
    "INFO",
    "DEBUG"
};

bool Logger::deactivate = false;


Logger::Logger(Target target, std::string name)
  : target(target), name(name)
{
  if(deactivate)
  {
    this->target = NONE;
  }
  else if(target == FILE)
  {
    time_t rawtime;
    struct tm* timeinfo;
    std::time(&rawtime);
    timeinfo = std::localtime(&rawtime);
    file.open((name + "-" + std::string(std::asctime(timeinfo)).substr(0, 24) + ".log").c_str());
    OPENANN_CHECK(file.is_open());
  }
  else if(target == APPEND_FILE)
  {
    file.open((name + ".log").c_str(), std::fstream::app);
    OPENANN_CHECK(file.is_open());
  }
}


Logger::~Logger()
{
  if(file.is_open())
    file.close();
}


bool Logger::isActive()
{
  return target != NONE;
}


FloatingPointFormatter::FloatingPointFormatter(fpt value, int precision)
  : value(value), precision(precision)
{
}


Logger& operator<<(Logger& logger, const FloatingPointFormatter& t)
{
  switch(logger.target)
  {
    case Logger::CONSOLE:
      std::cout << std::fixed << std::setprecision(t.precision) << t.value << std::resetiosflags(std::ios_base::fixed) << std::flush;
      break;
    case Logger::APPEND_FILE:
    case Logger::FILE:
      logger.file << std::fixed << std::setprecision(t.precision) << t.value << std::flush;
      break;
    default: // do not log
      break;
  }
  return logger;
}


Log::Log()
{
}


Log::~Log()
{
     Stream() << message.str() << std::endl; 
}


 std::ostream& Log::get(LogLevel level)
{
    OPENANN_CHECK(level != kDisabled);

    time_t now;
    time(&now);
    struct tm* current = localtime(&now);
    char current_time[32];

    this->level = level;
    
    strftime(current_time, sizeof(current_time), "%F %X", current);

    message << std::setw(6) << LevelToString[level] << "  " << current_time << "  ";

    return message;
}


std::ostream& Log::Stream()
{
    static std::ostream& gStream = std::cout;
    return gStream;
}


Log::LogLevel& Log::Level()
{
    static Log::LogLevel gLevel = kDebug;
    return gLevel;
}

std::ostream& operator<<(std::ostream& os, const FloatingPointFormatter& t)
{
      os << std::fixed << std::setprecision(t.precision) << t.value << std::resetiosflags(std::ios_base::fixed);
      return os;
}



}
