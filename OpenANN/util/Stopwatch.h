#ifndef OPENANN_UTIL_STOPWATCH_H_
#define OPENANN_UTIL_STOPWATCH_H_

#include <sys/time.h>

class Stopwatch
{
  unsigned long begin, duration;

  /** @return The system time with microsecond precision. */
  inline unsigned long getSystime()
  {
    timeval t;
    gettimeofday(&t,0);
    return (unsigned long) t.tv_sec * 1000000L + (unsigned long) t.tv_usec;
  }
public:
  enum Precision {MICROSECOND, MILLISECOND, HUNDREDTHS, TENTHS, SECONDS};

  /** Starts the stopwatch. */
  Stopwatch() : begin(getSystime()), duration(0) {};

  /** Starts or restarts the stopwatch. */
  void start()
  {
    begin = getSystime();
  }

  /**
   * This does not really stop the stopwatch. It just calculates the duration
   * since start.
   * @return The passed time since start.
   */
  inline unsigned long stop()
  {
    duration = getSystime() - begin;
    return duration;
  }

  /**
   * This does not really stop the stopwatch. It just calculates the duration
   * since start.
   * @return The passed time since start.
   */
  inline unsigned long stop(Precision p)
  {
    unsigned long duration = stop();
    switch(p)
    {
    case SECONDS:
      return duration / 1000000L;
    case TENTHS:
      return duration / 100000L;
    case HUNDREDTHS:
      return duration / 10000L;
    case MILLISECOND:
      return duration / 1000L;
    case MICROSECOND:
    default:
      return duration;
    }
  }

  /** 
   * @return The last calculated duration. Requires that stop() has been
   *    called before. Otherwise it returns 0.
   */
  inline unsigned long getDuration()
  {
    return duration;
  }
};

#endif // OPENANN_UTIL_STOPWATCH_H_

