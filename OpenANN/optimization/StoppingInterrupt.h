#ifndef OPENANN_OPTIMIZATION_STOPPING_INTERRIPT_H_
#define OPENANN_OPTIMIZATION_STOPPING_INTERRIPT_H_

namespace OpenANN
{

/**
 * @class StoppingInterrupt
 *
 * A system-independent interface for checking interrupts that can signals the
 * end of the optimization process externally.
 *
 * If a specific signal integration for a plattform is not supported in
 * OpenANN, this method will always return false and have no influence to the
 * stopping criterias.
 *
 * This is definitly NOT THREAD-SAFE!
 */
class StoppingInterrupt
{
  static int observers;
  static bool stoppingInterruptSignal;
public:
  /**
   * Register the current interrupt handlers to catch OS-specific signals.
   */
  StoppingInterrupt();

  /**
   * Automatically remove interrupt handlers to restore old OS-specific
   * default state.
   */
  ~StoppingInterrupt();

  bool isSignaled();

private:
  static void setStoppingInterruptSignal(int param);
};

} // namespace OpenANN

#endif // OPENANN_OPTIMIZATION_STOPPING_INTERRIPT_H_
