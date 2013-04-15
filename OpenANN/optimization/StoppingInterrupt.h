#pragma once 

#include <set>

namespace OpenANN {


/**
   * A system-independent interface for checking interrupts that 
   * can signals the end of the optimization process externally. 
   *
   * If a specific signal integration for a plattform is not supported in OpenANN,
   * this method will always return false and have no influence to the
   * stopping criterias.
   *
   * This signal processing is definitly NOT MULTI-THREADED SUPPORTED
   */
class StoppingInterrupt
{
public:
  /** 
   * Register the current interrupt handlers for catching os-relating signals 
   */
  StoppingInterrupt();

  /** 
   * Automatically remove interrupt handlers to old os-related default state
   */
  ~StoppingInterrupt();

  bool isSignaled();

private:
  static void setStoppingInterruptSignal(int param);

private:
  static int observers;
  static bool stopping_interrupt_signal;
};



} 
