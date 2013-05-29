#include <OpenANN/optimization/StoppingInterrupt.h>
#include <OpenANN/util/AssertionMacros.h>
#include <csignal>

namespace OpenANN
{


bool StoppingInterrupt::stoppingInterruptSignal = false;
int  StoppingInterrupt::observers = 0;


StoppingInterrupt::StoppingInterrupt()
{
  OPENANN_CHECK(!stoppingInterruptSignal);

  if(observers == 0)
    std::signal(SIGINT, StoppingInterrupt::setStoppingInterruptSignal);

  ++observers;
}

StoppingInterrupt::~StoppingInterrupt()
{
  --observers;

  if(observers == 0)
  {
    std::signal(SIGINT, SIG_DFL);
    stoppingInterruptSignal = false;
  }
}


void StoppingInterrupt::setStoppingInterruptSignal(int param)
{
  stoppingInterruptSignal = true;
}


bool StoppingInterrupt::isSignaled()
{
  OPENANN_CHECK(observers > 0);
  return stoppingInterruptSignal;
}

}
