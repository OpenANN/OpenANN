#include <OpenANN/optimization/StoppingInterrupt.h>
#include <OpenANN/util/AssertionMacros.h>

#include <csignal>

namespace OpenANN {


bool StoppingInterrupt::stopping_interrupt_signal = false;
int  StoppingInterrupt::observers = 0;


StoppingInterrupt::StoppingInterrupt()
{
    OPENANN_CHECK(!stopping_interrupt_signal);

    if(observers == 0) 
        std::signal(SIGINT, StoppingInterrupt::setStoppingInterruptSignal);

    ++observers;
}

StoppingInterrupt::~StoppingInterrupt()
{
    --observers;

    if(observers == 0) {
        std::signal(SIGINT, SIG_DFL);
        stopping_interrupt_signal = false;
    }
}


void StoppingInterrupt::setStoppingInterruptSignal(int param) 
{
    stopping_interrupt_signal = true;
}


bool StoppingInterrupt::isSignaled()
{
    OPENANN_CHECK(observers > 0);

    return stopping_interrupt_signal;
}

}
