#include <OpenANN/optimization/StoppingCriteria.h>
#include <limits>

namespace OpenANN {

StoppingCriteria StoppingCriteria::defaultValue;

StoppingCriteria::StoppingCriteria()
    : maximalFunctionEvaluations(-1),
      maximalIterations(-1),
      maximalRestarts(0),
      minimalValue(-std::numeric_limits<fpt>::max()),
      minimalValueDifferences(0),
      minimalSearchSpaceStep(0)
{
}

}
