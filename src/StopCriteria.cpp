#include <StopCriteria.h>
#include <limits>

namespace OpenANN {

StopCriteria StopCriteria::defaultValue;

StopCriteria::StopCriteria()
    : maximalFunctionEvaluations(-1),
      maximalIterations(-1),
      maximalRestarts(0),
      minimalValue(-std::numeric_limits<fpt>::max()),
      minimalValueDifferences(0),
      minimalSearchSpaceStep(0)
{
}

}
