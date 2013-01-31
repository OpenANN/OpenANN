#include <OpenANN>
#include <DeepNetwork.h>
#include <Test/Stopwatch.h>
#include <io/Logger.h>
#include <io/DirectStorageDataSet.h>

using namespace OpenANN;

int main()
{
  int D = 1000;
  int F = 5;
  int N = 10;
  Logger logger(Logger::CONSOLE);
  Mt X(D, N);
  Mt Y(F, N);
  DirectStorageDataSet ds(X, Y);

  {
    Stopwatch sw;
    MLP mlp;
    mlp.input(D);
    mlp.output(F);
    mlp.trainingSet(ds);
    logger << "Construct MLP: " << sw.stop(Stopwatch::MILLISECOND) << " ms\n";

    mlp.gradient();
    logger << "Gradient MLP: " << sw.stop(Stopwatch::MILLISECOND) << " ms\n";

    mlp.gradientFD();
    logger << "GradientFD MLP: " << sw.stop(Stopwatch::MILLISECOND) << " ms\n";
  }

  {
    Stopwatch sw;
    DeepNetwork net(DeepNetwork::SSE);
    net.inputLayer(D);
    net.outputLayer(F, LINEAR);
    net.trainingSet(ds);
    logger << "Construct Net: " << sw.stop(Stopwatch::MILLISECOND) << " ms\n";

    net.gradient();
    logger << "Gradient Net: " << sw.stop(Stopwatch::MILLISECOND) << " ms\n";

    net.gradientFD();
    logger << "GradientFD Net: " << sw.stop(Stopwatch::MILLISECOND) << " ms\n";
  }
}
