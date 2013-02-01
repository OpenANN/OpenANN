#include <OpenANN>
#include <DeepNetwork.h>
#include <Test/Stopwatch.h>
#include <io/Logger.h>
#include <io/DirectStorageDataSet.h>

using namespace OpenANN;

int main()
{
  Logger logger(Logger::CONSOLE);

  int D = 1000;
  int F = 10;
  int N = 10;
  Mt X(D, N);
  Mt Y(F, N);
  DirectStorageDataSet ds(X, Y);
  int numGetSet = 10000;
  int forwardProps = 1000000;

  {
    Stopwatch sw;

    sw.start();
    DeepNetwork net(DeepNetwork::SSE);
    net.inputLayer(D);
    net.outputLayer(F, LINEAR);
    net.trainingSet(ds);
    logger << "Construct Net: " << sw.stop(Stopwatch::MILLISECOND) << " ms\n";

    sw.start();
    net.gradient();
    logger << "Gradient Net: " << sw.stop(Stopwatch::MILLISECOND) << " ms\n";

    sw.start();
    net.gradientFD();
    logger << "GradientFD Net: " << sw.stop(Stopwatch::MILLISECOND) << " ms\n";

    sw.start();
    for(int i = 0; i < numGetSet; i++)
      net.setParameters(net.currentParameters());
    logger << "Get/Set Net: " << sw.stop(Stopwatch::MILLISECOND) << " ms\n";

    sw.start();
    for(int i = 0; i < forwardProps; i++)
      net(X.col(0));
    logger << "Forward Net: " << sw.stop(Stopwatch::MILLISECOND) << " ms\n";
  }

  logger << "=======\n";

  {
    Stopwatch sw;

    sw.start();
    MLP mlp;
    mlp.input(D);
    mlp.output(F, MLP::SSE, MLP::ID);
    mlp.trainingSet(ds);
    logger << "Construct MLP: " << sw.stop(Stopwatch::MILLISECOND) << " ms\n";

    sw.start();
    mlp.gradient();
    logger << "Gradient MLP: " << sw.stop(Stopwatch::MILLISECOND) << " ms\n";

    sw.start();
    mlp.gradientFD();
    logger << "GradientFD MLP: " << sw.stop(Stopwatch::MILLISECOND) << " ms\n";

    sw.start();
    for(int i = 0; i < numGetSet; i++)
      mlp.setParameters(mlp.currentParameters());
    logger << "Get/Set MLP: " << sw.stop(Stopwatch::MILLISECOND) << " ms\n";

    sw.start();
    for(int i = 0; i < forwardProps; i++)
      mlp(X.col(0));
    logger << "Forward MLP: " << sw.stop(Stopwatch::MILLISECOND) << " ms\n";
  }
}
