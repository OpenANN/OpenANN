#include <QApplication>
#include <OpenANN>
#include <io/DirectStorageDataSet.h>
#include <Random.h>
#include "Plot.h"

int main(int argc, char** argv)
{
  OpenANN::OpenANNLibraryInfo::print();

  const int N = 50;
  Mt in(1, N);
  Mt out(1, N);
  OpenANN::RandomNumberGenerator rng;
  for(int n = 0; n < N; n++)
  {
    in(0, n) = 2.0f * (fpt) n / (fpt) N - 1.0 + 1.0 / (fpt) N;
    out(0, n) = in(0, n) + rng.sampleNormalDistribution<fpt>()*0.1;
  }
  OpenANN::DirectStorageDataSet dataSet(in, out);
  QApplication app(argc, argv);
  Plot plot(dataSet);
  plot.show();
  plot.resize(400, 400);

  return app.exec();
}
