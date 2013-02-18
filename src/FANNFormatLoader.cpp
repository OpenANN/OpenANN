#include <io/FANNFormatLoader.h>
#include <AssertionMacros.h>

namespace OpenANN
{

FANNFormatLoader::FANNFormatLoader(const std::string& benchmark,
                                   int maxTrainingSamples, int maxTestSamples,
                                   bool loadNow)
  : progressLogger(Logger::CONSOLE), trainingN(0), testN(0), D(0), F(0),
    trainingData((benchmark + ".train").c_str()),
    testData((benchmark + ".test").c_str())
{
  prepareLoading(maxTrainingSamples, maxTestSamples);
  if(loadNow)
    load();
}

void FANNFormatLoader::prepareLoading(int maxTrainingSamples, int maxTestSamples)
{
  OPENANN_CHECK(trainingData.is_open());
  OPENANN_CHECK(testData.is_open());
  trainingData >> trainingN >> D >> F;
  if(maxTrainingSamples >= 0)
    trainingN = std::min(trainingN, maxTrainingSamples);
  int D2 = 0, F2 = 0;
  testData >> testN >> D2 >> F2;
  if(maxTestSamples >= 0)
    testN = std::min(testN, maxTestSamples);
  OPENANN_CHECK_EQUALS(D, D2);
  OPENANN_CHECK_EQUALS(F, F2);
}

void FANNFormatLoader::load()
{
  allocateMemory();
  load(true);
  load(false);
}

void FANNFormatLoader::allocateMemory()
{
  trainingInput.resize(D, trainingN);
  trainingOutput.resize(F, trainingN);
  testInput.resize(D, testN);
  testOutput.resize(F, testN);
}

void FANNFormatLoader::load(bool train)
{
  int& N = train ? trainingN : testN;
  Mt& input = train ? trainingInput : testInput;
  Mt& output = train ? trainingOutput : testOutput;
  std::fstream& stream = train ? trainingData : testData;
  for(int n = 0; n < N; n++)
  {
    int idx = 0;
    fpt del = 0.0;
    for(int d = 0; d < D; d++)
      stream >> input(idx++, n);
    for(int f = 0; f < F; f++)
      stream >> output(f, n);
  }
  progressLogger << "Loaded " << (train ? "training" : "test") << " set.\n";
}

}
