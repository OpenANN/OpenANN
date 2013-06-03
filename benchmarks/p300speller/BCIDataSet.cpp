#include "BCIDataSet.h"
#include "Decimator.h"
#include <OpenANN/util/AssertionMacros.h>
#include <fstream>

BCIDataSet::BCIDataCache::BCIDataCache(int size, int D)
  : size(size * 1024 * 1024), D(D)
{
}

bool BCIDataSet::BCIDataCache::hasInstance(int epoch, int offset)
{
  return cache.find(std::make_pair(epoch, offset)) != cache.end();
}

Eigen::VectorXd& BCIDataSet::BCIDataCache::getInstance(int epoch, int offset)
{
  return cache[std::make_pair(epoch, offset)];
}

bool BCIDataSet::BCIDataCache::hasSpace()
{
  return size >= (cache.size() + 1) * D * sizeof(double);
}

void BCIDataSet::BCIDataCache::cacheInstance(int epoch, int offset, const Eigen::VectorXd& instance)
{
  cache[std::make_pair(epoch, offset)] = instance;
}

void BCIDataSet::BCIDataCache::clear()
{
  cache.clear();
}

BCIDataSet::BCIDataSet(const std::string directory, const std::string& subject,
                       const std::string dataType, bool loadNow)
  : directory(directory), subject(subject),
    dataType(dataType == "test" ? TEST : (dataType == "demo" ? DEMO : TRAINING)),
    debugLogger(OpenANN::Logger::NONE),
    iteration(0),
    comp(false), decimated(false), downSamplingFactor(1),
    cache(1000, 0)
{
  determineDimension();
  if(loadNow)
    load();
}

void BCIDataSet::load()
{
  loadFlashing();
  loadStimulusCode();
  loadStimulusType();
  loadTargetChar();
  loadSignal();
  setupInterface();
}

void BCIDataSet::determineDimension()
{
  std::ifstream file(fileName("Flashing").c_str());
  OPENANN_CHECK(file.is_open());
  sampling = 240;
  channels = 64;
  epochs = dataType == TEST ? 100 : 85;
  if(dataType == DEMO)
    readEpochs = 1;
  else
    readEpochs = epochs;
  maxT = 7794;
  D = sampling * channels;
  F = 1;
  debugLogger << sampling << " samples, " << channels << " channels, "
              << epochs << " epochs, " << maxT << " steps\n";
  flashing.resize(maxT, epochs);
  stimulusCode.resize(maxT, epochs);
  stimulusType.resize(maxT, epochs);
  targetChar.resize(epochs);
  signal.resize(epochs, Eigen::MatrixXd(channels, maxT));
  tempInstance.resize(channels * sampling);
}

void BCIDataSet::loadFlashing()
{
  std::ifstream file(fileName("Flashing").c_str());
  OPENANN_CHECK(file.is_open());

  for(int t = 0; t < maxT; t++)
  {
    for(int e = 0; e < epochs; e++)
    {
      file >> flashing(t, e);
    }
  }
  debugLogger << "Loaded flashing.\n";
}

void BCIDataSet::loadStimulusCode()
{
  std::ifstream file(fileName("StimulusCode").c_str());
  OPENANN_CHECK(file.is_open());

  for(int t = 0; t < maxT; t++)
  {
    for(int e = 0; e < epochs; e++)
    {
      file >> stimulusCode(t, e);
    }
  }
  debugLogger << "Loaded stimulus code.\n";
}

void BCIDataSet::loadStimulusType()
{
  std::ifstream file(fileName("StimulusType").c_str());
  OPENANN_CHECK(file.is_open());

  for(int t = 0; t < maxT; t++)
  {
    for(int e = 0; e < epochs; e++)
    {
      file >> stimulusType(t, e);
    }
  }
  debugLogger << "Loaded stimulus type.\n";
}

void BCIDataSet::loadTargetChar()
{
  std::ifstream file(fileName("TargetChar").c_str());
  OPENANN_CHECK(file.is_open());

  int c = 0;
  for(int e = 0; e < epochs; e++)
  {
    file >> c;
    targetChar[e] = (char)c;
  }
  debugLogger << "Loaded target char.\n";
}

void BCIDataSet::loadSignal()
{
  std::ifstream file(fileName("Signal").c_str());
  OPENANN_CHECK(file.is_open());

  for(int e = 0; e < readEpochs; e++)
  {
    for(int c = 0; c < channels; c++)
    {
      for(int t = 0; t < maxT; t++)
      {
        file >> signal[e](c, t);
      }
    }
    if(debugLogger.isActive())
    {
      double progress = 100.0 * (double)(e + 1) / (double) readEpochs;
      if(e % 10 == 0 || e == readEpochs - 1)
      {
        debugLogger << "[";
        int p = 0;
        for(; p < (int)(progress + 0.5); p++)
          debugLogger << "#";
        for(; p < 100; p++)
          debugLogger << " ";
        debugLogger << "] (" << (int)(progress + 0.5) << "%)\n";
      }
    }
  }
  debugLogger << "Loaded signal.\n";
}

void BCIDataSet::setupInterface()
{
  N = 0;
  std::vector<int> lastFlashing(readEpochs, 0);
  instanceStart.resize(readEpochs);
  instanceLabel.resize(readEpochs);
  Eigen::VectorXd label(1);
  for(int t = 0; t < maxT; t++)
  {
    for(int e = 0; e < readEpochs; e++)
    {
      if(flashing(t, e) > 0.0 && lastFlashing[e] == 0)
      {
        N++;
        instanceStart[e].push_back(t);
        label(0, 0) = ((int) stimulusType(t, e)) * 2 - 1;
        instanceLabel[e].push_back(label);
      }
      lastFlashing[e] = (int) flashing(t, e);
    }
  }
  clear();
}

void BCIDataSet::clear()
{
  cache.clear();
  cache.D = D;
  tempInstance.resize(D);
}

std::string BCIDataSet::fileName(const std::string& type)
{
  return directory + "/Subject_" + subject + "_" +
         (dataType == TEST ? "Test_" : "Train_") + type + ".txt";
}

void BCIDataSet::decimate(int factor)
{
  downSamplingFactor = factor;
  D = sampling / downSamplingFactor * channels;
  decimated = true;
  clear();
}

void BCIDataSet::compress(const Eigen::MatrixXd& compressionMatrix)
{
  compressor.init(compressionMatrix);
  D = compressionMatrix.rows();
  comp = true;
  clear();
}

void BCIDataSet::reset()
{
  decimated = false;
  downSamplingFactor = 1;
  comp = false;
  compressor.reset();
  cache.clear();
  iteration = 0;
  D = sampling * channels;
}

Eigen::VectorXd& BCIDataSet::getInstance(int i)
{
  int epoch = 0, t0 = 0;
  getOffsets(i, epoch, t0);
  buildInstance(epoch, t0);
  return tempInstance;
}

void BCIDataSet::getOffsets(int i, int& epoch, int& t0)
{
  epoch = i / instanceLabel[0].size();
  int number = i % instanceLabel[0].size();
  t0 = instanceStart[epoch][number];
}

void BCIDataSet::buildInstance(int epoch, int t0)
{
  if(cache.hasInstance(epoch, t0))
  {
    tempInstance = cache.getInstance(epoch, t0);
  }
  else
  {
    Eigen::MatrixXd original = extractInstance(epoch, t0);
    if(decimated)
    {
      Decimator decimator(downSamplingFactor);
      Eigen::MatrixXd decimatedSignal = decimator.decimate(original);
      if(comp)
      {
        Eigen::VectorXd uncompressed = toVector(decimatedSignal);
        tempInstance = compressor.compress(uncompressed);
      }
      else
        tempInstance = toVector(decimatedSignal);
    }
    else
    {
      if(comp)
      {
        Eigen::VectorXd uncompressed = toVector(original);
        tempInstance = compressor.compress(uncompressed);
      }
      else
        tempInstance = toVector(original);
    }
    if(cache.hasSpace())
      cache.cacheInstance(epoch, t0, tempInstance);
  }
}

Eigen::MatrixXd BCIDataSet::extractInstance(int epoch, int t0)
{
  Eigen::MatrixXd instance(channels, sampling);
  for(int c = 0; c < channels; c++)
    for(int t = 0; t < sampling; t++)
      instance(c, t) = signal[epoch](c, t0 + t);
  return instance;
}

Eigen::VectorXd BCIDataSet::toVector(const Eigen::MatrixXd& matrix)
{
  Eigen::VectorXd vector(matrix.rows()*matrix.cols());
  for(int r = 0; r < matrix.rows(); r++)
    for(int c = 0; c < matrix.cols(); c++)
      vector(r * matrix.cols() + c) = matrix(r, c);
  return vector;
}

Eigen::VectorXd& BCIDataSet::getTarget(int i)
{
  int epoch = i / instanceLabel[0].size();
  int number = i % instanceLabel[0].size();
  return instanceLabel[epoch][number];
}

char BCIDataSet::getTargetChar(int i)
{
  int epoch = 0, t0 = 0;
  getOffsets(i, epoch, t0);
  return targetChar[epoch];
}

void BCIDataSet::finishIteration(OpenANN::Learner& learner)
{
  ++iteration;
}

int BCIDataSet::evaluate(OpenANN::Learner& learner, int trials)
{
  char characters[6][7] =
  {
    "ABCDEF",
    "GHIJKL",
    "MNOPQR",
    "STUVWX",
    "YZ1234",
    "56789_"
  };
  Eigen::VectorXd y(1);
  int correct = 0;
  std::vector<char> predictions(readEpochs, 0);

  for(int e = 0; e < readEpochs; e++)
  {
    double score[12] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    int repetitions[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    std::vector<int>& offsets = instanceStart[e];
    for(size_t i = 0; i < offsets.size(); i++)
    {
      int t0 = offsets[i];
      buildInstance(e, t0);
      y = learner(tempInstance);
      int rowColCode = (int) stimulusCode(t0, e) - 1;
      if(repetitions[rowColCode] < trials)
      {
        score[rowColCode] += y(0);
        repetitions[rowColCode]++;
      }
    }

    int maxRow = -1, maxCol = -1;
    double maxRowScore = -std::numeric_limits<double>::max(),
           maxColScore = -std::numeric_limits<double>::max();
    for(int i = 0; i < 6; i++)
    {
      if(score[i] > maxColScore)
      {
        maxCol = i;
        maxColScore = score[i];
      }
    }
    for(int i = 6; i < 12; i++)
    {
      if(score[i] > maxRowScore)
      {
        maxRow = i - 6;
        maxRowScore = score[i];
      }
    }

    char actual = targetChar[e];
    predictions[e] = characters[maxRow][maxCol];
    if(actual == predictions[e])
      correct++;
  }
  return correct;
}
