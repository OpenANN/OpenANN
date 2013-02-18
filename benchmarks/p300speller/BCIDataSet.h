#ifndef BCIDATASET_H
#define BCIDATASET_H

#include <Compressor.h>
#include <io/DataSet.h>
#include <io/Logger.h>
#include <Learner.h>
#include <string>
#include <vector>
#include <map>

class BCIDataSet : public OpenANN::DataSet
{
public:
  class BCIDataCache
  {
  public:
    //! Size in byte.
    size_t size;
    int D;
    std::map<std::pair<int, int>, Vt> cache;

    /**
     * Initialize the data cache.
     * @param size Size in Megabyte.
     * @param D Data dimension.
     */
    BCIDataCache(int size, int D);
    bool hasInstance(int epoch, int offset);
    Vt& getInstance(int epoch, int offset);
    bool hasSpace();
    void cacheInstance(int epoch, int offset, const Vt& instance);
    void clear();
  };

  std::string directory;
  std::string subject;
  enum DataType {
    TRAINING, TEST, DEMO
  } dataType;

  Mt flashing;
  Mt stimulusCode;
  Mt stimulusType;
  std::vector<char> targetChar;
  std::vector<Mt> signal;

  int sampling;
  int channels;
  int epochs;
  int readEpochs;
  int maxT;
  int N;
  int D;
  int F;
  std::vector<std::vector<int> > instanceStart;
  std::vector<std::vector<Vt> > instanceLabel;

  OpenANN::Logger debugLogger;

  Vt tempInstance;

  int iteration;

  bool comp;
  OpenANN::Compressor compressor;
  bool decimated;
  int downSamplingFactor;

  BCIDataCache cache;

  BCIDataSet(const std::string directory, const std::string& subject,
      const std::string dataType, bool loadNow = true);
  virtual ~BCIDataSet() {}
  void load();
  void determineDimension();
  void loadFlashing();
  void loadStimulusCode();
  void loadStimulusType();
  void loadTargetChar();
  void loadSignal();
  void setupInterface();
  void clear();
  std::string fileName(const std::string& type);
  void decimate(int factor = 1);
  void compress(const Mt& compressionMatrix);
  void reset();
  virtual int samples() { return N; }
  virtual int inputs() { return D; }
  virtual int outputs() { return F; }
  virtual Vt& getInstance(int i);
  void getOffsets(int i, int& epoch, int& t0);
  void buildInstance(int epoch, int t0);
  Mt extractInstance(int epoch, int t0);
  Vt toVector(const Mt& matrix);
  virtual Vt& getTarget(int i);
  char getTargetChar(int i);
  virtual void finishIteration(OpenANN::Learner& mlp);
  int evaluate(OpenANN::Learner& mlp, int trials);
};

#endif // BCIDATASET_H
