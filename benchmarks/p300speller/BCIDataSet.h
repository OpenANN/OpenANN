#ifndef BCIDATASET_H_
#define BCIDATASET_H_

#include <OpenANN/Compressor.h>
#include <OpenANN/io/DataSet.h>
#include <OpenANN/io/Logger.h>
#include <OpenANN/Learner.h>
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
    std::map<std::pair<int, int>, Eigen::VectorXd> cache;

    /**
     * Initialize the data cache.
     * @param size Size in Megabyte.
     * @param D Data dimension.
     */
    BCIDataCache(int size, int D);
    bool hasInstance(int epoch, int offset);
    Eigen::VectorXd& getInstance(int epoch, int offset);
    bool hasSpace();
    void cacheInstance(int epoch, int offset, const Eigen::VectorXd& instance);
    void clear();
  };

  std::string directory;
  std::string subject;
  enum DataType
  {
    TRAINING, TEST, DEMO
  } dataType;

  Eigen::MatrixXd flashing;
  Eigen::MatrixXd stimulusCode;
  Eigen::MatrixXd stimulusType;
  std::vector<char> targetChar;
  std::vector<Eigen::MatrixXd> signal;

  int sampling;
  int channels;
  int epochs;
  int readEpochs;
  int maxT;
  int N;
  int D;
  int F;
  std::vector<std::vector<int> > instanceStart;
  std::vector<std::vector<Eigen::VectorXd> > instanceLabel;

  OpenANN::Logger debugLogger;

  Eigen::VectorXd tempInstance;

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
  void compress(const Eigen::MatrixXd& compressionMatrix);
  void reset();
  virtual int samples() { return N; }
  virtual int inputs() { return D; }
  virtual int outputs() { return F; }
  virtual Eigen::VectorXd& getInstance(int i);
  void getOffsets(int i, int& epoch, int& t0);
  void buildInstance(int epoch, int t0);
  Eigen::MatrixXd extractInstance(int epoch, int t0);
  Eigen::VectorXd toVector(const Eigen::MatrixXd& matrix);
  virtual Eigen::VectorXd& getTarget(int i);
  char getTargetChar(int i);
  virtual void finishIteration(OpenANN::Learner& mlp);
  int evaluate(OpenANN::Learner& mlp, int trials);
};

#endif // BCIDATASET_H_
