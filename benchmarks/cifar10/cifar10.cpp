#include <OpenANN/OpenANN>
#include <OpenANN/optimization/MBSGD.h>
#include <OpenANN/io/DirectStorageDataSet.h>
#include <OpenANN/util/OpenANNException.h>
#ifdef PARALLEL_CORES
#include <omp.h>
#endif

class CIFARLoader
{
  OpenANN::Logger debugLogger;
  std::string directory;
public:
  std::vector<std::string> trainFiles, testFiles;
  Eigen::MatrixXd trainingInput, trainingOutput, testInput, testOutput;
  int C, X, Y, D, F, trainingN, testN, NperFile;

  CIFARLoader(const std::string& directory)
      : debugLogger(OpenANN::Logger::NONE), directory(directory)
  {
    setup();
    load(trainFiles, trainingInput, trainingOutput);
    load(testFiles, testInput, testOutput);
  }

  void setup()
  {
    C = 3;         // 3 color channels
    X = 32;        // 32 rows
    Y = 32;        // 32 cols
    D = C * X * Y; // 3072 inputs
    F = 10;        // 10 classes
    NperFile = 10000;
    trainFiles.push_back("data_batch_1.bin");
    trainFiles.push_back("data_batch_2.bin");
    trainFiles.push_back("data_batch_3.bin");
    trainFiles.push_back("data_batch_4.bin");
    trainFiles.push_back("data_batch_5.bin");
    testFiles.push_back("test_batch.bin");
    trainingN = trainFiles.size() * NperFile;
    testN = testFiles.size() * NperFile;
    trainingInput.resize(D, trainingN);
    trainingOutput.resize(F, trainingN);
    testInput.resize(D, testN);
    testOutput.resize(F, testN);
  }

  void load(std::vector<std::string>& file_names, Eigen::MatrixXd& inputs, Eigen::MatrixXd& outputs)
  {
    int instance = 0;
    char values[D+1];
    for(int f = 0; f < file_names.size(); f++)
    {
      std::fstream file((directory + "/" + file_names[f]).c_str(),
                        std::ios::in | std::ios::binary);
      if(!file.is_open())
        throw OpenANN::OpenANNException("Could not open file '"
            + file_names[f] + "' in directory '" + directory + "'.");
      for(int n = 0; n < NperFile; n++, instance++)
      {
        if(file.eof())
          throw OpenANN::OpenANNException("Reached unexpected end of file "
              + file_names[f] + ".");

        file.read(values, D+1);
        debugLogger << "Instance #" << instance << ", class: " << (int) values[0] << "\n\n";
        if(values[0] < 0 || values[0] >= F)
          throw OpenANN::OpenANNException("Unknown class detected.");
        outputs.col(instance).fill(0.0);
        outputs.col(instance)(*reinterpret_cast<unsigned char*>(&values[0])) = 1.0;

        int idx = 0;
        for(int c = 0; c < C; c++)
        {
          for(int x = 0; x < X; x++)
          {
            for(int y = 0; y < Y; y++, idx++)
            {
              // Scale data to [-1, 1]
              inputs(idx, instance) = ((double) *reinterpret_cast<unsigned char*>(&values[idx+1])) / 128.0 - 1.0;
              debugLogger << inputs(idx, instance) << " ";
            }
            debugLogger << "\n";
          }
          debugLogger << "\n";
        }
        debugLogger << "\n";
      }
    }
  }
};

/**
 * \page CIFAR10Benchmark CIFAR-10
 *
 * <a href="http://www.cs.toronto.edu/~kriz/cifar.html" target=_blank>
 * The CIFAR-10 dataset</a>
 */

int main(int argc, char** argv)
{
#ifdef PARALLEL_CORES
  omp_set_num_threads(PARALLEL_CORES);
#endif
  OpenANN::Logger interfaceLogger(OpenANN::Logger::CONSOLE);

  bool bigNet = false;
  std::string directory = ".";
  if(argc > 1)
    directory = std::string(argv[1]);
  if(argc > 2)
    bigNet = true;

  CIFARLoader loader(directory);

  OpenANN::Net net;                                                      // Nodes per layer:
  net.inputLayer(loader.C, loader.X, loader.Y, true);                    //   3 x 32 x 32
  if(bigNet)
  {
     net.convolutionalLayer(200, 5, 5, OpenANN::RECTIFIER, 0.05)         // 200 x 28 x 28
        .maxPoolingLayer(2, 2)                                           // 200 x 14 x 14
        .convolutionalLayer(150, 3, 3, OpenANN::RECTIFIER, 0.05)         // 150 x 12 x 12
        .maxPoolingLayer(2, 2)                                           // 150 x  6 x  6
        .convolutionalLayer(100, 3, 3, OpenANN::RECTIFIER, 0.05)         // 100 x  4 x  4
        .maxPoolingLayer(2, 2)                                           // 100 x  2 x  2
        .fullyConnectedLayer(300, OpenANN::RECTIFIER, 0.05, true, 15.0)  // 300
        .fullyConnectedLayer(100, OpenANN::RECTIFIER, 0.05, true, 15.0); // 100
  }
  else
  {
     net.convolutionalLayer(50, 5, 5, OpenANN::RECTIFIER, 0.05)          //  50 x 28 x 28
        .maxPoolingLayer(2, 2)                                           //  50 x 14 x 14
        .convolutionalLayer(30, 3, 3, OpenANN::RECTIFIER, 0.05)          //  30 x 12 x 12
        .maxPoolingLayer(2, 2)                                           //  30 x  6 x  6
        .convolutionalLayer(20, 3, 3, OpenANN::RECTIFIER, 0.05)          //  20 x  4 x  4
        .maxPoolingLayer(2, 2)                                           //  20 x  2 x  2
        .fullyConnectedLayer(100, OpenANN::RECTIFIER, 0.05, true, 15.0)  // 100
        .fullyConnectedLayer(50, OpenANN::RECTIFIER, 0.05, true, 15.0);  //  50
  }
  net.outputLayer(loader.F, OpenANN::LINEAR, 0.05)                            //  10
     .trainingSet(loader.trainingInput, loader.trainingOutput);
  OpenANN::DirectStorageDataSet testSet(loader.testInput, loader.testOutput,
                                        OpenANN::DirectStorageDataSet::MULTICLASS,
                                        OpenANN::Logger::FILE);
  net.testSet(testSet);
  net.setErrorFunction(OpenANN::CE);
  interfaceLogger << "Created MLP.\n" << "D = " << loader.D << ", F = "
      << loader.F << ", N = " << loader.trainingN << ", L = " << net.dimension() << "\n";

  OpenANN::StoppingCriteria stop;
  stop.maximalIterations = 100;
  OpenANN::MBSGD optimizer(0.01, 0.6, 10, 0.0, 1.0, 0.0, 0.0, 1.0, 0.01, 100.0);
  net.initialize();
  optimizer.setOptimizable(net);
  optimizer.setStopCriteria(stop);
  while(optimizer.step());

  interfaceLogger << "Error = " << net.error() << "\n\n";
  interfaceLogger << "Wrote data to dataset-*.log.\n";

  return EXIT_SUCCESS;
}
