#ifndef CIFAR_LOADER_H_
#define CIFAR_LOADER_H_

#include <OpenANN/util/OpenANNException.h>
#include <fstream>
#include <string>
#include <vector>

/**
 * @class CIFARLoader
 *
 * Loads the CIFAR-10 dataset.
 *
 * The dataset is provided via the members trainingInput, trainingOutput,
 * testInput and testOutput.
 */
class CIFARLoader
{
  std::string directory;
  std::vector<std::string> trainFiles, testFiles;
public:
  Eigen::MatrixXd trainingInput, trainingOutput, testInput, testOutput;
  int C, X, Y, D, F, trainingN, testN, NperFile;

  CIFARLoader(const std::string& directory)
    : directory(directory)
  {
    setup();
    load(trainFiles, trainingInput, trainingOutput);
    load(testFiles, testInput, testOutput);
  }

private:
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
    trainingInput.resize(trainingN, D);
    trainingOutput.resize(trainingN, F);
    testInput.resize(testN, D);
    testOutput.resize(testN, F);
  }

  void load(std::vector<std::string>& file_names, Eigen::MatrixXd& inputs, Eigen::MatrixXd& outputs)
  {
    int instance = 0;
    char values[D + 1];
    for(int f = 0; f < file_names.size(); f++)
    {
      std::fstream file((directory + "/" + file_names[f]).c_str(),
                        std::ios::in | std::ios::binary);
      if(!file.is_open())
        throw OpenANN::OpenANNException("Could not open file '"
                                        + file_names[f] + "' in directory '" 
                                        + directory + "'.");
      for(int n = 0; n < NperFile; n++, instance++)
      {
        if(file.eof())
          throw OpenANN::OpenANNException("Reached unexpected end of file "
                                          + file_names[f] + ".");

        file.read(values, D + 1);
        if(values[0] < 0 || values[0] >= F)
          throw OpenANN::OpenANNException("Unknown class detected.");
        outputs.row(instance).setZero();
        outputs.row(instance)(*reinterpret_cast<unsigned char*>(&values[0])) = 1.0;

        int idx = 0;
        for(int c = 0; c < C; c++)
        {
          for(int x = 0; x < X; x++)
          {
            for(int y = 0; y < Y; y++, idx++)
            {
              // Scale data to [-1, 1]
              inputs(instance, idx) = ((double) * reinterpret_cast<unsigned char*>(&values[idx + 1])) / 128.0 - 1.0;
            }
          }
        }
      }
    }
  }
};

#endif // CIFAR_LOADER_H_
