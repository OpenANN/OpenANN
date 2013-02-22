#pragma once

#include <AssertionMacros.h>
#include <Eigen/Dense>
#include <fstream>
#include <io/Logger.h>
#include <stdint.h>
#include <endian.h>
#include "Distorter.h"

class IDXLoader
{
public:
  int padToX;
  int padToY;
  std::string directory;
  int trainingN;
  int testN;
  int D;
  int F;
  Mt trainingInput, trainingOutput, testInput, testOutput;
  OpenANN::Logger debugLogger;

  IDXLoader(int padToX = 29, int padToY = 29, int loadTraininN = -1,
            int loadTestN = -1, std::string directory = "mnist/")
    : padToX(padToX), padToY(padToY), directory(directory), trainingN(0),
      testN(0), D(0), F(0), debugLogger(OpenANN::Logger::CONSOLE)
  {
    load(true, loadTraininN);
    load(false, loadTestN);
    debugLogger << "Loaded MNIST data set.\n"
        << "trainingN = " << trainingN << "\n"
        << "testN = " << testN << "\n"
        << "D = " << D << ", F = " << F << "\n";
  }

  void load(bool train, int maxN)
  {
    int& N = train ? trainingN : testN;
    Mt& input = train ? trainingInput : testInput;
    Mt& output = train ? trainingOutput : testOutput;
    unsigned char tmp = 0;

    std::string fileName = train ?
            directory + "/" + std::string("train-images-idx3-ubyte") :
            directory + "/" + std::string("t10k-images-idx3-ubyte");
    std::fstream inputFile(fileName.c_str(), std::ios::in | std::ios::binary);
    if(!inputFile.is_open())
    {
      debugLogger << "Could not find file \"" << fileName << "\".\n"
                  << "Please download the MNIST data set.\n";
      exit(1);
    }
    int8_t zero = 0, encoding = 0, dimension = 0;
    int32_t images = 0, rows = 0, cols = 0, items = 0;
    inputFile.read(reinterpret_cast<char*>(&zero), sizeof(zero));
    OPENANN_CHECK_EQUALS(0, (int) zero);
    inputFile.read(reinterpret_cast<char*>(&zero), sizeof(zero));
    OPENANN_CHECK_EQUALS(0, (int) zero);
    inputFile.read(reinterpret_cast<char*>(&encoding), sizeof(encoding));
    OPENANN_CHECK_EQUALS(8, (int) encoding);
    inputFile.read(reinterpret_cast<char*>(&dimension), sizeof(dimension));
    OPENANN_CHECK_EQUALS(3, (int) dimension);
    read(inputFile, images);
    read(inputFile, cols);
    read(inputFile, rows);
    D = (int) (rows * cols);
    N = (int) images;
    if(maxN > 0)
      N = maxN;
    if(D < padToX * padToY)
      D = padToX * padToY;
    int colNumber = padToX > (int)cols ? padToX : (int)cols;

    input.resize(D, N);
    for(int n = 0; n < N; n++)
    {
      int r = 0;
      for(; r < (int) rows; r++)
      {
        int c = 0;
        for(; c < (int) cols; c++)
        {
          read(inputFile, tmp);
          double value = (double) tmp;
          input(r*colNumber+c, n) = 1.0-value/255.0; // scale to [0:1]
        }
        int lastC = c-1;
        for(; c < padToX; c++)
        {
          input(r*colNumber+c, n) = input(r*colNumber+lastC, n);
        }
      }
      int lastR = r-1;
      for(; r < padToY; r++)
      {
        for(int c = 0; c < padToX; c++)
        {
          input(r*colNumber+c, n) = input(lastR*colNumber+c, n);
        }
      }
    }

    std::string labelFileName = train ?
            directory + "/" + std::string("train-labels-idx1-ubyte") :
            directory + "/" + std::string("t10k-labels-idx1-ubyte");
    std::fstream labelFile(labelFileName.c_str(), std::ios::in | std::ios::binary);
    if(!labelFile.is_open())
    {
      debugLogger << "Could not find file \"" << labelFileName << "\".\n"
                  << "Please download the MNIST data set.\n";
      exit(1);
    }
    labelFile.read(reinterpret_cast<char*>(&zero), sizeof(zero));
    OPENANN_CHECK_EQUALS(0, (int) zero);
    labelFile.read(reinterpret_cast<char*>(&zero), sizeof(zero));
    OPENANN_CHECK_EQUALS(0, (int) zero);
    labelFile.read(reinterpret_cast<char*>(&encoding), sizeof(encoding));
    OPENANN_CHECK_EQUALS(8, (int) encoding);
    labelFile.read(reinterpret_cast<char*>(&dimension), sizeof(dimension));
    OPENANN_CHECK_EQUALS(1, (int) dimension);
    read(labelFile, items);
    OPENANN_CHECK_EQUALS(images, items);
    F = 10;

    output.resize(F, N);
    for(int n = 0; n < N; n++)
    {
      read(labelFile, tmp);
      for(int c = 0; c < F; c++)
        output(c, n) = (255 - (int) tmp) == c ? 1.0 : 0.0;
    }
  }

  void distort(int multiplier = 10)
  {
    debugLogger << "Start creating " << (multiplier-1) << " distortions.\n";
    int originalN = trainingN;
    trainingN *= multiplier;
    Mt originalInput = trainingInput;
    trainingInput.resize(D, trainingN);
    trainingInput.block(0, 0, D, originalN) = originalInput;
    Mt originalOutput = trainingOutput;
    trainingOutput.resize(F, trainingN);
    trainingOutput.block(0, 0, F, originalN) = originalOutput;

    Vt instance(D);
    for(int distortion = 1; distortion < multiplier; distortion++)
    {
      Distorter distorter;
      distorter.createDistortionMap(padToX, padToY);
      for(int n = distortion*originalN; n < (distortion+1)*originalN; n++)
      {
        instance = originalInput.col(n % originalN);
        distorter.applyDistortion(instance);
        trainingInput.col(n) = instance;
      }
      trainingOutput.block(0, distortion*originalN, F, originalN) = originalOutput;
      debugLogger << "Finished distortion " << distortion << ".\n";
    }
  }

private:
  template<typename T>
  void read(std::fstream& stream, T& t)
  {
    stream.read(reinterpret_cast<char*>(&t), sizeof(t));
    if(sizeof(t) == 4)
      t = htobe32(t);
    else if(sizeof(t) == 1)
      t = 255 - t;
  }
};
