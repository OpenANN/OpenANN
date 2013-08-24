#include <OpenANN/OpenANN>
#include <OpenANN/RBM.h>
#include <OpenANN/io/Logger.h>
#include <OpenANN/io/DirectStorageDataSet.h>
#include <OpenANN/Evaluator.h>
#include <OpenANN/util/OpenANNException.h>
#include <OpenANN/optimization/MBSGD.h>
#include <OpenANN/optimization/StoppingCriteria.h>
#include "IDXLoader.h"
#include <QGLWidget>
#include <QKeyEvent>
#include <QApplication>
#include <GL/glu.h>
#ifdef PARALLEL_CORES
#include <omp.h>
#endif

/**
 * \page MNISTRBM Restricted Boltzmann Machine on MNIST dataset
 *
 * An RBM with 50 hidden nodes is trained on the MNIST dataset and then used
 * to generate features for another neural network that will be trained
 * supervised.
 */

class RBMVisualization : public QGLWidget
{
  OpenANN::RBM& rbm;
  OpenANN::DataSet& dataSet;
  int neuronRows, neuronCols;
  int width, height;
  int instance;
  int offset;
  int fantasy;
  bool showFilters;
public:
  RBMVisualization(OpenANN::RBM& rbm, OpenANN::DataSet& dataSet,
                   int neuronRows, int neuronCols, int width, int height,
                   QWidget* parent = 0, const QGLWidget* shareWidget = 0,
                   Qt::WindowFlags f = 0)
    : rbm(rbm), dataSet(dataSet), neuronRows(neuronRows),
      neuronCols(neuronCols), width(width), height(height), instance(0),
      offset(0), fantasy(0), showFilters(false)
  {
  }

  virtual void initializeGL()
  {
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glShadeModel(GL_SMOOTH);
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glPointSize(5.0);
  }

  virtual void resizeGL(int width, int height)
  {
    this->width = width;
    this->height = height;
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f, (float) width / (float) height, 1.0f, 200.0f);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glClearDepth(1.0f);
  }

  virtual void paintGL()
  {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    int xOffset = -70;
    int yOffset = -50;
    int zoom = -200;
    glTranslatef(xOffset, yOffset, zoom);

    glColor3f(0.0f, 0.0f, 0.0f);
    glLineWidth(5.0);

    float scale = 1.0;

    if(showFilters)
    {
      double mi = rbm.getWeights().minCoeff();
      double ma = rbm.getWeights().maxCoeff();
      double range = ma - mi;
      for(int row = 0, filter = 0; row < neuronRows; row++)
      {
        for(int col = 0; col < neuronCols; col++, filter++)
        {
          glBegin(GL_QUADS);
          for(int yIdx = 0; yIdx < 28; yIdx++)
          {
            for(int xIdx = 0; xIdx < 28; xIdx++)
            {
              int h = filter + offset;
              if(h >= rbm.getWeights().rows())
                throw OpenANN::OpenANNException("Illegal index for hidden unit");
              int idx = yIdx * 28 + xIdx;
              if(idx >= rbm.getWeights().cols())
                throw OpenANN::OpenANNException("Illegal index for pixel");
              float c = (rbm.getWeights()(h, idx) - mi) / range;
              float x = xIdx * scale + col * 29.0f * scale - 30.0f;
              float y = (28.0f - yIdx) * scale - row * scale * 29.0f + 90.0f;
              glColor3f(c, c, c);
              glVertex2f(x, y);
              glVertex2f(x + scale, y);
              glVertex2f(x + scale, y + scale);
              glVertex2f(x, y + scale);
            }
          }
          glEnd();
        }
      }
    }
    else
    {
      rbm.reconstructProb(instance, offset);
      for(int row = 0; row < neuronRows; row++)
      {
        for(int col = 0; col < neuronCols; col++)
        {
          glBegin(GL_QUADS);
          for(int yIdx = 0; yIdx < 28; yIdx++)
          {
            for(int xIdx = 0; xIdx < 28; xIdx++)
            {
              int idx = yIdx * 28 + xIdx;
              float c = rbm.getVisibleProbs()(0, idx);
              float x = xIdx * scale + col * 29.0f * scale - 30.0f;
              float y = (28.0f - yIdx) * scale - row * scale * 29.0f + 90.0f;
              glColor3f(c, c, c);
              glVertex2f(x, y);
              glVertex2f(x + scale, y);
              glVertex2f(x + scale, y + scale);
              glVertex2f(x, y + scale);
            }
          }
          glEnd();
          rbm.sampleHgivenV();
          rbm.sampleVgivenH();
        }
      }
    }

    glColor3f(0.0f, 0.0f, 0.0f);
    renderText(10, 35, "MNIST data set", QFont("Helvetica", 20));

    glFlush();
  }

  virtual void keyPressEvent(QKeyEvent* keyEvent)
  {
    switch(keyEvent->key())
    {
    case Qt::Key_Up:
      instance++;
      if(instance >= dataSet.samples())
        instance = dataSet.samples() - 1;
      update();
      break;
    case Qt::Key_Down:
      instance--;
      if(instance < 0)
        instance = 0;
      update();
      break;
    case Qt::Key_Left:
      offset--;
      if(offset < 0)
        offset = 0;
      update();
      break;
    case Qt::Key_Right:
    {
      offset++;
      int tooHi = offset + neuronRows * neuronCols - rbm.hiddenUnits();
      if(tooHi > 0)
        offset -= tooHi;
      update();
      break;
    }
    case Qt::Key_Minus:
      fantasy--;
      if(fantasy < 0)
        fantasy = 0;
      update();
      break;
    case Qt::Key_Plus:
      fantasy++;
      update();
      break;
    case Qt::Key_S:
      showFilters = !showFilters;
      update();
      break;
    default:
      QGLWidget::keyPressEvent(keyEvent);
      break;
    }
  }
};

int main(int argc, char** argv)
{
#ifdef PARALLEL_CORES
  omp_set_num_threads(PARALLEL_CORES);
#endif

  std::string directory = "./";
  if(argc > 1)
    directory = std::string(argv[1]);

  IDXLoader loader(28, 28, 60000, 10000, directory);
  OpenANN::DirectStorageDataSet trainSet(&loader.trainingInput,
                                         &loader.trainingOutput);

  OpenANN::Net net;
  net.inputLayer(1, loader.padToX, loader.padToY)
  .setRegularization(0.01, 0.0)
  .restrictedBoltzmannMachineLayer(300, 1, 0.1, false)
  .outputLayer(loader.F, OpenANN::SOFTMAX)
  .setErrorFunction(OpenANN::CE)
  .trainingSet(trainSet);

  OpenANN::RBM& rbm = (OpenANN::RBM&) net.getLayer(1);
  rbm.trainingSet(trainSet);

  OpenANN::MBSGD optimizer(0.01, 0.5, 16);
  OpenANN::StoppingCriteria stop;
  stop.maximalIterations = 5;
  optimizer.setOptimizable(rbm);
  optimizer.setStopCriteria(stop);
  optimizer.optimize();

  OpenANN::MulticlassEvaluator evaluator(1, OpenANN::Logger::FILE);
  OpenANN::DirectStorageDataSet testSet(&loader.testInput, &loader.testOutput,
                                        &evaluator);
  net.validationSet(testSet);

  OpenANN::StoppingCriteria stopNet;
  stopNet.maximalIterations = 10;
  OpenANN::MBSGD netOptimizer(0.01, 0.5, 16, false, 1.0, 0.0, 0.0, 1.0, 0.01,
                              100.0);
  netOptimizer.setOptimizable(net);
  netOptimizer.setStopCriteria(stopNet);
  netOptimizer.optimize();

  QApplication app(argc, argv);
  RBMVisualization visual(rbm, trainSet, 5, 7, 800, 600);
  visual.show();
  visual.resize(800, 600);
  return app.exec();
}
