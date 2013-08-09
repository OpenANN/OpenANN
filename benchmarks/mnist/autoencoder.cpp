#include <OpenANN/OpenANN>
#include <OpenANN/SparseAutoEncoder.h>
#include <OpenANN/io/Logger.h>
#include <OpenANN/io/DirectStorageDataSet.h>
#include <OpenANN/Evaluator.h>
#include <OpenANN/util/OpenANNException.h>
#include <OpenANN/optimization/LBFGS.h>
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
 * \page MNISTSAE Sparse auto-encoder on MNIST dataset
 */

class SparseAutoEncoderVisualization : public QGLWidget
{
  OpenANN::SparseAutoEncoder& sae;
  OpenANN::DataSet& dataSet;
  int H;
  int neuronRows, neuronCols;
  int width, height;
  int instance;
  int offset;
  bool showFilters;
public:
  SparseAutoEncoderVisualization(OpenANN::SparseAutoEncoder& sae,
                                 OpenANN::DataSet& dataSet, int H,
                                 int neuronRows, int neuronCols, int width,
                                 int height, QWidget* parent = 0,
                                 const QGLWidget* shareWidget = 0,
                                 Qt::WindowFlags f = 0)
    : sae(sae), dataSet(dataSet), H(H), neuronRows(neuronRows),
      neuronCols(neuronCols), width(width), height(height), instance(0),
      offset(0), showFilters(false)
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
      Eigen::MatrixXd W1 = sae.getInputWeights();
      Eigen::MatrixXd W2 = sae.getOutputWeights();
      double mi = sae.getInputWeights().minCoeff();
      double ma = sae.getInputWeights().maxCoeff();
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
              int h = (filter + offset) / 2;
              bool in = (filter + offset) % 2 == 0;
              int idx = yIdx * 28 + xIdx;
              float c = ((in ? W1(h, idx) : W2(idx, h)) - mi) / range;
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
      for(int row = 0; row < neuronRows; row++)
      {
        for(int col = 0; col < neuronCols; col++)
        {
          Eigen::VectorXd out = sae.reconstruct(dataSet.getInstance(offset + row*neuronCols+col));
          glBegin(GL_QUADS);
          for(int yIdx = 0; yIdx < 28; yIdx++)
          {
            for(int xIdx = 0; xIdx < 28; xIdx++)
            {
              int idx = yIdx * 28 + xIdx;
              float c = out(idx);
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
      int tooHi = offset + neuronRows * neuronCols - H/2;
      if(tooHi > 0)
        offset -= tooHi;
      update();
      break;
    }
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

  IDXLoader loader(28, 28, 10000, 1, directory);
  OpenANN::DirectStorageDataSet trainSet(&loader.trainingInput,
                                         &loader.trainingInput);

  int H = 196;
  OpenANN::SparseAutoEncoder sae(loader.D, H, 3.0, 0.1, 3e-3, OpenANN::LOGISTIC);
  sae.trainingSet(trainSet);

  OpenANN::LBFGS optimizer(20);
  OpenANN::StoppingCriteria stop;
  stop.maximalIterations = 400;
  optimizer.setOptimizable(sae);
  optimizer.setStopCriteria(stop);
  optimizer.optimize();

  OpenANN::MulticlassEvaluator evaluator(1, OpenANN::Logger::FILE);
  OpenANN::DirectStorageDataSet testSet(&loader.testInput, &loader.testInput,
                                        &evaluator);
  sae.validationSet(testSet);

  QApplication app(argc, argv);
  SparseAutoEncoderVisualization visual(sae, trainSet, H, 5, 7, 800, 600);
  visual.show();
  visual.resize(800, 600);
  return app.exec();
}
