#include <OpenANN/OpenANN>
#include <OpenANN/io/DirectStorageDataSet.h>
#include <OpenANN/io/Logger.h>
#include "IDXLoader.h"
#include <QGLWidget>
#include <QKeyEvent>
#include <QApplication>
#include <GL/glu.h>
#ifdef PARALLEL_CORES
#include <omp.h>
#endif

class MNISTVisualization : public QGLWidget
{
  OpenANN::Net& net;
  OpenANN::DataSet& dataSet;
  int rows, cols;
  int width, height;
  int instance;
  OpenANN::Logger debugLogger;
public:
  MNISTVisualization(OpenANN::Net& net, OpenANN::DataSet& dataSet,
                     int rows, int cols,
                     QWidget* parent = 0, const QGLWidget* shareWidget = 0,
                     Qt::WindowFlags f = 0)
    : net(net), dataSet(dataSet), rows(rows), cols(cols), width(800),
      height(600), instance(0), debugLogger(OpenANN::Logger::CONSOLE)
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

    Eigen::VectorXd x = dataSet.getInstance(instance);
    Eigen::VectorXd y = net(x);
    Eigen::VectorXd yt = dataSet.getTarget(instance);

    int prediction = 0;
    y.maxCoeff(&prediction);
    int label = 0;
    yt.maxCoeff(&label);

    glColor3f(0.0f, 0.0f, 0.0f);
    glLineWidth(5.0);

    float scale = 0;
    float translateY = 0;

    for(int l = 0; l < net.numberOflayers(); l++)
    {
      float translateX = 0;
      OpenANN::Layer& layer = net.getLayer(l);
      Eigen::MatrixXd layerOutput = layer.getOutput();
      double mi = layerOutput.minCoeff();
      double ma = layerOutput.maxCoeff();
      OpenANN::OutputInfo info = net.getOutputInfo(l);
      float featureMaps = 1, cols = 1, rows = 1;
      if(info.dimensions.size() == 3)
      {
        featureMaps = info.dimensions[0];
        cols = info.dimensions[1];
        rows = info.dimensions[2];
      }
      else if(info.dimensions.size() == 2)
      {
        cols = info.dimensions[0];
        rows = info.dimensions[1];
      }
      else
      {
        cols = info.dimensions[0];
      }
      glColor3f(0.0f, 0.0f, 0.0f);
      renderText(10, 35 * (net.numberOflayers() - l + 1), QString::number(featureMaps)
                 + "x" + QString::number(rows) + "x" + QString::number(cols),
                 QFont("Helvetica", 20));
      glBegin(GL_QUADS);
      scale = 1.0;
      if(featureMaps > 1 || rows == 1)
        scale = 120.0f / (featureMaps * cols + featureMaps - 1);
      else
        translateX += 60.0f - cols / 2.0f;
      for(int fm = 0; fm < featureMaps; fm++, translateX += (cols + 1) * scale)
      {
        for(int row = 0; row < rows; row++)
        {
          for(int col = 0; col < cols; col++)
          {
            float c = (layerOutput(0, fm * rows * cols + row * cols + col) - mi) / (ma - mi);
            float x = translateX + col * scale;
            float y = translateY + (rows - row) * scale;
            glColor3f(c, c, c);
            glVertex2f((float) x, (float) y);
            glVertex2f((float) x + scale, (float) y);
            glVertex2f((float) x + scale, (float) y + scale);
            glVertex2f((float) x, (float) y + scale);
          }
        }
      }
      glEnd();
      translateY += (rows + 1) * scale;
    }

    glColor3f(0.0f, 0.0f, 0.0f);
    renderText(20, 30, QString("Instance #") + QString::number(instance + 1)
               + QString(", label: ") + QString::number(label) + ", prediction: "
               + QString::number(prediction), QFont("Helvetica", 20));

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
  OpenANN::Logger interfaceLogger(OpenANN::Logger::CONSOLE);

  std::string directory = "mnist/";
  if(argc > 1)
    directory = std::string(argv[1]);

  IDXLoader loader(28, 28, 60000, 10000, directory);

  OpenANN::Net net;
  net.inputLayer(1, loader.padToX, loader.padToY)
  .convolutionalLayer(20, 5, 5, OpenANN::RECTIFIER, 0.05)
  .maxPoolingLayer(2, 2)
  .convolutionalLayer(20, 5, 5, OpenANN::RECTIFIER, 0.05)
  .maxPoolingLayer(2, 2)
  .fullyConnectedLayer(150, OpenANN::RECTIFIER, 0.05)
  .fullyConnectedLayer(100, OpenANN::RECTIFIER, 0.05)
  .outputLayer(loader.F, OpenANN::LINEAR, 0.05)
  .trainingSet(loader.trainingInput, loader.trainingOutput);
  OpenANN::DirectStorageDataSet testSet(&loader.testInput, &loader.testOutput);
  net.initialize();

  // Load parameters
  std::ifstream file("weights.log");
  Eigen::VectorXd weights = net.currentParameters();
  for(int i = 0; i < net.dimension(); i++)
    file >> weights(i);
  net.setParameters(weights);

  net.validationSet(testSet);
  net.setErrorFunction(OpenANN::CE);
  interfaceLogger << "Created MLP.\n" << "D = " << loader.D << ", F = "
                  << loader.F << ", N = " << loader.trainingN << ", L = " << net.dimension() << "\n";

  QApplication app(argc, argv);
  MNISTVisualization visual(net, testSet, loader.padToX, loader.padToY);
  visual.show();
  visual.resize(800, 600);

  return app.exec();
}
