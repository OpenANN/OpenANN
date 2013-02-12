#include "TwoSpiralsVisualization.h"
#include <AssertionMacros.h>
#include <GL/glu.h>
#include <QApplication>

TwoSpiralsVisualization::TwoSpiralsVisualization(
    const Mt& trainingInput,
    const Mt& trainingOutput,
    const Mt& testInput,
    const Mt& testOutput)
    : width(500), height(500),
      trainingSet(trainingInput, trainingOutput), testSet(testInput, testOutput),
      showTraining(true), showTest(true), showPrediction(true), showSmooth(true),
      net(new DeepNetwork), eventLogger(Logger::CONSOLE)
{
  std::memset(classes, 0, sizeof(fpt)*100*100);
  trainingSet.setVisualization(this);
  QObject::connect(this, SIGNAL(updatedData()), this, SLOT(repaint()));

  // initialize MLP
  net->inputLayer(trainingInput.rows())
    .fullyConnectedLayer(20, TANH)
    .fullyConnectedLayer(20, TANH)
    .outputLayer(trainingOutput.rows(), TANH)
    .trainingSet(trainingSet);

  // set stop criteria
  stop.maximalIterations = 10000;
  stop.minimalSearchSpaceStep = 1e-16;
  stop.minimalValueDifferences = 1e-16;
}

TwoSpiralsVisualization::~TwoSpiralsVisualization()
{
  delete net;
}

void TwoSpiralsVisualization::predictClass(int x, int y, fpt predictedClass)
{
  classesMutex.lock();
  classes[x][y] = predictedClass;
  classesMutex.unlock();
  if(x == 99 && y == 99)
    emit updatedData();
}

void TwoSpiralsVisualization::initializeGL()
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

void TwoSpiralsVisualization::resizeGL(int width, int height)
{
  this->width = width;
  this->height = height;
  glViewport(0, 0, width, height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(45.0f, (float) width / (float) height, 1.0f, 100.0f);
  glMatrixMode(GL_MODELVIEW);
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glClearDepth(1.0f);
}

void TwoSpiralsVisualization::paintGL()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glLoadIdentity();
  glTranslatef(-0.5f,-0.5f,-1.25f);

  if(showPrediction)
    paintPrediction();

  if(showTraining)
    paintDataSet(true);

  if(showTest)
    paintDataSet(false);

  glFlush();
}

void TwoSpiralsVisualization::paintPrediction()
{
  for(int x = 0; x < 100; x++)
  {
    for(int y = 0; y < 100; y++)
    {
      classesMutex.lock();
      float c;
      Vt v(2);
      v(0) = (fpt) x / 100.0f;
      v(1) = (fpt) y / 100.0f;
      if(showSmooth)
        c = classes[x][y]/2.0f + 0.5f;
      else
        c = classes[x][y] < 0.0 ? 0.0f : 1.0f;
      classesMutex.unlock();
      glColor3f(c, c, c);
      float minX = (float) x / 100.0f - 0.005;
      float maxX = minX + 0.01;
      float minY = (float) y / 100.0f - 0.005;
      float maxY = minY + 0.01;
      glBegin(GL_QUADS);
        glVertex2f(minX, maxY);
        glVertex2f(maxX, maxY);
        glVertex2f(maxX, minY);
        glVertex2f(minX, minY);
      glEnd();
    }
  }
}

void TwoSpiralsVisualization::paintDataSet(bool training)
{
  TwoSpiralsDataSet& dataSet = training ? trainingSet : testSet;
  glBegin(GL_POINTS);
    for(int n = 0; n < dataSet.samples(); n++)
    {
      if(dataSet.getTarget(n)(0) > 0.0)
        glColor3f(1.0f, 0.0f, 0.0f);
      else
        glColor3f(1.0f, 1.0f, 0.0f);
      glVertex2d(dataSet.getInstance(n)(0), dataSet.getInstance(n)(1));
    }
  glEnd();
}

void TwoSpiralsVisualization::keyPressEvent(QKeyEvent* keyEvent)
{
  switch(keyEvent->key())
  {
    case Qt::Key_Q:
      eventLogger << "Switching training set on/off.\n";
      showTraining = !showTraining;
      update();
      break;
    case Qt::Key_W:
      eventLogger << "Switching test set on/off.\n";
      showTest = !showTest;
      update();
      break;
    case Qt::Key_E:
      eventLogger << "Switching prediction on/off.\n";
      showPrediction = !showPrediction;
      update();
      break;
    case Qt::Key_R:
      eventLogger << "Switching smooth classes on/off.\n";
      showSmooth = !showSmooth;
      update();
      break;
    case Qt::Key_A:
      eventLogger << "Training with restart (" << net->dimension() << " parameters)...";
      net->train(BATCH_LMA, SSE, stop);
      eventLogger << " finished.\n";
      break;
    case Qt::Key_Escape:
      eventLogger << "Quitting application.\n";
      QApplication::quit();
      break;
    default:
      QGLWidget::keyPressEvent(keyEvent);
      break;
  }
}

TwoSpiralsDataSet::TwoSpiralsDataSet(const Mt& inputs, const Mt& outputs)
  : in(inputs), out(outputs), dataSet(this->in, this->out), visualization(0)
{
}

void TwoSpiralsDataSet::setVisualization(TwoSpiralsVisualization* visualization)
{
  this->visualization = visualization;
}

void TwoSpiralsDataSet::finishIteration(Learner& learner)
{
  if(visualization)
  {
    for(int x = 0; x < 100; x++)
    {
      for(int y = 0; y < 100; y++)
      {
        Vt in(2);
        in << (fpt)x/fpt(100), (fpt)y/fpt(100);
        Vt out = learner(in);
        visualization->predictClass(x, y, out(0, 0));
      }
    }
  }
}
