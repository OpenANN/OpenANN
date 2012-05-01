#pragma once

#include <OpenANN>
#include <AssertionMacros.h>
#include <io/Logger.h>
#include <io/DirectStorageDataSet.h>
#include <Eigen/Dense>
#include <QGLWidget>
#include <QKeyEvent>
#include <QMutex>

using namespace OpenANN;

class TwoSpiralsVisualization;

class TwoSpiralsDataSet : public DataSet
{
  Mt in, out;
  DirectStorageDataSet dataSet;
  TwoSpiralsVisualization* visualization;
public:
  TwoSpiralsDataSet(const Mt& inputs, const Mt& outputs)
    : in(inputs), out(outputs), dataSet(this->in, this->out), visualization(0)
  {
  }
  void setVisualization(TwoSpiralsVisualization* visualization)
  {
    this->visualization = visualization;
  }
  virtual ~TwoSpiralsDataSet() {}
  virtual int samples() { return dataSet.samples(); }
  virtual int inputs() { return dataSet.inputs(); }
  virtual int outputs() { return dataSet.outputs(); }
  virtual Vt& getInstance(int i) { return dataSet.getInstance(i); }
  virtual Vt& getTarget(int i) { return dataSet.getTarget(i); }
  virtual void finishIteration(MLP& mlp);
};

class TwoSpiralsVisualization : public QGLWidget
{
  Q_OBJECT
  int width, height;
  QMutex classesMutex;
  fpt classes[100][100];
  TwoSpiralsDataSet trainingSet;
  TwoSpiralsDataSet testSet;
  bool showTraining, showTest, showPrediction, showSmooth;
  MLP* mlp;
  StopCriteria stop;
  Logger eventLogger;

public:
  TwoSpiralsVisualization(
    const Mt& trainingInput,
    const Mt& trainingOutput,
    const Mt& testInput,
    const Mt& testOutput)
    : width(500), height(500),
      trainingSet(trainingInput, trainingOutput), testSet(testInput, testOutput),
      showTraining(true), showTest(true), showPrediction(true), showSmooth(true),
      mlp(new MLP), eventLogger(Logger::CONSOLE)
  {
    for(int x = 0; x < 100; x++)
    {
      for(int y = 0; y < 100; y++)
      {
        classes[x][y] = 0;
      }
    }
    mlp->input(trainingInput.rows())
      .fullyConnectedHiddenLayer(20, MLP::TANH, 3)
      .fullyConnectedHiddenLayer(20, MLP::TANH, 6)
      .output(trainingOutput.rows(), MLP::SSE, MLP::TANH, 1)
      .trainingSet(trainingSet)
      .testSet(testSet);
    stop.maximalIterations = 10000;
    stop.minimalSearchSpaceStep = 1e-18;
    QObject::connect(this, SIGNAL(updatedData()), this, SLOT(repaint()));
  }

  virtual ~TwoSpiralsVisualization()
  {
    delete mlp;
  }

  void predictClass(int x, int y, fpt predictedClass)
  {
    classesMutex.lock();
    classes[x][y] = predictedClass;
    classesMutex.unlock();
    if(x == 99 && y == 99)
      emit updatedData();
  }

protected:
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
    gluPerspective(45.0f, (float) width / (float) height, 1.0f, 100.0f);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClearDepth(1.0f);
  }

  virtual void paintGL()
  {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    glTranslatef(-0.5f,-0.5f,-1.25f);
    glColor3f(0.0f,0.0f,0.0f);

    if(showPrediction)
    {
      for(int x = 0; x < 100; x++)
      {
        for(int y = 0; y < 100; y++)
        {
          classesMutex.lock();
          float c;
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

    if(showTraining)
    {
      glBegin(GL_POINTS);
        for(int n = 0; n < trainingSet.samples(); n++)
        {
          if(trainingSet.getTarget(n)(0) > 0.0)
            glColor3f(1.0f, 0.0f, 0.0f);
          else
            glColor3f(1.0f, 1.0f, 0.0f);
          glVertex2d(trainingSet.getInstance(n)(0), trainingSet.getInstance(n)(1));
        }
      glEnd();
    }

    if(showTest)
    {
      glBegin(GL_POINTS);
        for(int n = 0; n < testSet.samples(); n++)
        {
          if(testSet.getTarget(n)(0) > 0.0)
            glColor3f(1.0f, 0.0f, 0.0f);
          else
            glColor3f(1.0f, 1.0f, 0.0f);
          glVertex2d(testSet.getInstance(n)(0), testSet.getInstance(n)(1));
        }
      glEnd();
    }

    glFlush();
  }

  virtual void keyPressEvent(QKeyEvent* keyEvent)
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
        eventLogger << "Training with restart (" << mlp->dimension() << " parameters)...";
        mlp->training(MLP::BATCH_LMA);
        trainingSet.setVisualization(this);
        mlp->fit(stop);
        eventLogger << " finished.\n";
        break;
      case Qt::Key_S:
        eventLogger << "Training without restart (" << mlp->dimension() << " parameters)...";
        mlp->training(MLP::BATCH_LMA, false);
        trainingSet.setVisualization(this);
        mlp->fit(stop);
        eventLogger << " finished.\n";
        break;
      case Qt::Key_B:
      {
        eventLogger << "Starting Benchmark.\n";
        int runs = 100;
        std::vector<std::vector<int> > compressions;
        std::vector<int> none;
        std::vector<int> no_compression(3);
        no_compression[0] = 3; no_compression[1] = 21; no_compression[2] = 21;
        std::vector<int> mid_compression(3);
        mid_compression[0] = 3; mid_compression[1] = 12; mid_compression[2] = 12;
        std::vector<int> high_compression(3);
        high_compression[0] = 3; high_compression[1] = 6; high_compression[2] = 6;
        std::vector<int> very_high_compression(3);
        very_high_compression[0] = 3; very_high_compression[1] = 6; very_high_compression[2] = 1;

        compressions.push_back(none);
        compressions.push_back(no_compression);
        compressions.push_back(mid_compression);
        compressions.push_back(high_compression);
        compressions.push_back(very_high_compression);
        for(size_t c = 0; c < compressions.size(); c++)
        {
          for(int run = 0; run < runs; run++)
          {
            MLP m;
            m.input(trainingSet.inputs());
            if(compressions[c].size() == 3)
              m.fullyConnectedHiddenLayer(20, MLP::TANH, compressions[c][0]);
            else
              m.fullyConnectedHiddenLayer(20);
            if(compressions[c].size() == 3)
              m.fullyConnectedHiddenLayer(20, MLP::TANH, compressions[c][1]);
            else
              m.fullyConnectedHiddenLayer(20);
            if(compressions[c].size() == 3)
              m.output(trainingSet.outputs(), MLP::SSE, MLP::TANH, compressions[c][2]);
            else
              m.output(trainingSet.outputs(), MLP::SSE);
            m.trainingSet(trainingSet)
             .testSet(testSet);
            m.training(MLP::BATCH_LMA, false);
            m.fit(stop);
          }
        }
        eventLogger << "Finished Benchmark.\n";
        break;
      }
      case Qt::Key_P:
        eventLogger << "Parameters:\n" << mlp->currentParameters().transpose() << "\n";
        break;
      default:
        QGLWidget::keyPressEvent(keyEvent);
        break;
    }
  }

signals:
  void updatedData();
};
