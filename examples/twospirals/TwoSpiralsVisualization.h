#pragma once

#include <OpenANN>
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
  TwoSpiralsDataSet(const Mt& inputs, const Mt& outputs);
  void setVisualization(TwoSpiralsVisualization* visualization);
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
  TwoSpiralsVisualization(const Mt& trainingInput, const Mt& trainingOutput,
      const Mt& testInput, const Mt& testOutput);
  virtual ~TwoSpiralsVisualization();
  void predictClass(int x, int y, fpt predictedClass);

protected:
  virtual void initializeGL();
  virtual void resizeGL(int width, int height);
  virtual void paintGL();
  virtual void keyPressEvent(QKeyEvent* keyEvent);

signals:
  void updatedData();
};
