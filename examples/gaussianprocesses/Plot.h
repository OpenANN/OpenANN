#ifndef PLOT_H
#define PLOT_H

#include <io/DataSet.h>
#include <GP.h>
#include <QGLWidget>

class Plot : public QGLWidget
{
  Q_OBJECT
  int width, height;
  OpenANN::DataSet& dataSet;
  OpenANN::GP gp;

public:
  Plot(OpenANN::DataSet& dataSet, QWidget* parent = 0,
       const QGLWidget* shareWidget = 0, Qt::WindowFlags f = 0);

protected:
  virtual void initializeGL();
  virtual void resizeGL(int width, int height);
  virtual void paintGL();
  virtual void keyPressEvent(QKeyEvent* keyEvent);
};

#endif // PLOT_H