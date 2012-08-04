#include "Plot.h"
#include <QKeyEvent>
#include <GL/glu.h>

Plot::Plot(OpenANN::DataSet& dataSet, QWidget* parent,
           const QGLWidget* shareWidget, Qt::WindowFlags f)
    : QGLWidget(parent, shareWidget, f), dataSet(dataSet),
      gp(1.0, 1.0, 0.27, 0.0, 0.0)
{
  gp.trainingSet(dataSet);
  gp.buildModel();
}

void Plot::initializeGL()
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

void Plot::resizeGL(int width, int height)
{
  this->width = width;
  this->height = height;
  glViewport(0, 0, width, height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(45.0f, (float) width / (float) height, 1.0f, 100.0f);
  glMatrixMode(GL_MODELVIEW);
  glClearColor(1.0, 1.0, 1.0, 1.0);
  glClearDepth(1.0f);
}

void Plot::paintGL()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glLoadIdentity();
  glTranslatef(0.0f,0.0f,-3.0f);
  glColor3f(0.0f,0.0f,0.0f);
  glLineWidth(1.0);

  glBegin(GL_LINES);
    glVertex2f(-1.0f, 0.0f);
    glVertex2f(1.0f, 0.0f);
    glVertex2f(0.0f, -1.0f);
    glVertex2f(0.0f, 1.0f);
  glEnd();

  glBegin(GL_POINTS);
    for(int n = 0; n < dataSet.samples(); n++)
    {
      glVertex2f(dataSet.getInstance(n)(0), dataSet.getTarget(n)(0));
    }
  glEnd();

  glColor4f(0.5f,0.5f,0.5f,0.5);
  glBegin(GL_QUAD_STRIP);
    for(int i = -1200; i <= 1200; i++)
    {
      fpt x = (fpt) i / 1000.0f;
      Vt in(1);
      in(0) = x;
      Vt out = gp(in);
      fpt y = out(0);
      fpt var = gp.variance();
      fpt stdDev = sqrt(var);
      glVertex2f(x, y+stdDev);
      glVertex2f(x, y-stdDev);
    }
  glEnd();

  glColor3f(1.0f,0.0f,0.0f);
  glBegin(GL_LINE_STRIP);
    for(int i = -1200; i <= 1200; i++)
    {
      fpt x = (fpt) i / 1000.0f;
      Vt in(1);
      in(0) = x;
      Vt out = gp(in);
      glVertex2f(x, out(0));
    }
  glEnd();

  glFlush();
}

void Plot::keyPressEvent(QKeyEvent* keyEvent)
{
  switch(keyEvent->key())
  {
    default:
      QGLWidget::keyPressEvent(keyEvent);
      break;
  }
}
