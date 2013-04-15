#include "DoublePoleBalancingVisualization.h"
#include <QApplication>
#include <QCoreApplication>
#include <QKeyEvent>
#include <GL/glu.h>
#include "DoublePoleBalancing.h"
#include "SinglePoleBalancing.h"
#include "NeuroEvolutionAgent.h"
#include <OpenANN/rl/RandomAgent.h>
#include <OpenANN/io/Logger.h>
#include <OpenANN/util/EigenWrapper.h>
#include <OpenANN/util/AssertionMacros.h>
#include <Test/Stopwatch.h>
#include <cmath>
#include <map>
#include <numeric>
#include <vector>

DoublePoleBalancingVisualization::DoublePoleBalancingVisualization(
    bool singlePole, bool fullyObservable, bool alphaBetaFilter,
    bool doubleExponentialSmoothing, QWidget* parent,
    const QGLWidget* shareWidget, Qt::WindowFlags f)
    : QGLWidget(parent, shareWidget, f),
      width(800), height(400), singlePole(singlePole),
      fullyObservable(fullyObservable), alphaBetaFilter(alphaBetaFilter),
      doubleExponentialSmoothing(doubleExponentialSmoothing),
      position(0.0), angle1(4.0*M_PI/180.0), angle2(0.0), force(0.0),
      pause(1000)
{
}

void DoublePoleBalancingVisualization::initializeGL()
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

void DoublePoleBalancingVisualization::resizeGL(int width, int height)
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

void DoublePoleBalancingVisualization::paintGL()
{
  float wheelRadius = 0.075;
  float cartHeight = 0.5f;
  float pole1Length = 0.5f;
  float pole2Length = 0.05;

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glLoadIdentity();
  glTranslatef(0.0f,-0.5f,-3.0f);

  glColor3f(0.0f,0.0f,0.0f);
  glLineWidth(1.0);
  glBegin(GL_LINES);
    glVertex2f(-2.4f, 0.0f);
    glVertex2f(2.4f, 0.0f);
  glEnd();
  for(float tic = -2.4f; tic <= 2.4f; tic += 0.6f)
  {
    glBegin(GL_LINES);
      glVertex2f(tic, -0.1f);
      glVertex2f(tic, 0.0f);
    glEnd();
  }

  glLineWidth(10.0);

  // force
  glBegin(GL_LINES);
    glVertex2f(position, 1.5f);
    glVertex2f(position+force/10.0, 1.5f);
  glEnd();

  // cart
  glBegin(GL_LINE_STRIP);
    for(float angle = 0.0f; angle < 360.0f; angle += 5.0f)
      glVertex2f(position-0.5f + wheelRadius + sin(angle*M_PI/180.0)
          * wheelRadius, wheelRadius + cos(angle*M_PI/180.0) * wheelRadius);
  glEnd();
  glBegin(GL_LINE_STRIP);
    for(float angle = 0.0f; angle < 360.0f; angle += 5.0f)
      glVertex2f(position-0.5f + 1.0 - wheelRadius + sin(angle*M_PI/180.0)
          * wheelRadius, wheelRadius + cos(angle*M_PI/180.0) * wheelRadius);
  glEnd();
  glRectf(position-0.5f, 2.0f*wheelRadius, position-0.5f+1.0f, cartHeight);

  // poles
  glColor3f(1.0f,0.0f,0.0f);
  float pole1height = sin(angle1+M_PI/2.0f) * pole1Length;
  float pole1displacement = -cos(angle1+M_PI/2.0f) * pole1Length;
  glBegin(GL_LINES);
    glVertex2f(position-0.5f+0.45f, cartHeight);
    glVertex2f(position-0.5f+0.45f+pole1displacement, cartHeight+pole1height);
  glEnd();
  float pole2height = sin(angle2+M_PI/2.0f) * pole2Length;
  float pole2displacement = -cos(angle2+M_PI/2.0f) * pole2Length;
  glBegin(GL_LINES);
    glVertex2f(position-0.5f+0.55f, cartHeight);
    glVertex2f(position-0.5f+0.55f+pole2displacement, cartHeight+pole2height);
  glEnd();

  glFlush();
}

void DoublePoleBalancingVisualization::keyPressEvent(QKeyEvent* keyEvent)
{
  switch(keyEvent->key())
  {
    case Qt::Key_R:
      run();
      break;
    case Qt::Key_Plus:
      pause = (int) ((double) pause * 0.9);
      break;
    case Qt::Key_Minus:
      pause = (int) ((double) pause * 1.1);
      break;
    case Qt::Key_Escape:
      QApplication::exit();
      break;
    default:
      QGLWidget::keyPressEvent(keyEvent);
      break;
  }
}

void DoublePoleBalancingVisualization::run()
{
  Logger environmentLogger(Logger::NONE), returnLogger(Logger::CONSOLE, "steps");

  Environment* env;
  if(singlePole)
    env = new SinglePoleBalancing(fullyObservable);
  else
    env = new DoublePoleBalancing(fullyObservable);
  Environment& environment = *env;
  NeuroEvolutionAgent agent(0, false, "linear", true, singlePole? 1 : 5,
      fullyObservable, alphaBetaFilter, doubleExponentialSmoothing);
  agent.abandoneIn(environment);
  int best = -1;
  for(int i = 0; i < 100000; i++)
  {
    environment.restart();
    while(!environment.terminalState())
    {
      Environment::State lastState = environment.getState();
      OPENANN_CHECK_MATRIX_BROKEN(lastState);
      position = lastState(0);
      angle1 = lastState(fullyObservable ? 2 : 1);
      angle2 = singlePole ? 0.0 : lastState(fullyObservable ? 4 : 2);
      agent.chooseAction();
      environmentLogger << "(s, a, r, s') = (" << lastState.transpose()
          << ", " << environment.getAction() << ", " << environment.reward()
          << ", " << environment.getState().transpose() << ")\n";
      force = environment.getAction()(0);
      update();
      QCoreApplication::processEvents();
      usleep(pause);
    }
    returnLogger << "Episode " << (double) i << ", "
        << (double) environment.stepsInEpisode() << " steps\n";
    if(environment.stepsInEpisode() >= 100000)
    {
      best = environment.stepsInEpisode();
      break;
    }
    else if(environment.stepsInEpisode() > best)
      best = environment.stepsInEpisode();
  }
  returnLogger << "longest episode: " << best << " steps\n";

  delete env;
}
