#pragma once

#include <QGLWidget>
#include <QKeyEvent>
#include <QCoreApplication>
#include <GL/glu.h>
#include <cmath>
#include <vector>
#include <map>
#include <numeric>
#include "DoublePoleBalancing.h"
#include "SinglePoleBalancing.h"
#include "NeuroEvolutionAgent.h"
#include <rl/RandomAgent.h>
#include <io/Logger.h>
#include "Test/Stopwatch.h"
#include <EigenWrapper.h>
#include <AssertionMacros.h>

class DoublePoleBalancingVisualization : public QGLWidget
{
  struct Configuration
  {
    int h;
    bool b;
    std::string a;
    bool compress;
    int m;
    bool operator<(const Configuration& o) const
    {
      return m < o.m;
    }
  };
  struct Evaluation
  {
    int failure;
    long episodes;
    long cycles;
    Evaluation operator+(const Evaluation& o) const
    {
      Evaluation res = {failure + o.failure, episodes + o.episodes, cycles + o.cycles};
      return res;
    }
    Evaluation operator-(const Evaluation& o) const
    {
      Evaluation res = {failure, episodes - o.episodes, cycles - o.cycles};
      return res;
    }
    Evaluation operator/(int runs) const
    {
      Evaluation res = {failure, (episodes + 0.5) / runs, (cycles + 0.5) / runs};
      return res;
    }
    Evaluation sqr() const
    {
      Evaluation res = {failure, episodes * episodes, cycles * cycles};
      return res;
    }
    Evaluation sqrt() const
    {
      Evaluation res = {failure, std::sqrt((double)episodes), std::sqrt((double)cycles)};
      return res;
    }
  };

  Q_OBJECT
  int width, height;
  bool singlePole, fullyObservable, alphaBetaFilter;
  double position, angle1, angle2, force;
  int pause;

public:
  DoublePoleBalancingVisualization(bool singlePole, bool fullyObservable, bool alphaBetaFilter,
                                   QWidget* parent = 0, const QGLWidget* shareWidget = 0, Qt::WindowFlags f = 0)
    : QGLWidget(parent, shareWidget, f),
      width(800), height(400),
      singlePole(singlePole), fullyObservable(fullyObservable), alphaBetaFilter(alphaBetaFilter),
      position(0.0), angle1(4.0*M_PI/180.0), angle2(0.0), force(0.0),
      pause(1000)
  {
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
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glClearDepth(1.0f);
  }

  virtual void paintGL()
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

    // Force
    glBegin(GL_LINES);
      glVertex2f(position, 1.5f);
      glVertex2f(position+force/10.0, 1.5f);
    glEnd();

    // Cart
    glBegin(GL_LINE_STRIP);
      for(float angle = 0.0f; angle < 360.0f; angle += 5.0f)
        glVertex2f(position-0.5f + wheelRadius + sin(angle*M_PI/180.0) * wheelRadius,
                  wheelRadius + cos(angle*M_PI/180.0) * wheelRadius);
    glEnd();
    glBegin(GL_LINE_STRIP);
      for(float angle = 0.0f; angle < 360.0f; angle += 5.0f)
        glVertex2f(position-0.5f + 1.0 - wheelRadius + sin(angle*M_PI/180.0) * wheelRadius,
                  wheelRadius + cos(angle*M_PI/180.0) * wheelRadius);
    glEnd();
    glRectf(position-0.5f, 2.0f*wheelRadius, position-0.5f+1.0f, cartHeight);

    // Poles
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

  virtual void keyPressEvent(QKeyEvent* keyEvent)
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
      case Qt::Key_B:
        benchmark();
        break;
      default:
        QGLWidget::keyPressEvent(keyEvent);
        break;
    }
  }

  void run()
  {
    Logger environmentLogger(Logger::CONSOLE), returnLogger(Logger::CONSOLE, "steps");

    Environment* env;
    if(singlePole)
      env = new SinglePoleBalancing(fullyObservable);
    else
      env = new DoublePoleBalancing(fullyObservable);
    Environment& environment = *env;
    NeuroEvolutionAgent agent(0, false, "linear", true, singlePole? 1 : 5, fullyObservable, alphaBetaFilter);
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
        angle2 = singlePole ? 0.0 : lastState( fullyObservable ? 4 : 2);
        agent.chooseAction();
        environmentLogger << "(s, a, r, s') = (" << lastState.transpose() << ", "
            << environment.getAction() << ", " << environment.reward() << ", "
            << environment.getState().transpose() << ")\n";
        force = environment.getAction()(0);
        update();
        QCoreApplication::processEvents();
        usleep(pause);
      }
      returnLogger << "Episode " << (double) i << ", " << (double) environment.stepsInEpisode() << " steps\n";
      if(environment.stepsInEpisode() >= 100000)
      {
        best = environment.stepsInEpisode();
        break;
      }
      else if(environment.stepsInEpisode() > best)
        best = environment.stepsInEpisode();
    }
    std::cout << "longest episode: " << best << " steps" << std::endl;

    delete env;
  }

  void benchmark()
  {
    Logger progressLogger(Logger::CONSOLE), resultLogger(Logger::FILE, "pole-balancing-results");

    int maxEpisodes = 100000;
    int runs = 1000;

    resultLogger << "Results - " << (singlePole ? "Single" : "Double")
        << " Pole Balancing with" << (fullyObservable ? "" : "out") << " verlocities"
        << (alphaBetaFilter ? " using alpha beta filters" : "") << "\n";
    resultLogger << "Maximal number of episodes is " << maxEpisodes << "\n";
    resultLogger << "Results are averaged over " << runs << " runs\n";

    std::vector<Configuration> configs;
    if(singlePole)
    {
      //                      h,    b,       a,     comp.,  m
      Configuration trial1 = {0,  false,  "linear", false,  0};
      Configuration trial2 = {0,  false,  "linear", true,   4};
      Configuration trial3 = {0,  false,  "linear", true,   3};
      Configuration trial4 = {0,  false,  "linear", true,   2};
      Configuration trial5 = {0,  false,  "linear", true,   1};
      configs.push_back(trial1);
      configs.push_back(trial2);
      if(fullyObservable || alphaBetaFilter)
      {
        configs.push_back(trial3);
        configs.push_back(trial4);
      }
      if(fullyObservable)
      {
        configs.push_back(trial5);
      }
    }
    else
    {
      int h = 0;
      //                      h,    b,       a,     comp.,  m
      Configuration trial1 = {h,  false,  "linear", false,  0};
      Configuration trial2 = {h,  false,  "linear", true,   6};
      Configuration trial3 = {h,  false,  "linear", true,   5};
      Configuration trial4 = {h,  false,  "linear", true,   4};
      configs.push_back(trial1);
      configs.push_back(trial2);
      configs.push_back(trial3);
      configs.push_back(trial4);
    }
    std::map<Configuration, std::vector<Evaluation> > evaluations;

    Environment* env;
    if(singlePole)
      env = new SinglePoleBalancing(fullyObservable);
    else
      env = new DoublePoleBalancing(fullyObservable);
    Environment& environment = *env;
    for(std::vector<Configuration>::iterator c = configs.begin(); c != configs.end(); c++)
    {
      progressLogger << "new configuration, m = " << c->m << "\n";
      Configuration& config = *c;
      evaluations[config] = std::vector<Evaluation>();
      for(int run = 0; run < runs; run++)
      {
        NeuroEvolutionAgent agent(config.h, config.b, config.a, config.compress, config.m, fullyObservable, alphaBetaFilter);
        agent.abandoneIn(environment);
        progressLogger << "run " << (run+1) << " (L=" << agent.dimension() << ") - ";
        Evaluation eval = {0, 0, 0};
        bool success = false;
        Stopwatch sw;
        for(int episode = 0; episode < maxEpisodes; episode++)
        {
          environment.restart();
          while(!environment.terminalState())
            agent.chooseAction();
          eval.cycles += environment.stepsInEpisode();
          eval.episodes++;
          if(!isMatrixBroken(environment.getState()) && environment.successful())
          {
            success = true;
            break;
          }
        }
        progressLogger << sw.stop(Stopwatch::MILLISECOND) << " ms - " << success << "\n";
        eval.failure = !success;
        evaluations[config].push_back(eval);
      }

      Evaluation init = {0, 0, 0};
      Evaluation mean = std::accumulate(evaluations[config].begin(), evaluations[config].end(), init) / runs;
      std::vector<int> episodesVector(runs);
      std::vector<int> cyclesVector(runs);
      double sep = 0.0, scyc = 0.0;
      for(int i = 0; i < runs; i++)
      {
        double epdiff = mean.episodes - evaluations[config][i].episodes;
        sep += epdiff*epdiff/(double)runs;
        double cycdiff = mean.cycles - evaluations[config][i].cycles;
        scyc = cycdiff*cycdiff/(double)runs;
        episodesVector[i] = evaluations[config][i].episodes;
        cyclesVector[i] = evaluations[config][i].cycles;
      }
      sep = sqrt(sep);
      scyc = sqrt(scyc);
      std::sort(episodesVector.begin(), episodesVector.end());
      std::sort(cyclesVector.begin(), cyclesVector.end());
      int minEpisodes = episodesVector[0];
      int maxEpisodes = episodesVector[runs-1];
      int medianEpisodes = episodesVector[runs/2];
      int minCycles = cyclesVector[0];
      int maxCycles = cyclesVector[runs-1];
      int medianCycles = cyclesVector[runs/2];
      resultLogger << "h = " << config.h << ", b = " << config.b
          << ", a = " << config.a << ", m = " << config.m << "\n";
      resultLogger << "mean:\t" << mean.episodes << "\t" << mean.cycles << "\n";
      resultLogger << "stdDev:\t" << sep << "\t" << scyc << "\n";
      resultLogger << "min:\t" << minEpisodes << "\t" << minCycles << "\n";
      resultLogger << "max:\t" << maxEpisodes << "\t" << maxCycles << "\n";
      resultLogger << "median:\t" << medianEpisodes << "\t" << medianCycles << "\n";
      resultLogger << "failures:\t" << mean.failure << "\n";
    }

    delete env;
  }
};
