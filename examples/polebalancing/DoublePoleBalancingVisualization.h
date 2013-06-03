#ifndef DOUBLEPOLEBALANCINGVISUALIZATION_H_
#define DOUBLEPOLEBALANCINGVISUALIZATION_H_

#include <QGLWidget>

class DoublePoleBalancingVisualization : public QGLWidget
{
  Q_OBJECT
  int width, height;
  bool singlePole, fullyObservable, alphaBetaFilter, doubleExponentialSmoothing;
  double position, angle1, angle2, force;
  int pause;

public:
  DoublePoleBalancingVisualization(bool singlePole, bool fullyObservable,
                                   bool alphaBetaFilter, bool doubleExponentialSmoothing,
                                   QWidget* parent = 0, const QGLWidget* shareWidget = 0,
                                   Qt::WindowFlags f = 0);

protected:
  virtual void initializeGL();
  virtual void resizeGL(int width, int height);
  virtual void paintGL();
  virtual void keyPressEvent(QKeyEvent* keyEvent);
  void run();
};

#endif // DOUBLEPOLEBALANCINGVISUALIZATION_H_
