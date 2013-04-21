#include <OpenANN/OpenANN>
#include <OpenANN/io/Logger.h>
#include "IDXLoader.h"
#include "EnhancedDataSet.h"
#include <QGLWidget>
#include <QKeyEvent>
#include <QApplication>
#include <GL/glu.h>
#ifdef PARALLEL_CORES
#include <omp.h>
#endif

class DataVisualization : public QGLWidget
{
  EnhancedDataSet& dataSet;
  int xImages, yImages;
  int rows, cols;
  int width, height;
  int instance;
  OpenANN::Logger debugLogger;
public:
  DataVisualization(EnhancedDataSet& dataSet, int rows, int cols,
                    QWidget* parent = 0, const QGLWidget* shareWidget = 0,
                    Qt::WindowFlags f = 0)
      : dataSet(dataSet), xImages(5), yImages(5), rows(rows), cols(cols),
        width(800), height(600), instance(0), debugLogger(OpenANN::Logger::CONSOLE)
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
    int yOffset = -85;
    int zoom = -200;
    glTranslatef(xOffset,yOffset,zoom);

    glColor3f(0.0f,0.0f,0.0f);
    glLineWidth(5.0);

    float scale = 1.0;

    for(int yIdx = 0; yIdx < yImages; yIdx++)
    {
      for(int xIdx = 0; xIdx < xImages; xIdx++)
      {
        Eigen::VectorXd image = dataSet.getInstance(instance+xImages*yIdx+xIdx);

        float translateX = xIdx * (rows+1);
        float translateY = yIdx * (cols+1);
        glBegin(GL_QUADS);
        for(int row = 0; row < rows; row++)
        {
          for(int col = 0; col < cols; col++)
          {
            float c = image(row*cols+col);
            float x = translateX + col * scale;
            float y = translateY + (rows - row) * scale;
            glColor3f(c, c, c);
            glVertex2f((float) x, (float) y);
            glVertex2f((float) x+scale, (float) y);
            glVertex2f((float) x+scale, (float) y+scale);
            glVertex2f((float) x, (float) y+scale);
          }
        }
        glEnd();
      }
    }

    glColor3f(0.0f,0.0f,0.0f);
    renderText(20, 30, QString("Instance #") + QString::number(instance+1) +
               QString(" - ") + QString::number(instance+xImages*yImages),
               QFont("Helvetica", 20));

    glFlush();
  }

  virtual void keyPressEvent(QKeyEvent* keyEvent)
  {
    switch(keyEvent->key())
    {
      case Qt::Key_Up:
        instance += xImages * yImages;
        if(instance >= dataSet.samples() - xImages * yImages)
          instance = dataSet.samples() - xImages * yImages;
        update();
        break;
      case Qt::Key_Down:
        instance -= xImages * yImages;
        if(instance < 0)
          instance = 0;
        update();
        break;
      case Qt::Key_Left:
        dataSet.distort();
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

  IDXLoader loader(30, 30, 60000, 10000, directory);
  Distorter distorter;
  EnhancedDataSet dataSet(loader.trainingInput, loader.trainingOutput, 1, distorter);

  QApplication app(argc, argv);
  DataVisualization visual(dataSet, loader.padToX, loader.padToY);
  visual.show();
  visual.resize(800, 600);

  return app.exec();
}
