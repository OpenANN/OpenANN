#include "TwoSpiralsVisualization.h"

void TwoSpiralsDataSet::finishIteration(MLP& mlp)
{
  if(visualization)
  {
    for(int x = 0; x < 100; x++)
    {
      for(int y = 0; y < 100; y++)
      {
        Vt in(2);
        in << (fpt)x/fpt(100), (fpt)y/fpt(100);
        Vt out = mlp(in);
        visualization->predictClass(x, y, out(0, 0));
      }
    }
  }
}
