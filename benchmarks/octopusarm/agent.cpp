#include "agent.h"
#include <MLP.h>
#include <optimization/IPOPCMAES.h>
#include <AssertionMacros.h>
#include <EigenWrapper.h>
#include <io/Logger.h>
#include <Random.h>
#include <ctime>
#include <fstream>
#include <stdlib.h>
#include <time.h>

using namespace OpenANN;

/**
 * \page OctopusArm Octopus Arm
 *
 * TODO: description
 */

int num_states = 0, num_actions = 0;
IPOPCMAES opt;
MLP mlp;
double episodeReturn;
Logger logger(Logger::CONSOLE);
int hiddenUnits;
int parameters;
double bestReturn;
Vt bestParameters;

int agent_init(int num_state_variables, int num_action_variables, int argc, const char *agent_param[])
{
  num_states = num_state_variables;
  num_actions = num_action_variables;

  parameters = 0;
  hiddenUnits = 10;
  if(argc > 0)
    parameters = atoi(agent_param[0]);
  if(argc > 1)
    hiddenUnits = atoi(agent_param[1]);

  mlp.input(num_states)
    .fullyConnectedHiddenLayer(hiddenUnits, MLP::TANH, parameters > 0 ? parameters : -1, "random");
  mlp.output(num_actions, MLP::SSE, MLP::SIGMOID, parameters > 0 ? hiddenUnits+1 : -1, "random");
  bestParameters = mlp.currentParameters();
  bestReturn = -std::numeric_limits<double>::max();

  StopCriteria stop;
  stop.maximalFunctionEvaluations = 5000;
  stop.maximalRestarts = 1000;
  opt.setOptimizable(mlp);
  opt.setStopCriteria(stop);
  opt.restart();

  logger << mlp.dimension() << " parameters, " << num_states
      << " state components, " << num_actions << " action components\n";
  return 0;
}

const char* agent_get_name()
{
  std::stringstream stream;
  stream << "Neuroevolution_h_" << hiddenUnits << "_p_" << parameters;
  return stream.str().c_str();
}

Vt convert(double state[])
{
  Vt s(num_states);
  for(int i = 0; i < num_states; i++)
    s(i) = state[i];
  return s;
}

void convert(const Vt& action, double* out)
{
  for(int i = 0; i < num_actions; i++)
    out[i] = action(i);
}

int chooseAction(double state_data[], double out_action[])
{
  Vt state = convert(state_data);
  OPENANN_CHECK_MATRIX_BROKEN(state);
  Vt y = mlp(state);
  Vt action(num_actions);

  action = y;
  if(isMatrixBroken(action))
    action.fill(0.0);

  convert(action, out_action);
  return 0;
}

int agent_start(double state_data[], double out_action[])
{
  mlp.setParameters(opt.getNext());
  episodeReturn = 0;
  chooseAction(state_data, out_action);
  return 0;
}

int agent_step(double state_data[], double reward, double out_action[])
{
  chooseAction(state_data, out_action);
  episodeReturn += reward;
  return  0;
}

int agent_end(double reward) {
  episodeReturn += reward;
  logger << "agend end, return = " << episodeReturn << "\n";
  if(episodeReturn > bestReturn)
  {
    bestReturn = episodeReturn;
    bestParameters = mlp.currentParameters();
  }
  RandomNumberGenerator rng;
  opt.setError(-episodeReturn+0.1*episodeReturn*rng.sampleNormalDistribution<double>());
  if(opt.terminated())
    opt.restart();
  return 0;
}

void agent_cleanup()
{
  time_t rawtime;
  struct tm* timeinfo;
  std::time(&rawtime);
  timeinfo = std::localtime(&rawtime);
  std::ofstream result((std::string(std::asctime(timeinfo)).substr(0, 24) + "-best.log").c_str());
  result << "Best Return = " << bestReturn << std::endl;
  result << "Hidden Units = " << hiddenUnits << std::endl;
  result << "Parameters = " << parameters << std::endl;
  result << "Best Parameters = " << std::endl << bestParameters << std::endl;
  result.close();
}
