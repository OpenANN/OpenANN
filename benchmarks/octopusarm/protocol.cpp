#include <sys/types.h>
#include <sys/socket.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>

#include "protocol.h"
#include "agent.h"

/*** PROTOCOL DEFINITIONS ***/
char SEPARATOR = '\n';
#define CMD_GETTASK   "GET_TASK"
#define CMD_STARTLOG  "START_LOG"
#define CMD_START   "START"
#define CMD_STEP    "STEP"
#define CMD_SETSTATE  "SET_STATE"



#define BUFFER_LENGTH 1024
char buffer[BUFFER_LENGTH];

int numStates = 0, numActions = 0;

/* Retrieves a single token from the socket. Tokens are separeted
* by the SEPARATOR character.
*/
const char* getNextTokenFromSocket(int sd)
{
  const char* ret = buffer; // We always return our buffer, unless there is an error
  int rc, i;
  char current;

  // Currently this naive approach only works for messages
  // smaller than BUFFER_LENGTH...
  for(i = 0; i < BUFFER_LENGTH - 1; i++)
  {
    rc = recv(sd, &current, 1, 0);
    if(rc < 0)
    {
      // There was an error reading on the socket
      ret = NULL;
      perror("Error reading next token: ");
      break;
    }
    else if(current == SEPARATOR)
    {
      break;
    }
    else
    {
      buffer[i] = current;
    }
  }
  buffer[i] = '\0'; // Null terminator on the buffer
  return ret;
}

/* Retrieves a token and converts to an int. */
int getIntFromSocket(int sd, int* data)
{
  const char* token = getNextTokenFromSocket(sd);
  if(token == NULL)
  {
    // There was an error reading the socket
    return -1;
  }
  else
  {
    *data = atoi(token);
    return 0;
  }
}

/* Retrieves a token and converts to a double. */
int getDoubleFromSocket(int sd, double* data)
{
  const char* token = getNextTokenFromSocket(sd);
  if(token == NULL)
  {
    // There was an error reading the socket
    return -1;
  }
  else
  {
    *data = atof(token);
    return 0;
  }
}

/* Retrieves a token and converts to a boolean value (0 or 1). */
int getFlagFromSocket(int sd, int* data)
{
  const char* token = getNextTokenFromSocket(sd);
  if(token == NULL)
  {
    // There was an error reading the socket
    return -1;
  }
  else if(token[0] == '0')
  {
    *data = 0;
    return 0;
  }
  else if(token[0] == '1')
  {
    *data = 1;
    return 0;
  }
  else
  {
    // Invalid value for flag
    printf("Received invalid flag value: %d\n", token[0]);
    return -1;
  }
}

/* Retrieves a sequence of numStates token and converts them to doubles. */
int getStatesFromSocket(int sd, double states[])
{
  int receivedStateCount = 0, i;
  int rc = getIntFromSocket(sd, &receivedStateCount);
  if(rc >= 0 && receivedStateCount == numStates)
  {
    for(i = 0; i < numStates; i++)
    {
      rc = getDoubleFromSocket(sd, &(states[i]));
      if(rc < 0) break;
    }
  }
  else
  {
    if(receivedStateCount != numStates)
    {
      printf("Mismatch in number of states numStates=%d, received:%d\n", numStates, receivedStateCount);
    }
    rc = -1;
  }
  return rc;
}

/* Sends a single token on the socket. Tokens should not contain any SEPARATOR chars. */
int sendTokenOnSocket(int sd, const char* token)
{
  int len = strlen(token);
  // We send the message
  int rc = send(sd, token, len, 0);
  if(rc == len)
  {
    // followed by the separator
    rc = send(sd, &(SEPARATOR), 1, 0);
    if(rc != 1) rc = -1;
  }
  else
  {
    // Not all bytes were sent out
    rc = -1;
  }
  return rc;
}

int sendIntOnSocket(int sd, int data)
{
  char tokenBuffer[BUFFER_LENGTH];
  sprintf(tokenBuffer, "%d", data);
  return sendTokenOnSocket(sd, tokenBuffer);
}

int sendDoubleOnSocket(int sd, double data)
{
  char tokenBuffer[BUFFER_LENGTH];
  sprintf(tokenBuffer, "%.16g", data);
  return sendTokenOnSocket(sd, tokenBuffer);
}

/* Sends numActions on the socket. */
int sendActionsOnSocket(int sd, double actions[])
{
  int len = 0;
  char buffer[10240];
  int i;

  sprintf(buffer, "%d%c", numActions, SEPARATOR);
  len = strlen(buffer);
  for(i = 0; i < numActions; i++)
  {
    sprintf(buffer + len, "%.16g%c", actions[i], SEPARATOR);
    len += strlen(buffer + len);
  }
  // We remove the last terminator
  len--;
  buffer[len] = '\0';
  return sendTokenOnSocket(sd, buffer);;
}

/* Removes all non-alphanumeric characters and replaces them with '_'. */
void clean_name(char* name)
{
  char* p;
  for(p = name; *p != '\0'; ++p)
  {
    if(!isalnum(*p))
    {
      *p = '_';
    }
  }
}

/* Retrieves the task specification and initializes the agent with it.
* Then it gets the agent name, cleans it and sends it to the environment.
*/
int performInit(int sd, int argc, const char* agent_params[])
{
  const char* agent_name;
  char* clean_agent_name;
  // Send Get_TASK command
  int rc = sendTokenOnSocket(sd, CMD_GETTASK);

  // Retrieve the number of states and actions
  if(rc >= 0) rc = getIntFromSocket(sd, &numStates);
  if(rc >= 0) rc = getIntFromSocket(sd, &numActions);
  if(rc >= 0) rc = agent_init(numStates, numActions, argc, agent_params);
  if(rc >= 0)
  {
    agent_name = agent_get_name();
    if(!agent_name) agent_name = "Unnamed";
    clean_agent_name = strdup(agent_name);
    clean_name(clean_agent_name);
    // Send START_LOG command and the agent name
    rc = sendTokenOnSocket(sd, CMD_STARTLOG);
    if(rc >= 0) rc = sendTokenOnSocket(sd, clean_agent_name);
    free(clean_agent_name);
  }

  return rc;
}

/* Runs an entire episode. */
int performEpisode(int sd)
{
  // Send the INIT command
  int rc = sendTokenOnSocket(sd, CMD_START);
  int terminal;
  double* pStates = (double*)malloc(numStates * sizeof(double));
  double* pActions = (double*)malloc(numActions * sizeof(double));
  double reward = 0;

  if(rc >= 0)
  {
    // Get terminality flag
    rc = getFlagFromSocket(sd, &terminal);
  }
  if(rc >= 0)
  {
    // Get initial states
    rc = getStatesFromSocket(sd, pStates);
  }
  if(rc >= 0)
  {
    if(terminal == 1)
    {
      // FIXME The reward here is still undefined...
      rc = agent_end(reward);
    }
    else
    {
      // Tell agent to start processing states and query
      // actions...
      rc = agent_start(pStates, pActions);
      while(terminal == 0 && rc >= 0)
      {
        // Send the step command with the new state
        rc = sendTokenOnSocket(sd, CMD_STEP);
        if(rc < 0) break;
        rc = sendActionsOnSocket(sd, pActions);
        if(rc < 0) break;
        // Get the reward, terminality flag and new state of
        // this step
        rc = getDoubleFromSocket(sd, &reward);
        if(rc < 0) break;
        rc = getFlagFromSocket(sd, &terminal);
        if(rc < 0) break;
        rc = getStatesFromSocket(sd, pStates);
        if(rc < 0) break;
        // If we are not in a terminal state, tell agent to step
        if(terminal == 0) rc = agent_step(pStates, reward, pActions);
      }
      if(rc >= 0)
      {
        // We have reached the terminal state (since there was
        // no error to get us out of the loop)
        rc = agent_end(reward);
      }
    }
  }
  return rc;
}

void performCleanup()
{
  agent_cleanup();
}
