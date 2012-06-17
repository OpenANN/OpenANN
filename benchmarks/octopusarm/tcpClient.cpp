/* fpont 12/99 */
/* pont.net    */
/* tcpClient.c */

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h> /* close */

#include "protocol.h"

/* Do all of the socket initialization, bind the local address
 * connect to the server address, etc...
 */
int initSocket(const char* serverAddress, int serverPort, int* sd) {
  int rc;
  struct sockaddr_in localAddr, servAddr;
  struct hostent *h;

  h = gethostbyname(serverAddress);
  if(h==NULL) {
    printf("Unknown host '%s'\n",serverAddress);
    return -1;
  }

  servAddr.sin_family = h->h_addrtype;
  memcpy((char *) &servAddr.sin_addr.s_addr, h->h_addr_list[0], h->h_length);
  servAddr.sin_port = htons(serverPort);

  /* create socket */
  *sd = socket(AF_INET, SOCK_STREAM, 0);
  if(*sd<0) {
    perror("cannot open socket ");
    return -2;
  }

  /* bind any port number */
  localAddr.sin_family = AF_INET;
  localAddr.sin_addr.s_addr = htonl(INADDR_ANY);
  localAddr.sin_port = htons(0);

  rc = bind(*sd, (struct sockaddr *) &localAddr, sizeof(localAddr));
  if(rc<0) {
    printf("Cannot bind port TCP %u\n",serverPort);
    perror("error ");
    close(*sd);
    return -3;
  }

  /* connect to server */
  rc = connect(*sd, (struct sockaddr *) &servAddr, sizeof(servAddr));
  if(rc<0) {
    perror("cannot connect ");
    close(*sd);
    return -4;
  }
  return 0;
}

int main(int argc, char *argv[]) {

  int sd, rc;
  unsigned int port = 0;
  unsigned int episode = 0;

  /* Check arguments, assign port and episode values */
  if(argc < 4) { 
    printf("usage: %s <server> <port> <episode> [<agent-specific parameters>]\n",argv[0]);
    return 1;
  } else if((port = atoi(argv[2])) == 0 ) {
    printf("The port number is invalid\n");
    return 1;
  } else if((episode = atoi(argv[3])) == 0 ) {
    printf("The episode number is invalid\n");
    return 1;
  }

  rc = initSocket(argv[1], port, &sd);
  if(rc<0) {
    printf("could not initialize socket ");
    return 1;

  }

	// Now start to send the data
  rc = performInit(sd, argc-4, (const char**)&argv[4]);
  if(rc<0) {
    perror("failed intialization of the agent");
    return 1;
  }
  for(unsigned int i = 0; i < episode; ++i) {
    rc = performEpisode(sd);
    if(rc<0) break;
  }
  performCleanup();
  close(sd);

  if(rc >= 0) {
    return 0;
  }
  return rc;
}

