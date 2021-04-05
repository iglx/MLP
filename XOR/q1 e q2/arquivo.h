#ifndef ARQUIVO_H
#define ARQUIVO_H
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


float * lerArquivo(char *arquivo, double **plantas);
void  criarArquivo(char* arquivo, int nepoca, double *erro_global_medio);

#endif // ARQUIVO_H
