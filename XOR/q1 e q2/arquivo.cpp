#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>

using namespace std;

void lerArquivo(char *arquivo, double **plantas){
    FILE *arq;
    cout<<"chegou aqui";
    arq = fopen("plantas.txt","r");
    cout<<"passou aqui 2";

    if (arq == NULL){
        printf("Problemas na ABERTURA do arquivo\n");
        //exit(0);
    }
    cout<"Deu certo ler parece";

    float dados[4];
    //4.3,3.0,1.1,0.1,Iris-
int cont=0, x ;
    //estão divididos em 30 de cada
    //setosa, versicolor e virginica
   while(fscanf(arq,"%f,%f,%f,%f",&dados[0],&dados[1],&dados[2],&dados[3]) !=EOF){
        cout<<dados[0]<<" - "<<dados[1]<<"-"<<dados[2]<<"-"<<dados[3]<<"-"<<cont+1<<endl;
        for(x=0;x<4;x++)
            plantas[cont][x] = dados[x];
        //fprintf(stdout,"%f\n",c);
        cont++;
    }
    fclose(arq);
    system("pause");
}

void criarArquivo(char* arquivo, int nepoca, double *erro_global_medio){ // Salva erro

    int i,result;

    FILE *arq;
    arq = fopen(arquivo, "wt");	// Cria um arquivo texto para gravação
    if (arq == NULL)				// Se nào conseguiu criar
    {
        printf("Problemas na CRIACAO do arquivo\n");
        return;
    }


    for(i=1;i<=nepoca;i++){

                result = fprintf(arq,"%d,\t%lf,\n",i, erro_global_medio[i]);
                if (result == EOF)
                {
                    printf("Erro na Gravacao\n");
                    return;


    }

}
    fclose(arq);
}
