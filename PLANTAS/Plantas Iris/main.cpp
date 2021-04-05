#include "mlp.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>
#include "arquivo.h"
#include <string.h>

using namespace std;
using namespace cv;

int main(){
    FILE *arq;
    String nome;
    int qtd_plantas = 90; // 30 de cada tipo de plantas
    double taxa[5]={0.15,0.35,0.55,0.75,0.95};
    int Nepocas[4] = {60,120,240,500};

    for(int cama = 1; cama<=2; cama++){ // camadas
            for(int n = 0; n < 5;n++){ //taxa de aprendizado

                unsigned long inp=4;  // Numero de entradas da RNA
                unsigned long outp=2; // Numero de saidas

                unsigned long hid=2;       // Numero de neuronios escondidos
                unsigned long layers[3];    // Vetor para defini��o das camadas
                layers[0]=inp;              // Defini��o da primeira camada (de entradas): 2 elementos
                layers[1]=hid;              // Defini��o da segunda camada (de neuronios escondidos): 2 elementos
                layers[2]=outp;             // Defini��o da terceira camada (de neuronios de saida): 1 elemento
                unsigned long layers2[4];    // Vetor para defini��o das camadas
                layers2[0]=inp;              // Defini��o da primeira camada (de entradas): 2 elementos
                layers2[2]=hid;              // Defini��o da segunda camada (de neuronios escondidos): 2 elementos
                layers2[2]=hid;
                layers2[3]=outp;             // Defini��o da terceira camada (de neuronios de saida): 1 elemento

                MLP rna;

                if (cama == 1)
                    remove("pesos1.txt");
                else if(cama==2)
                    remove("pesos2.txt");

                float taxaAprend = taxa[n];   // Taxa de aprendizagem
                if(cama = 1){
                    rna.SetData(taxaAprend, layers,3);
                }else{
                    rna.SetData(taxaAprend, layers2,4);
                }


                if (cama == 1){
                    cout<<"chegou aqui";
                    arq = fopen("pesos1.txt","r");
                    if (arq == NULL){
                        cout<<"chegou aqui2";
                        rna.RandomizeWB();  // Gera pesos aleatorios para pesos e bias
                        cout<<"passou aqui";
                        rna.salvarPesos("pesos1.txt");
                    }
                } else if (cama == 2){
                     cout<<"chegou aqui3";
                     arq = fopen("pesos2.txt","r");
                     if (arq == NULL){
                        cout<<"chegou aqui4";
                        rna.RandomizeWB();  // Gera pesos aleatorios para pesos e bias
                        cout<<"passou aqui";
                        rna.salvarPesos("pesos2.txt");
                     }
                }

                //pesos
                //rna.RandomizeWB();  // Gera pesos aleatorios para pesos e bias
                //remove("wbs.txt");
                //rna.salvarPesos("wbs.txt");

                for(int Nepo = 0 ; Nepo < 4; Nepo++){ // numero de epocas

                    int nPadroes =3;

                    //vetor que armazena as entradas
                    double **tr_inp = (double**)malloc(qtd_plantas*sizeof(double*));
                    for (int i = 0; i < qtd_plantas; i++){
                         tr_inp[i] = (double*)malloc(inp * sizeof(double));
                    }

                    lerArquivo(tr_inp);


                    //saidas
                    double** tr_out = (double**) malloc( nPadroes * sizeof (double*));
                    for (int i = 0; i < nPadroes; ++i)
                            tr_out[i] = (double*) malloc( outp * sizeof (double)); // Vetor de saidas desejadas

                    //SAIDAS DESEJADAS
                    //setosa
                    tr_out[0][0]=0;
                    tr_out[0][1]=0;

                    //Versicolor
                    tr_out[1][0]=0;
                    tr_out[1][1]=1;

                    //virginica
                    tr_out[2][0]=1;
                    tr_out[2][1]=1;

                    double epoca=0;       // Contador de �pocas
                    epoca=Nepocas[Nepo];

                    //rna.carregarPesos("wbs.txt");

                    if (cama == 1)
                        rna.carregarPesos("pesos1.txt");
                    else
                        rna.carregarPesos("pesos2.txt");
                    double *erro_global_medio=(double*)malloc(epoca*sizeof(double));
                    float perc=0, max=epoca-1;


                    //FAZENDO O TREINAMENTO
                    for(int i=0;i<epoca;i++){
                        perc=(i*100.0)/max;
                        system ("cls");
                        cout<<perc<<"% concluido (treino).\n";
                        double erro_interacao;
                        double media_erro_interacao=0;
                        for(int j=0;j<qtd_plantas; j++){ // 4 vezes
                            if(j<30){
                                //Setosa
                                rna.Train(tr_inp[j],tr_out[0],&erro_interacao); // Treina rede com: Entradas e saidas desejadas
                            }
                            else if(j>=30 && j<60){
                                //versicolor
                                rna.Train(tr_inp[j],tr_out[1],&erro_interacao); // Treina rede com: Entradas e saidas desejadas
                            }
                            else{
                                //virginica
                                rna.Train(tr_inp[j],tr_out[2],&erro_interacao);
                            }
                            media_erro_interacao += erro_interacao;
                        }
                        media_erro_interacao/=qtd_plantas;
                        erro_global_medio[i] = media_erro_interacao;
                    }

                    int camadas = cama;
                    int numero = Nepocas[Nepo];
                    //salvando os erros
                    string strE = to_string(numero);
                    string strT = to_string(taxaAprend);
                    string strC = to_string(camadas);
                    nome = "ERROS-Epoca="+strE+"-Taxa="+strT+"-Camada="+strC+".txt";
                    char* nomes = (char*)nome.c_str();
                    criarArquivo(nomes,Nepocas[Nepo],erro_global_medio);


            }//END TAXA
        }//END EPOCA
    }//END CAMADA

    return 1;
}
