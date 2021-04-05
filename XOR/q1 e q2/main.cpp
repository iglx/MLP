#include "mlp.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>
#include "arquivo.h"

using namespace std;
using namespace cv;

int main(){
    String nome;
    double taxa[5]={0.15,0.35,0.55,0.75,0.95};
    int Nepocas[4] = {60,120,240,500};
    int Nepo;
    double *erro_global_medio;
    for(int cama = 1; cama<=2; cama++){ // camadas
//        MLP rna;
//        rna.RandomizeWB();  // Gera pesos aleatorios para pesos e bias
//        remove("wbs.txt");
//        rna.salvarPesos("wbs.txt");
        for(int n = 0; n < 5;n++){ //taxa de aprendizado

            unsigned long inp=2;        // Numero de entradas da RNA
            unsigned long outp=1;       // Numero de saidas


            unsigned long hid=2;        // Numero de neuronios escondidos
            unsigned long layers[3];    // Vetor para defini��o das camadas
            layers[0]=inp;              // Defini��o da primeira camada (de entradas): 2 elementos
            layers[1]=hid;              // Defini��o da segunda camada (de neuronios escondidos): 2 elementos
            layers[2]=outp;             // Defini��o da terceira camada (de neuronios de saida): 1 elemento
            unsigned long layers2[4];   // Vetor para defini��o das camadas
            layers2[0]=inp;             // Defini��o da primeira camada (de entradas): 2 elementos
            layers2[1]=hid;             // Defini��o da segunda camada (de neuronios escondidos): 2 elementos
            layers2[2]=hid;
            layers2[3]=outp;            // Defini��o da terceira camada (de neuronios de saida): 1 elemento

            //MLP rna; // Cria objeto tipo MLP (rede neural)
            MLP rna; // Cria objeto tipo MLP (rede neural)

            float taxaAprend = taxa[n];   // Taxa de aprendizagem
            if(cama = 1){
                rna.SetData(taxaAprend, layers,3);
            }else{
                rna.SetData(taxaAprend, layers2,4);
            }

            //pesos
            rna.RandomizeWB();  // Gera pesos aleatorios para pesos e bias
            remove("wbs.txt");
            rna.salvarPesos("wbs.txt");

        for(Nepo = 0 ; Nepo < 4; Nepo++){ // numero de epocas

                int nPadroes =4;

                double** tr_inp = (double**) malloc( nPadroes * sizeof (double*));
                for (int i = 0; i < nPadroes; ++i)
                    tr_inp[i] = (double*) malloc( inp * sizeof (double)); // Vetor de entradas para treinamento

                double** tr_out = (double**) malloc( nPadroes * sizeof (double*));
                for (int i = 0; i < nPadroes; ++i)
                    tr_out[i] = (double*) malloc( outp * sizeof (double)); // Vetor de saidas desejadas

                tr_inp[0][0]=-1;
                tr_inp[0][1]=-1;
                tr_inp[1][0]=-1;
                tr_inp[1][1]=1;
                tr_inp[2][0]=1;
                tr_inp[2][1]=-1;
                tr_inp[3][0]=1;
                tr_inp[3][1]=1;

                tr_out[0][0]=0;
                tr_out[1][0]=1;
                tr_out[2][0]=1;
                tr_out[3][0]=0;

                double epoca=0;       // Contador de �pocas

                epoca=Nepocas[Nepo];
                erro_global_medio=(double*)malloc(epoca*sizeof(double));
                rna.carregarPesos("wbs.txt");
                float perc=0, max=epoca-1;
                for(int i=0;i<epoca;i++){
                    perc=(i*100.0)/max;
                    system ("cls");
                    cout<<perc<<"% concluido (treino).\n";
                    double erro_interacao;
                    double media_erro_interacao=0;
                    for(int j=0;j<nPadroes; j++){ // 4 vezes
                        rna.Train(tr_inp[j],tr_out[j],&erro_interacao); // Treina rede com: Entradas e saidas desejadas
                        media_erro_interacao += erro_interacao;
                    }
                    media_erro_interacao/=nPadroes;
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

            }

        }

    }

    return 1;
}
