#include "mlp.h"
#include <algorithm>
#include <iostream>


// retorna um numero real entre -1.0f e 1.0f
#define RANDOM_CLAMP    (((float)rand()-(float)rand())/RAND_MAX)

//retorna um numero real entre 0 & 1
#define RANDOM_NUM		((float)rand()/(RAND_MAX+1))


MLP::MLP()
{
}

template <class T> const T& sinal (const T& a){
	if (a<0)
		return -1;
	else if (a>0)
		return 1;
	else return 0;
}

int MLP::SetData(double learning_rate,unsigned long layers[],unsigned long tot_layers) { //Função para definir vários parâmetros da rede
    if (tot_layers<2) return(-1); // Retorna um erro o numero de camadas é < 2 (obs.: a primeira camada é de entradas)
                                  // Pelo menos uma camada de entrada e uma de saída são necessárias
    net_learning_rate=learning_rate;
    net_layers= new unsigned long [tot_layers]; // inicializa o vetor de camadas
    Layers=new Layer[tot_layers];
    for(int i=0;i<tot_layers;i++){
        net_layers[i]=layers[i];
        Layers[i].Initialize(layers[i]); // Inicializar cada camada com o tamanho especificado
    }

    net_inputs=new double[layers[0]];
    net_outputs=new double[layers[tot_layers-1]];
    net_tot_layers=tot_layers;
    return 0;
}

int MLP::SetDataRPROP(unsigned long layers[],unsigned long tot_layers) {

	increaseFactor = 1.2; // eta_mais para o RPROP
	decreaseFactor = 0.5; // eta_menos para o RPROP
	maxDelta = 10; // para o RPROP
	minDelta = 1E-6; // para o RPROP

	if (tot_layers<2) return(-1); // Retorna um erro o numero de camadas é < 2 (obs.: a primeira camada é de entradas)
                                  // Pelo menos uma camada de entrada e uma de saída são necessárias
	net_layers= new unsigned long [tot_layers]; // inicializa o vetor de camadas
    Layers=new Layer[tot_layers];
    for(int i=0;i<tot_layers;i++){
        net_layers[i]=layers[i];
        Layers[i].Initialize(layers[i]); // Inicializar cada camada com o tamanho especificado
    }

    net_inputs=new double[layers[0]];
    net_outputs=new double[layers[tot_layers-1]];
    net_tot_layers=tot_layers;

	for(int i=0;i<net_tot_layers;i++){
        for(int j=0;j<net_layers[i];j++){
            if(i!=(net_tot_layers-1)){
				Layers[i].Neurons[j].n_deltaPrev = 0; // Usado no RPROP
				Layers[i].Neurons[j].gradient = 0; // Usado no RPROP
				Layers[i].Neurons[j].gradientPrev = 0; // Usado no RPROP
				Layers[i].Neurons[j].change = 0;
			}
			if(i!=0){ // A primeira camada não possui bias
				Layers[i].Neurons[j].n_deltaPrevBias = 0; // Usado no RPROP
				Layers[i].Neurons[j].gradientBias = 0; // Usado no RPROP
				Layers[i].Neurons[j].gradientPrevBias = 0; // Usado no RPROP
				Layers[i].Neurons[j].changeBias = 0;
			}
		}
	}
    return 0;
}

int MLP::SetInputs(double inputs[]){ //Função para setar as entradas
    for(int i=0;i<net_layers[0];i++){
        Layers[0].Neurons[i].n_value=inputs[i]; // As saidas da camada de entradas são iguais a entradas para o treinamento
    }                                           // Não existe processamento na camada 0
    return 0;
}

void MLP::RandomizeWB(void){ // Gera pesos aleatorios para pesos e bias
    int i,j,k;
    for(i=0;i<net_tot_layers;i++){
        for(j=0;j<net_layers[i];j++){
            if(i!=(net_tot_layers-1)){ // A ultima camada não possui pesos
                Layers[i].Neurons[j].SetDendrites(net_layers[i+1]); //Inicializa os denditos
                for(k=0;k<net_layers[i+1];k++){
                    Layers[i].Neurons[j].Dendrites[k].d_weight=GetRand(); // Atribui um numero aleatorio
                }
            }
            if(i!=0){ // A primeira camada não possui bias
                Layers[i].Neurons[j].n_bias=GetRand();
            }
        }
    }
}

double* MLP::GetOutput(void){ // Retorna a saida da rede
    double *outputs;
    int i,j,k;

    outputs=new double[net_layers[net_tot_layers-1]]; // Vetor temporario de saidas

    for(i=1;i<net_tot_layers;i++){
        for(j=0;j<net_layers[i];j++){
            Layers[i].Neurons[j].n_value=0;
            for(k=0;k<net_layers[i-1];k++){
                // Processamento do neuronio:
                Layers[i].Neurons[j].n_value=Layers[i].Neurons[j].n_value+Layers[i-1].Neurons[k].n_value*Layers[i-1].Neurons[k].Dendrites[j].d_weight;
            }
            Layers[i].Neurons[j].n_value=Layers[i].Neurons[j].n_value+Layers[i].Neurons[j].n_bias; // Adiciona bias
            Layers[i].Neurons[j].n_value=Limiter(Layers[i].Neurons[j].n_value);  // Função de ativação
        }
    }

    for(i=0;i<net_layers[net_tot_layers-1];i++){
        outputs[i]=Layers[net_tot_layers-1].Neurons[i].n_value;
    }
    return outputs; //Returna as saidas
}

void MLP::Update(void){ // Não passe de uma função fictícia
    //double *temp; // ponteiro temporario
    //temp=GetOutput();
    GetOutput(); // Força um novo calculo de saidas
    //delete temp;
}

/*void MLP::SetOutputs(double outputs[]){ //Define os valores da camada de saída
    for(int i=0;i<net_layers[net_tot_layers-1];i++){
        Layers[net_tot_layers-1].Neurons[i].n_value=outputs[i]; //Seta o valor
    }
}  */

double MLP::Limiter(double value){ // Função de ativação
    //return tanh(value);   // Tangente hiperbolica
    return (1.0/(1+exp(-value))); // Sigmoid
}

double MLP::GetRand(void){  //Retorna um número aleatório entre faixa de -1 a 1 usando o tempo para alimentar a função srand
    time_t timer;
    struct tm *tblock;
    timer=time(NULL);
    tblock=localtime(&timer);
    int seed=int(tblock->tm_sec+100*RANDOM_CLAMP+100*RANDOM_NUM);
    //srand(tblock->tm_sec);
    srand(seed);
    return (RANDOM_CLAMP+RANDOM_NUM);
}

double MLP::SigmaWeightDelta(unsigned long layer_no, unsigned long neuron_no){ //Calcula a soma de pesos*delta.
    double result=0.0;                                                         // Usado no backprop. layer_no é o numero da camada.
                                                                               // O número de camadas e número de neurônios pode ser zero
    for(int i=0;i<net_layers[layer_no+1];i++) { // Verre todos os neuronios da camada seguinte
        result=result+Layers[layer_no].Neurons[neuron_no].Dendrites[i].d_weight*Layers[layer_no+1].Neurons[i].n_delta; //Computa o somatorio
    }
    return result;
} // neuron_no é o número do neurônio. Esta função é usada no back-prop

int MLP::TrainMomentum(double inputs[],double outputs[], double a){ // O Algoritmo de aprendizado Backprop padrão
    int i,j,k;
    double Target, Actual, Delta, erro;
    SetInputs(inputs); // Seta as entradas
    Update(); // Atualiza os valores
    //SetOutputs(outputs); // seta as saidas
    ErroMedioIns=0;
    for(i=net_tot_layers-1;i>0;i--){ // Varre a rede da ultima camada até a primeira
        for(j=0;j<net_layers[i];j++) { // Varre cada neuronio da camada atual
            if(i==net_tot_layers-1){ // Camada de saida, necessita de atenção especial
                Target=outputs[j]; // Valor de saida desejado
                Actual=Layers[i].Neurons[j].n_value; // Valor atual
                erro = (Target - Actual);

                ErroMedioIns += (erro*erro)/2;
                Delta= erro * Actual * (1 - Actual); // erro * deriava da funçao de ativação (sigmoide)
                //Delta= (Target - Actual) * Actual * (1 - Actual); // erro * deriava da funçao de ativação (sigmoide)
                Layers[i].Neurons[j].n_delta=Delta; // Atualiza o delta
                for(k=0;k<net_layers[i-1];k++) {
                    Layers[i-1].Neurons[k].Dendrites[j].d_weight = a * Layers[i-1].Neurons[k].Dendrites[j].d_weight + Delta*net_learning_rate*Layers[i-1].Neurons[k].n_value; //Calcula novos pesos
                }

                Layers[i].Neurons[j].n_bias = Layers[i].Neurons[j].n_bias + Delta*net_learning_rate*1; //n_value é sempre 1 para o bias
            }
            else { // Camadas escondidas
                // valor de saida desejado:
                Actual=Layers[i].Neurons[j].n_value; // valor atual
                Delta=  Actual * (1 - Actual)* SigmaWeightDelta(i,j); // deriava da funçao de ativação (sigmoide) * (w*D da camada anterior) - ver: backprog
                for(k=0;k<net_layers[i-1];k++){
                    Layers[i-1].Neurons[k].Dendrites[j].d_weight += Delta*net_learning_rate*Layers[i-1].Neurons[k].n_value; //Calcula os novos pesos
                }
                if(i!=0) //Input layer does not have a bias
                    Layers[i].Neurons[j].n_bias = Layers[i].Neurons[j].n_bias + Delta*net_learning_rate*1; //n_value é sempre 1 para o bias
            }
        }
    }
    return 0;
}



int MLP::Train(double inputs[], double outputs[], double *erro_interacao){ // O Algoritmo de aprendizado Backprop padrão
    int i,j,k;
    int int_epoca=0, int_global=0;
    //double *historico_erro = (double*)malloc(num_epoca*sizeof(double));
    double Target, Actual, Delta, erro;
    SetInputs(inputs); // Seta as entradas
    Update(); // Atualiza os valores
    //SetOutputs(outputs); // seta as saidas
    ErroMedioIns=0;
    for(i=net_tot_layers-1;i>0;i--){ // Varre a rede da ultima camada até a primeira
        for(j=0;j<net_layers[i];j++) { // Varre cada neuronio da camada atual
            if(i==net_tot_layers-1){ // Camada de saida, necessita de atenção especial
                Target=outputs[j]; // Valor de saida desejado
                Actual=Layers[i].Neurons[j].n_value; // Valor atual
				erro = (Target - Actual);
                //printf("%lf\n e %d",erro, j);
               // system("pause");
				ErroMedioIns += (erro*erro)/2;
				Delta= erro * Actual * (1 - Actual); // erro * deriava da funçao de ativação (sigmoide)
                //Delta= (Target - Actual) * Actual * (1 - Actual); // erro * deriava da funçao de ativação (sigmoide)
                Layers[i].Neurons[j].n_delta=Delta; // Atualiza o delta
				for(k=0;k<net_layers[i-1];k++) {
                    Layers[i-1].Neurons[k].Dendrites[j].d_weight +=Delta*net_learning_rate*Layers[i-1].Neurons[k].n_value; //Calcula novos pesos
                }
                Layers[i].Neurons[j].n_bias = Layers[i].Neurons[j].n_bias + Delta*net_learning_rate*1; //n_value é sempre 1 para o bias
                *erro_interacao = erro;
            }


            else { // Camadas escondidas
                // valor de saida desejado:
                Actual=Layers[i].Neurons[j].n_value; // valor atual
                Delta=  Actual * (1 - Actual)* SigmaWeightDelta(i,j); // deriava da funçao de ativação (sigmoide) * (w*D da camada anterior) - ver: backprog
                for(k=0;k<net_layers[i-1];k++){
                    Layers[i-1].Neurons[k].Dendrites[j].d_weight += Delta*net_learning_rate*Layers[i-1].Neurons[k].n_value; //Calcula os novos pesos
                }
                if(i!=0) //Input layer does not have a bias
                    Layers[i].Neurons[j].n_bias = Layers[i].Neurons[j].n_bias + Delta*net_learning_rate*1; //n_value é sempre 1 para o bias
            }
        }
        //segundo modo
        /*erro_epoca /=int_epoca;
        erro_global+= erro_epoca;
        int_global++;*/
    }
    //*erro_interacao = ErroMedioIns;
   // erro_global/=int_global;
    return 0;
}

int MLP::TrainRPROP(double inputs[],double outputs[], int maxEpochs) {
	
	double Target, Actual, Delta, erro;
	int i,j,k;
	int epoch = 0;

	SetInputs(inputs); // Seta as entradas
    Update(); // Atualiza os valores
	
	while (epoch < maxEpochs)
	{
	  ++epoch;
	  if (epoch % 100 == 0 && epoch != maxEpochs){
		  for(i=net_tot_layers-1;i>0;i--){ // Varre a rede da ultima camada até a primeira
			for(j=0;j<net_layers[i];j++) { // Varre cada neuronio da camada atual
				if(i==net_tot_layers-1){ // Camada de saida, necessita de atenção especial
					Target=outputs[j]; // Valor de saida desejado
					Actual=Layers[i].Neurons[j].n_value; // Valor atual
					erro = (Target - Actual);
					ErroMedioIns += (erro*erro)/2;
					Delta= erro * Actual * (1 - Actual); // erro * deriava da funçao de ativação (sigmoide)
					Layers[i].Neurons[j].n_delta=Delta; // Atualiza o delta (delta = derivative)

					Layers[i].Neurons[j].gradient = Layers[i].Neurons[j].n_value * Delta;

					if ( (Layers[i].Neurons[j].gradientPrev * Layers[i].Neurons[j].gradient) > 0) {
						float delta = std::min((float)(Layers[i].Neurons[j].n_deltaPrev * increaseFactor), maxDelta);

						Layers[i].Neurons[j].change = (-sinal(Layers[i].Neurons[j].gradient)) * delta;

						for(k=0;k<net_layers[i-1];k++) {
							Layers[i-1].Neurons[k].Dendrites[j].d_weight += Layers[i].Neurons[j].change; //Calcula novos pesos
						}

						Layers[i].Neurons[j].gradientPrev = Layers[i].Neurons[j].gradient;
						Layers[i].Neurons[j].n_deltaPrev = delta;
					}

					if ( (Layers[i].Neurons[j].gradientPrev * Layers[i].Neurons[j].gradient) < 0) {
						float delta = std::max((float)(Layers[i].Neurons[j].n_deltaPrev * decreaseFactor), minDelta);

						for(k=0;k<net_layers[i-1];k++) {
							Layers[i-1].Neurons[k].Dendrites[j].d_weight += -Layers[i].Neurons[j].change; //Calcula novos pesos
						}

						Layers[i].Neurons[j].gradientPrev = 0;
						Layers[i].Neurons[j].n_deltaPrev = delta;
					}

					if ( (Layers[i].Neurons[j].gradientPrev * Layers[i].Neurons[j].gradient) == 0) {
						float delta = Layers[i].Neurons[j].n_deltaPrev;
						
						Layers[i].Neurons[j].change = (-sinal(Layers[i].Neurons[j].gradient)) * delta;

						for(k=0;k<net_layers[i-1];k++) {
							Layers[i-1].Neurons[k].Dendrites[j].d_weight += Layers[i].Neurons[j].change; //Calcula novos pesos
						}

						Layers[i].Neurons[j].gradientPrev = Layers[i].Neurons[j].gradient;
						Layers[i].Neurons[j].n_deltaPrev = delta;
					}

					// Ajuste do bias:
					Layers[i].Neurons[j].gradientBias = Layers[i].Neurons[j].n_bias * Delta;

					if ( (Layers[i].Neurons[j].gradientPrevBias * Layers[i].Neurons[j].gradientBias) > 0) {
						float delta = std::min((float)(Layers[i].Neurons[j].n_deltaPrevBias * increaseFactor), maxDelta);

						Layers[i].Neurons[j].changeBias = (-sinal(Layers[i].Neurons[j].gradientBias)) * delta;

						for(k=0;k<net_layers[i-1];k++) {
							Layers[i-1].Neurons[k].n_bias += Layers[i].Neurons[j].changeBias; //Calcula novos pesos
						}

						Layers[i].Neurons[j].gradientPrevBias = Layers[i].Neurons[j].gradientBias;
						Layers[i].Neurons[j].n_deltaPrevBias = delta;
					}

					if ( (Layers[i].Neurons[j].gradientPrevBias * Layers[i].Neurons[j].gradientBias) < 0) {
						float delta = std::max((float)(Layers[i].Neurons[j].n_deltaPrevBias * decreaseFactor), minDelta);

						for(k=0;k<net_layers[i-1];k++) {
							Layers[i-1].Neurons[k].n_bias += -Layers[i].Neurons[j].changeBias; //Calcula novos pesos
						}

						Layers[i].Neurons[j].gradientPrevBias = 0;
						Layers[i].Neurons[j].n_deltaPrevBias = delta;
					}

					if ( (Layers[i].Neurons[j].gradientPrevBias * Layers[i].Neurons[j].gradientBias) == 0) {
						float delta = Layers[i].Neurons[j].n_deltaPrevBias;
						
						Layers[i].Neurons[j].changeBias = (-sinal(Layers[i].Neurons[j].gradient)) * delta;

						for(k=0;k<net_layers[i-1];k++) {
							Layers[i-1].Neurons[k].Dendrites[j].d_weight += Layers[i].Neurons[j].changeBias; //Calcula novos pesos
						}

						Layers[i].Neurons[j].gradientPrevBias = Layers[i].Neurons[j].gradientBias;
						Layers[i].Neurons[j].n_deltaPrevBias = delta;
					}


				}
				else { // Camadas escondidas
					// valor de saida desejado:
					Actual=Layers[i].Neurons[j].n_value; // valor atual
					Delta=  Actual * (1 - Actual)* SigmaWeightDelta(i,j); // deriava da funçao de ativação (sigmoide) * (w*D da camada anterior) - ver: backprog
					for(k=0;k<net_layers[i-1];k++){
						if ( (Layers[i].Neurons[j].gradientPrev * Layers[i].Neurons[j].gradient) > 0) {
							float delta = std::min((float)(Layers[i].Neurons[j].n_deltaPrev * increaseFactor), maxDelta);

							Layers[i].Neurons[j].change = (-sinal(Layers[i].Neurons[j].gradient)) * delta;

							for(k=0;k<net_layers[i-1];k++) {
								Layers[i-1].Neurons[k].Dendrites[j].d_weight += Layers[i].Neurons[j].change; //Calcula novos pesos
							}

							Layers[i].Neurons[j].gradientPrev = Layers[i].Neurons[j].gradient;
							Layers[i].Neurons[j].n_deltaPrev = delta;
						}

						if ( (Layers[i].Neurons[j].gradientPrev * Layers[i].Neurons[j].gradient) < 0) {
							float delta = std::max((float)(Layers[i].Neurons[j].n_deltaPrev * decreaseFactor), minDelta);

							for(k=0;k<net_layers[i-1];k++) {
								Layers[i-1].Neurons[k].Dendrites[j].d_weight += -Layers[i].Neurons[j].change; //Calcula novos pesos
							}

							Layers[i].Neurons[j].gradientPrev = 0;
							Layers[i].Neurons[j].n_deltaPrev = delta;
						}

						if ( (Layers[i].Neurons[j].gradientPrev * Layers[i].Neurons[j].gradient) == 0) {
							float delta = Layers[i].Neurons[j].n_deltaPrev;
						
							Layers[i].Neurons[j].change = (-sinal(Layers[i].Neurons[j].gradient)) * delta;

							for(k=0;k<net_layers[i-1];k++) {
								Layers[i-1].Neurons[k].Dendrites[j].d_weight += Layers[i].Neurons[j].change; //Calcula novos pesos
							}

							Layers[i].Neurons[j].gradientPrev = Layers[i].Neurons[j].gradient;
							Layers[i].Neurons[j].n_deltaPrev = delta;
						}
					}
					if(i!=0){ //Input layer does not have a bias
						Layers[i].Neurons[j].gradientBias = Layers[i].Neurons[j].n_bias * Delta;

						if ( (Layers[i].Neurons[j].gradientPrevBias * Layers[i].Neurons[j].gradientBias) > 0) {
							float delta = std::min((float)(Layers[i].Neurons[j].n_deltaPrevBias * increaseFactor), maxDelta);

							Layers[i].Neurons[j].changeBias = (-sinal(Layers[i].Neurons[j].gradientBias)) * delta;

							for(k=0;k<net_layers[i-1];k++) {
								Layers[i-1].Neurons[k].n_bias += Layers[i].Neurons[j].changeBias; //Calcula novos pesos
							}

							Layers[i].Neurons[j].gradientPrevBias = Layers[i].Neurons[j].gradientBias;
							Layers[i].Neurons[j].n_deltaPrevBias = delta;
						}

						if ( (Layers[i].Neurons[j].gradientPrevBias * Layers[i].Neurons[j].gradientBias) < 0) {
							float delta = std::max((float)(Layers[i].Neurons[j].n_deltaPrevBias * decreaseFactor), minDelta);

							for(k=0;k<net_layers[i-1];k++) {
								Layers[i-1].Neurons[k].n_bias += -Layers[i].Neurons[j].changeBias; //Calcula novos pesos
							}

							Layers[i].Neurons[j].gradientPrevBias = 0;
							Layers[i].Neurons[j].n_deltaPrevBias = delta;
						}

						if ( (Layers[i].Neurons[j].gradientPrevBias * Layers[i].Neurons[j].gradientBias) == 0) {
							float delta = Layers[i].Neurons[j].n_deltaPrevBias;
						
							Layers[i].Neurons[j].changeBias = (-sinal(Layers[i].Neurons[j].gradient)) * delta;

							for(k=0;k<net_layers[i-1];k++) {
								Layers[i-1].Neurons[k].Dendrites[j].d_weight += Layers[i].Neurons[j].changeBias; //Calcula novos pesos
							}

							Layers[i].Neurons[j].gradientPrevBias = Layers[i].Neurons[j].gradientBias;
							Layers[i].Neurons[j].n_deltaPrevBias = delta;
						}
					}
				}
			}
		  }
		}
	  }

	return 0;
}

void MLP::salvarPesos(char* arquivo){ // Salva pesos da rede

    int i,j,k,result;

	FILE *arq;
	arq = fopen(arquivo, "wt");	// Cria um arquivo texto para gravação
	if (arq == NULL)				// Se nào conseguiu criar
	{
		printf("Problemas na CRIACAO do arquivo\n");
		return;
	}

    for(i=1;i<net_tot_layers;i++){
        for(j=0;j<net_layers[i];j++){
            
				if(i==net_tot_layers-1){ // Camada de saida
					for(k=0;k<net_layers[i-1];k++){
						result = fprintf(arq,"%f,", Layers[i-1].Neurons[k].Dendrites[j].d_weight);
					}
					result = fprintf(arq,"%f,", Layers[i].Neurons[j].n_bias);
				}
				else { // Camadas escondidas
					for(k=0;k<net_layers[i-1];k++){
						result = fprintf(arq,"%f,", Layers[i-1].Neurons[k].Dendrites[j].d_weight);
					}
					if(i!=0) //Input layer does not have a bias
						result = fprintf(arq,"%f,", Layers[i].Neurons[j].n_bias);
				}
				if (result == EOF)	
				{
					printf("Erro na Gravacao\n");
					return;
				}
            
        }
    }
	fclose(arq);
}

void MLP::carregarPesos(char *arquivo){ // Salva pesos da rede
	int i,j,k,result;

	FILE *arq;
	arq = fopen(arquivo, "r");	// Abrir um arquivo texto para leitura
	if (arq == NULL)				// Se nào conseguiu abrir
	{
		printf("Problemas na ABERTURA do arquivo\n");
		return;
	}

	float c, v; //c=valor float e v=virgula
	/* OK!
	while (fscanf(arq,"%f%c",&c,&v) != EOF) {
		//fscanf(arq,"%f%c",&c,&v);
		fprintf(stdout,"%f\n",c); 
	}
	*/

	for(i=1;i<net_tot_layers;i++){
        for(j=0;j<net_layers[i];j++){
            
			if(i==net_tot_layers-1){ // Camada de saida
				for(k=0;k<net_layers[i-1];k++){
					fscanf(arq,"%f%c",&c,&v);
					Layers[i-1].Neurons[k].Dendrites[j].d_weight = c;
				}
				fscanf(arq,"%f%c",&c,&v);
				Layers[i].Neurons[j].n_bias =c;
			}
			else { // Camadas escondidas
				for(k=0;k<net_layers[i-1];k++){
					fscanf(arq,"%f%c",&c,&v);
					Layers[i-1].Neurons[k].Dendrites[j].d_weight =c;
				}
				if(i!=0) //Input layer does not have a bias
					fscanf(arq,"%f%c",&c,&v);
					Layers[i].Neurons[j].n_bias =c;
			}
		}
    }

    fclose(arq); 
}

MLP::~MLP(){
    //delete Layers;
}
