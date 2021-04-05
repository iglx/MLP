#ifndef MLP_H
#define MLP_H

#include "layer.h"
#include <time.h>
#include <cstdlib> // Para rand()
#include <math.h> // Para tanh()
#include <stdio.h>

class MLP { // RNA
    public:
        double net_learning_rate; // Taxa de aprendizado

		double increaseFactor; // eta_mais para o RPROP
		double decreaseFactor; // eta_menos para o RPROP
		float maxDelta; // para o RPROP
		float minDelta; // para o RPROP

        Layer *Layers; // Ponteiro para o vetor de camadas
        unsigned long net_tot_layers; // Numero de camadas
        double *net_inputs; // Vetor de entradas
        double *net_outputs; // Vetor de saidas desejadas
        unsigned long *net_layers; // Vetor com o numero de neuronios em cada camada
		double ErroMedioIns; //erro m�dio instantaneo
        //double GetRand(void);
        MLP();

        int SetData(double learning_rate,unsigned long layers[],unsigned long tot_layers);

		int SetDataRPROP(unsigned long layers[],unsigned long tot_layers);

        int SetInputs(double inputs[]);

        void RandomizeWB(void);

        double * GetOutput(void);

        void Update(void);

        /*void SetOutputs(double outputs[]);  */

        double Limiter(double value);

        double GetRand(void);

        double SigmaWeightDelta(unsigned long layer_no, unsigned long neuron_no);

    /*Para a camada de saida:
    Delta = (TargetO - ActualO) * ActualO * (1 - ActualO)
    Weight = Weight + LearningRate * Delta * Input

    Para camadas escondidas:
    Delta =  ActualO * (1-ActualO) * Summation(Weight_from_current_to_next AND Delta_of_next)
    Weight = Weight + LearningRate * Delta * Input
    */

        int Train(double inputs[], double outputs[], double *erro_interacao);

		int TrainRPROP(double inputs[],double outputs[], int maxEpochs);

        int TrainMomentum(double inputs[],double outputs[], double a);

		void salvarPesos(char* arquivo);

		void carregarPesos(char* arquivo);

        ~MLP();
};

#endif // MLP_H
