#ifndef NEURON_H
#define NEURON_H

#include "dendrite.h"

class Neuron  {
      public:
         unsigned long n_ID; // ID de um determinado neuronio na rede
                             // Usado para encontrar um determinado neurônio em uma matriz
         double n_value; // Valor que o neuronio mantem atualmente
         double n_bias;  // Bias do neuronio
         double n_delta; // Usado no backprop.

		 double n_deltaPrev; // Usado no RPROP
		 double n_deltaPrevBias; // Usado no RPROP
		 double gradient; // Usado no RPROP
		 double gradientPrev; // Usado no RPROP
		 double gradientBias; // Usado no RPROP
		 double gradientPrevBias; // Usado no RPROP
		 double change;
		 double changeBias;

         Dendrite *Dendrites; // Dendritos
         Neuron(unsigned long ID=0,double value=0.0,double bias=0.0); // Contrutor e valores iniciais
		 ~Neuron();
         void SetDendrites(unsigned long dendrite);
};

#endif // NEURON_H
