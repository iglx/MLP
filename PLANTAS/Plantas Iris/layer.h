#ifndef LAYER_H
#define LAYER_H

#include "neuron.h"

class Layer{
    public:
        Neuron *Neurons; // Ponteiro para vetores de neuronios

        Layer();
        //Layer(unsigned long size=1);
        ~Layer();

        void Initialize(unsigned long size);

        Neuron GetNeuron(unsigned long index);

        void SetNeuron(Neuron neuron,unsigned long index);
};

#endif // LAYER_H
