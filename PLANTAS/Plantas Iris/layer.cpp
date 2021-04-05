#include "layer.h"

Layer::Layer()
{
}

/*Layer::Layer(unsigned long size=1){    //size é o numero de neuronios na camada
    Neurons = new Neuron [size];
}  */

Layer::~Layer(){ //destrutor: libera a memória reservadas para o vetor de neuronios
    //delete Neurons;
}

void Layer::Initialize(unsigned long size) {   // Inicializa a camada
    Neurons = new Neuron [size];
}

Neuron Layer::GetNeuron(unsigned long index){  // Retorna o neuronio de ID = index
    return Neurons[index];
}

void Layer::SetNeuron(Neuron neuron,unsigned long index){ // Cria o neuronio
    Neurons[index]=neuron;
}
