#include "neuron.h"

Neuron::Neuron(unsigned long ID,double value,double bias){
    n_ID=ID;
    n_value=value;
    n_bias=bias;
    n_delta=0.0;
}

void Neuron::SetDendrites(unsigned long dendrite){ // Define os dentritos do neuronio a aprtir de dendrite
    Dendrites=new Dendrite[dendrite];
    for(int i=0;i<dendrite;i++){
        Dendrites[i].d_points_to=i; // Inicializar o dendrito para apontar para a próxima camada
    }
}

Neuron::~Neuron(){
    //delete Dendrites;
}