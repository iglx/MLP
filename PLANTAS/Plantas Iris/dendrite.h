#ifndef DENDRITE_H
#define DENDRITE_H

class Dendrite {
    public:
        double d_weight; // Peso do neuronio
        unsigned long d_points_to; // Indice do neurónio da proxima camada (para qual aponta)
        Dendrite(double weight=0.0, unsigned long points_to=0);
		~Dendrite();
};

#endif // DENDRITE_H
