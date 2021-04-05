#include "opencv/cv.hpp"
#include "opencv/highgui.h"
#include <iostream>
#include "imagem.h"

using namespace cv;
using namespace std;

int separaImg(Mat src){

    Mat dst;
    dst = imread("C:\\Users\\igorl\\Music\\imagens\\recebe00.jpg",0);

    for(int i=0; i<src.rows; i++){
        for(int j=0; j<src.cols; j++){
            dst.at<uchar>(i,j) = src.at<uchar>(i,j);
        }
    }

    //criando matrzo 25x100
    int **matrizEnt = (int**) malloc( 25 * sizeof (int*));
    for (int i=0;i<25;++i){
        matrizEnt[i] = (int*) malloc( 100 * sizeof (int));
    }

    int cont1= 0;
    int cont2 = 0;
    for(int x=0; x<dst.rows; x=x+10){
        for(int y=0; y<dst.cols; y=y+10){
            cont1 = 0;
            for(int i=x; i<x+10; i++){
                for(int j=y; j<y+10; j++){
                    matrizEnt[cont2][cont1]=dst.at<uchar>(i,j);
                    cont1++;
                }
            }
            cont2++;
        }
    }

//    for(int i=0;i<25;i++){
//        for(int j=0;j<100;j++){
//            printf("Elemento [%d][%d]: ",i+1, j+1);
//            printf("%d\n",matrizEnt[i][j]);
//        }
//    }


    namedWindow("teste");
    imshow("teste",dst);

    namedWindow("teste2");
    imshow("teste2",src);

    waitKey(0);
    return **matrizEnt;
}
