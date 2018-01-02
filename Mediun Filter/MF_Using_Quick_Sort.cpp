#include<iostream>
#include <time.h>
#include <unistd.h>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<omp.h>

using namespace std;
using namespace cv;

//sort the window using quick sort

#include <stdio.h>
#include <stdlib.h>


int partition(int * a, int p, int r)
{
    int lt[r-p];
    int gt[r-p];
    int i;
    int j;
    int key = a[r];
    int lt_n = 0;
    int gt_n = 0;

#pragma omp parallel for
    for(i = p; i < r; i++){
        if(a[i] < a[r]){
            lt[lt_n++] = a[i];
        }else{
            gt[gt_n++] = a[i];
        }   
    }   

    for(i = 0; i < lt_n; i++){
        a[p + i] = lt[i];
    }   

    a[p + lt_n] = key;

    for(j = 0; j < gt_n; j++){
        a[p + lt_n + j + 1] = gt[j];
    }   

    return p + lt_n;
}

void quicksort(int * a, int p, int r)
{
    int div;

    if(p < r){ 
        div = partition(a, p, r); 
#pragma omp parallel sections
        {   
#pragma omp section
            quicksort(a, p, div - 1); 
#pragma omp section
            quicksort(a, div + 1, r); 

        }
    }
}

int main(int argc, char** argv )
{
      Mat src, dst;


	 if ( argc != 2 )
            {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
            }

      // Load an image
      src = imread( argv[1],CV_LOAD_IMAGE_GRAYSCALE );

      int nPixels = src.total();
      int ro=src.rows;
      int co=src.cols;

      if( !src.data )
      { return -1; }

      //create a sliding window of size 9
      int window[9];  
        dst = src.clone();
#pragma omp parallel for private(x) num_threads(src.rows) 
        for(int y = 0; y < src.rows; y++)
            for(int x = 0; x < src.cols; x++)
                dst.at<uchar>(y,x) = 0.0;

#pragma omp parallel for private(x) num_threads(src.rows) 
        for(int y = 1; y < src.rows - 1; y++){
            for(int x = 1; x < src.cols - 1; x++){

                // Pick up window element

                window[0] = src.at<uchar>(y - 1 ,x - 1);
                window[1] = src.at<uchar>(y, x - 1);
                window[2] = src.at<uchar>(y + 1, x - 1);
                window[3] = src.at<uchar>(y - 1, x);
                window[4] = src.at<uchar>(y, x);
                window[5] = src.at<uchar>(y + 1, x);
                window[6] = src.at<uchar>(y - 1, x + 1);
                window[7] = src.at<uchar>(y, x + 1);
                window[8] = src.at<uchar>(y + 1, x + 1);

                // sort the window to find median
                quicksort(window, 0, 8);

                // assign the median to centered element of the matrix
                dst.at<uchar>(y,x) = window[4];
            }
        }
        namedWindow("final");
        imshow("final", dst);

        namedWindow("initial");
        imshow("initial", src);

      waitKey();


    return 0;
}
