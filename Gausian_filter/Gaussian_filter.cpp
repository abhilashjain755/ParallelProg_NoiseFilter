#include <iostream>
#include <cmath>
#include <iomanip>
#include <time.h>
#include <unistd.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<omp.h>

using namespace std;
using namespace cv;


void createFilter(double gKernel[5][5],double sigma)
{
    // set standard deviation sigma
    double r, s = 2.0 * sigma * sigma;

    // sum is for normalization
    double sum = 0.0;

    // generate 5x5 kernel
    #pragma omp parallel for reduction(+ : sum) private(y) num_threads(5)
    for (int x = -2; x <= 2; x++)
    {
        for(int y = -2; y <= 2; y++)
        {
            r = sqrt(x*x + y*y);
            gKernel[x + 2][y + 2] = (exp(-(r*r)/s))/(M_PI * s);
            sum += gKernel[x + 2][y + 2];
        }
    }

    // normalize the Kernel
     #pragma omp parallel for shared(sum) private(j) num_threads(5)
    for(int i = 0; i < 5; ++i)
        for(int j = 0; j < 5; ++j)
            gKernel[i][j] /= sum;

}

//imwrite( "../../images/Gray_Image.jpg", gray_image );

int main(int argc, char** argv )
{
    Mat src, dst;
    double gKernel[5][5];
      float sum;

      /// Load an image
      src = imread( argv[1],CV_LOAD_IMAGE_GRAYSCALE );

      if( !src.data )
      { return -1; }
      double b = atof(argv[2]);

      // define the kernel
      createFilter(gKernel,b);

         dst = src.clone();
        #pragma omp parallel for private(x) num_threads(src.rows)
        for(int y = 0; y < src.rows; y++)
            for(int x = 0; x < src.cols; x++)
                dst.at<uchar>(y,x) = 0.0;

        //convolution operation
        #pragma omp parallel for collapse(3) num_threads(src.rows)
        for(int y = 1; y < src.rows - 1; y++){
            for(int x = 1; x < src.cols - 1; x++){
                sum = 0.0;
                for(int k = -1; k <= 1;k++){
                    for(int j = -1; j <=1; j++){
                        sum = sum + gKernel[j+1][k+1]*src.at<uchar>(y - j, x - k);
                    }
                }
                dst.at<uchar>(y,x) = sum;
            }
        }
       namedWindow("final");
      imshow("final", dst);

//        namedWindow("initial");
  //      imshow("initial", src);

      waitKey();

return 0;
    
}

