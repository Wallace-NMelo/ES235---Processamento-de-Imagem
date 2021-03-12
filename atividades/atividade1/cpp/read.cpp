#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>


using namespace cv;
using namespace std;

bool equal (Mat Image1, Mat Image2 ) {

    Mat D;
    absdiff(Image1, Image2, D);
    Scalar s = sum(D);
    return s == Scalar::all(0);
}


int main()
{
    // Load images
    Mat  R_grey = imread("R.png", IMREAD_GRAYSCALE);
    Mat  G_grey = imread("G.png", IMREAD_GRAYSCALE);
    Mat  B_grey = imread("B.png", IMREAD_GRAYSCALE);
    Mat RGB_orig = imread("RGB.png", IMREAD_COLOR);
    Mat RGB_gen, Diff;    
    
    //Merge Channels
    vector <Mat> RGB_ch = {R_grey,G_grey,B_grey};
    merge(RGB_ch, RGB_gen);
    
    //Comparing Images
    assert(equal(RGB_orig, RGB_gen));
    
    //Saving Image 
    imwrite("RGB_answer.png",RGB_gen);
    imshow("RGB_Original", RGB_orig);
    waitKey(0);
    imshow("RGB_Generated", RGB_gen);
    waitKey(0);
    

}