#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
   double r_vec[3] = {-0.155, -0.344, 0.021};
   double R_matrix[9];
   CvMat pr_vec;
   CvMat pR_matrix;

   cvInitMatHeader(&pr_vec,1,3,CV_64FC1,r_vec,CV_AUTOSTEP);
   cvInitMatHeader(&pR_matrix,3,3,CV_64FC1,R_matrix,CV_AUTOSTEP);
   cvRodrigues2(&pr_vec, &pR_matrix,0);

   for(int i = 0; i < 9; i++)
   {
       cout<<R_matrix[i]<<endl;
   }
}
