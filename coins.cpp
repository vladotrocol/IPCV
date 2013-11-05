#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

typedef struct{
	Mat* dX;
	Mat* dY;
	Mat* m;
	Mat* phi;
	Mat* h;
}sob;

int t = 55;
int minR = 4;
int maxR = 30;
Mat* coins1;
//Values for x-derivative kernel
float fkX[3][3] = {
					{-1,0,1},
			 		{-2,0,2},
			 		{-1,0,1}
			 	  };

//Values for y-derivative kernel
float fkY[3][3] = { 
					{-1,-2,-1},
			 		{0,0,0},
			 		{1,2,1}
			 	  };

sob* R;

//Padds the matrix with borders
//        1|123|3
//         ----- 
// 123    1|123|3
// 456 -> 4|456|6
// 789    7|789|9
// 		   -----       
//        7|789|9

Mat* padd_border_uchar(Mat* src){
	
	//The new size is old+borders(2 for x, 2 for y)	
	unsigned int h = src->rows+2; //unsigned int: 0 to 65535
	unsigned int w = src->cols+2;

	//init the resulting matrix with 0's floats
	Mat* r = new Mat( Mat::zeros(h, w, CV_8U) ); //CV_32F (32-bit floats)

	//Copy the corners of the image
	r->at<uchar>(0,0) = src->at<uchar>(0, 0); //north-west
	r->at<uchar>(0,w-1) = src->at<uchar>(0, w-3); //north-east
	r->at<uchar>(h-1,0) = src->at<uchar>(h-3, 0); //south-west
	r->at<uchar>(h-1,w-1) = src->at<uchar>(h-3, w-3); //south-east

	//Horizontal borders
	for(unsigned int it = w-2; it>0; it--){
		r->at<uchar>(0, it) = src->at<uchar>(0, it-1); //north border
		r->at<uchar>(h-1, it) = src->at<uchar>(h-3, it-1); //south border
	}

	//Vertical borders
	for(unsigned int it = h-2; it>0; it--){
		r->at<uchar>(it, 0) = src->at<uchar>(it-1, 0); //east border
		r->at<uchar>(it, w-1) = src->at<uchar>(it-1, w-3); //west border
	}

	//Fill centre of matrix
	for(unsigned int i = h-2; i>0; i--){
		for(unsigned int j = w-2; j>0; j--){
			r->at<uchar>(i,j) = src->at<uchar>(i-1,j-1);
		}
	}

	return r;
};


//Shift all values in matrix by min
//such that all values are positive
Mat* shift(Mat* src, int min){
	if(min<0){
		for(int i=0;i<src->rows;i++){
			for(int j=0;j<src->cols;j++){
				if(min>0){
					src->at<uchar>(i, j) -= min;
				}
			}
		}
	}
	return src;
};


//Set the range to 0-255
Mat* normalize(Mat* src, int max, int min){
	
	//compensate for the shift
	if(min<0){
		max-=min;
	}

	//Initialise result
	Mat* r = new Mat( Mat::zeros(src->rows, src->cols, CV_8U) );

	for(int i=0;i<src->rows;i++){
		for(int j=0;j<src->cols;j++){
			if(max>0){ //check fot /0
				r->at<uchar>(i, j) = (uchar)( ( (float)src->at<uchar>(i,j)) / (float)(max) * 255.0f);
			}
		}
	}
	return r;
};

//Convolve a matrix with a kernel
Mat* convolution(Mat* src, Mat* ker){

	//Padding the border
	Mat* psrc = padd_border_uchar(src);
	
	//Height and width after padding
	unsigned int h = psrc->rows;
	unsigned int w = psrc->cols;

	//Kernel radius (used as offsets for parsing image block by block)
	unsigned int rkh = (unsigned int)((ker->rows-1)*0.5f); //horizontal
	unsigned int rkw = (unsigned int)((ker->cols-1)*0.5f); //vertical

	//Resulting matrix
	Mat* r = new Mat( Mat::zeros(h, w, CV_8U) );
	
	int sum; //Total convolved sum for each pixel
	
	//needed for shifting and normalizing
	int max = -32766; 
	int min = 32766;

	//for all values in source
	for(unsigned int i=1;i<h-rkh-1;i++){
		for(unsigned int j=1;j<w-rkw-1;j++){
			sum=0.0; //reset for each pixels/block
			//each kernel-sized block
			for(unsigned int k=i-rkh;k<i+rkh+1;k++){
				for(unsigned int l=j-rkw;l<j+rkw+1;l++){
					sum += (float)psrc->at<uchar>(k, l)*ker->at<float>(k-i+rkh, l-j+rkw);
				}
			}
			//Compute final pixel value
			r->at<uchar>(i,j) = sum;
			
			//find min&max
			if(sum>max){
				max = sum;
			}
			else if(sum<min){
				min = sum;
			}
		}
	}
	//Return the normalized result
	return normalize(shift(r, min), max, min);
};

Mat* magnitude(Mat* dX, Mat* dY){
	Mat* r = new Mat( Mat::zeros(dX->rows, dX->cols, CV_8U) );
	for(unsigned int i=dX->rows-1; i>0;i--){
		for(unsigned j=dX->cols-1; j>0;j--){
			r->at<uchar>(i,j) = sqrt(dX->at<uchar>(i,j)*dX->at<uchar>(i,j)+
									dY->at<uchar>(i,j)*dY->at<uchar>(i,j));
		}
	}
	return r;
};

Mat* angle(Mat* dX, Mat* dY){
	Mat* r = new Mat( Mat::zeros(dX->rows, dX->cols, CV_8U) );
	for(unsigned int i=dX->rows-1; i>0;i--){
		for(unsigned j=dX->cols-1; j>0;j--){
			if(dX->at<uchar>(i,j)!=0){
				r->at<uchar>(i,j) = atan(dY->at<uchar>(i,j)/dX->at<uchar>(i,j));
			}
		}
	}
	return r;
};

sob* sobel(Mat* src){
	//Initialise kernels
	Mat* kX = new Mat(3, 3, CV_32F, &fkX);
	Mat* kY = new Mat(3, 3, CV_32F, &fkY);
	
	//Apply convolutions

	sob* R = new sob;

	R->dX = convolution(src, kX);
	R->dY = convolution(src, kY);
	R->m = magnitude(R->dX, R->dY);
	R->phi = angle(R->dX, R->dY);

	return R;
};

void apply_threshold(Mat* m){
	for(unsigned int i=m->rows-1;i>0;i--){
		for(unsigned int j=m->cols-1;j>0;j--){
			if(m->at<uchar>(i,j) > t){
				m->at<uchar>(i,j) = 255;
			}
			else{
				m->at<uchar>(i,j) = 0;	
			}
		}
	}
};


void hough() {
	float x,y,dx,dy;
	int x1,y1,x2,y2;
	int h= R->m->rows, w= R->m->cols;
	R->h = new Mat( Mat::zeros(h, w, CV_8U) );
	for (int i = 0; i < h;i++) {
		for (int j = 0; j <w;j++) {
			if (R->m->at<uchar>(i, j) > 0) {
				for(int theta=0; theta<360;theta++){
					x = i-100*cos(theta*M_PI/180);
					y = j-100*sin(theta*M_PI/180);
					if(x<w&&x>0&&y<h&&y>0){
						R->h->at<uchar>(i, j)+=20;
					}
				}
			}
		}
	}
}

void show(sob* R){
	//imshow("Coins dX", *R->dX);
	//imshow("Coins dY", *R->dY);
	imshow("Coins magnitude", *R->m);
	imshow("Coins hough", *R->h);

	imshow("Coins angle", *R->phi);
};

void draw( int, void* )
{
	R = sobel(coins1);
	apply_threshold(R->m);
	hough();
	show(R);
	delete R;
}




int main( int argc, char ** argv )
{
	coins1 = new Mat(imread("./images/coins1.png", CV_LOAD_IMAGE_GRAYSCALE));
	
	R = sobel(coins1);
 	namedWindow("Linear Blend", 1);
	createTrackbar( "Threshold", "Linear Blend", &t, 255, draw );
	createTrackbar( "Min Radius", "Linear Blend", &minR, 600, draw );
	createTrackbar( "Max Radius", "Linear Blend", &maxR, 600, draw );

	 //Display results
	//namedWindow("Coins dX", CV_WINDOW_AUTOSIZE);
	//namedWindow("Coins dY", CV_WINDOW_AUTOSIZE);
	namedWindow("Coins magnitude", CV_WINDOW_AUTOSIZE);
	namedWindow("Coins hough", CV_WINDOW_AUTOSIZE);
	namedWindow("Coins angle", CV_WINDOW_AUTOSIZE);
	draw(t, 0);
	waitKey();
	//cleanup
	delete coins1;


	return 0;
};

