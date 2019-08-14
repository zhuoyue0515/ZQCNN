#include "ZQ_CNN_Net.h"
#include "ZQ_CNN_Net_NCHWC.h"
#include <vector>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "ZQ_CNN_CompileConfig.h"
#if ZQ_CNN_USE_BLAS_GEMM
#if __ARM_NEON
#include <openblas/cblas.h>
#else
#include <openblas/cblas.h>
#pragma comment(lib,"libopenblas.lib")
#endif
#elif ZQ_CNN_USE_MKL_GEMM
#include <mkl/mkl.h>
#pragma comment(lib,"mklml.lib")
#else
#pragma comment(lib,"ZQ_GEMM.lib")
#endif

using namespace ZQ;
using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	if(argc < 2){
        fprintf(stderr, "usage: add image path \n");
        return 0;
        }

	int num_threads = 1;

#if ZQ_CNN_USE_BLAS_GEMM
	openblas_set_num_threads(num_threads);
#elif ZQ_CNN_USE_MKL_GEMM
	mkl_set_num_threads(num_threads);
#endif
	
	ZQ_CNN_Net net; 
#if __ARM_NEON
	ZQ_CNN_Net_NCHWC<ZQ_CNN_Tensor4D_NCHWC4> net2;
#else
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	ZQ_CNN_Net_NCHWC<ZQ_CNN_Tensor4D_NCHWC8> net2;
#elif ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	ZQ_CNN_Net_NCHWC<ZQ_CNN_Tensor4D_NCHWC4> net2;
#else
	ZQ_CNN_Net_NCHWC<ZQ_CNN_Tensor4D_NCHWC1> net2;
#endif
#endif
	
#if defined(_WIN32)
	if (!net.LoadFrom("model/det5-dw96-v2s.zqparams", "model/det5-dw96-v2s.nchwbin",true,1e-9,true)
		//|| !net2.LoadFrom("model/det3.zqparams", "model/det3_bgr.nchwbin", true, 1e-9))
		|| !net2.LoadFrom("model/det5-dw96-v2s.zqparams", "model/det5-dw96-v2s.nchwbin", true, 1e-9, true))
#else
	if (!net.LoadFrom("/home/ubuntu/ff-ML/face_API/ZQCNN/model/det5-dw112.zqparams", "/home/ubuntu/ff-ML/face_API/ZQCNN/model/det5-dw112.nchwbin", true, 1e-9, true))
#endif
	{
		cout << "failed to load model\n";
		return EXIT_FAILURE;
	}
	int net_H, net_W, net_C;
	net.GetInputDim(net_C, net_H, net_W);
	Mat img = imread(argv[1], 1);
	int show_H = img.rows, show_W = img.cols;
	if (img.empty())
	{
		cout << "failed to load image\n";
		return EXIT_FAILURE;
	}
	if (img.channels() == 1)
		cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
	Mat img1;
	cv::resize(img, img1, cv::Size(net_W, net_H));
	Mat draw_img;
	cv::resize(img, draw_img, cv::Size(show_W, show_H));

	ZQ_CNN_Tensor4D_NHW_C_Align128bit input;

	input.ConvertFromBGR(img1.data, img1.cols, img1.rows, img1.step[0]);
	net.Forward(input);
	const ZQ_CNN_Tensor4D* landmark = net.GetBlobByName("conv6-3");

	if (landmark == 0)
	{
		cout << "failed to get blob conv6-3\n";
		return EXIT_FAILURE;
	}
	const float* landmark_data = landmark->GetFirstPixelPtr();
	for (int i = 0; i < 106; i++)
	{
		char buf[10];
#if defined(_WIN32)
		sprintf_s(buf,10, "%d", i);
#else
		sprintf(buf, "%d", i);
#endif
		cv::Point pt = cv::Point(show_W * landmark_data[i * 2], show_H * landmark_data[i * 2 + 1]);

                cout<<pt.x<<","<<pt.y<<endl;
		cv::circle(draw_img, pt, 1, cv::Scalar(240, 0, 0), 2);
	}
	cout<<show_H<<","<<show_W<<endl;
	//namedWindow("landmark");
	//imshow("landmark", draw_img);
	cv::imwrite("/home/ubuntu/ff-ML/face_API/deploy/FF_Face_API/users_photo/landmark.jpg", draw_img);
	//waitKey(0);
	return EXIT_SUCCESS;
}
