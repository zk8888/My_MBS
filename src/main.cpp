#include "../include/MBS.hpp"
#include <iostream>

using namespace std;

void doMBS(const cv::Mat &I, const param pa, cv::Mat &pMap, cv::Mat &dMap);


int
main(){
	cv::VideoCapture cap("../sources/1.FLV");
	cv::Mat frame;
	if (!cap.isOpened())
		return 1;
	double rate = cap.get(CV_CAP_PROP_FPS);
	bool stop(false);
	cv::namedWindow("Source Video");
	cv::namedWindow("Dst Video");
	int delay = 1000 / rate;
	cv::Mat dMap;
	cv::Mat pMap;
	param pa;
	while (!stop)
	{
		if (!cap.read(frame))
			break;
		cv::imshow("Source Video", frame);
		doMBS(frame, pa, pMap, dMap);
		cv::imshow("Dst Video", pMap);

		if (cv::waitKey(delay) >= 0)
			stop = true;
	}
	/*cv::Mat testImg = cv::imread("69648.jpg", 1);
	cv::imshow("source", testImg);
	cv::Mat dst;
	cv::Mat pMap;
	cv::Mat dMap;
	param pa;
	pa.verbose = true;
	doMBS(testImg, pa, pMap, dMap);
	cv::imshow("dMap", dMap);
	cv::imshow("pMap", pMap);*/
	/*pMap.convertTo(dst, CV_8U, 255);
	cv::imwrite("test.jpg", dst);*/
	
	cap.release();
	return 0;
}