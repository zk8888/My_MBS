//17.7.30 ZK
#include "../include/MBS.hpp"
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include "ipp.h"
#include <ippi.h>

cv::Mat doWork(
	const cv::Mat& src,
	bool use_lab,
	bool remove_border,
	bool use_geodesic
	);

cv::Mat getCenterMap(const int dim)
{
	//cv::Mat grayImg(dim, dim, CV_8UC1);
	cv::Mat tmpImg(dim, dim, CV_32FC1);
	cv::Mat tmp2Img = tmpImg.clone();
	int iCenterX = dim / 2 - 1;
	int iCenterY = dim / 2 - 1;
	for (int i = 0; i < dim; ++i)
	{
		float * data = tmpImg.ptr<float>(i);
		for (int j = 0; j < dim; ++j)
		{
			data[j] = sqrt(pow(i - iCenterX, 2) + pow(j - iCenterY, 2));
		}
	}
	cv::normalize(tmpImg, tmp2Img, 1, 0, cv::NORM_MINMAX);
	tmp2Img = 1 - tmp2Img;
	return tmp2Img;
}

//???????????(?????????IPP??
/*void morphRDilate(const cv::Mat src, const cv::Mat mask, cv::Mat &dst, cv::InputArray kernel = NULL, int iterations = -1)
{
	assert(src != NULL&&mask != NULL&&dst != NULL&&src != dst);
	if (iterations < 0)
	{
		//???????
		cv::min(src, mask, dst);
		cv::dilate(dst, dst, kernel);
		cv::min(dst, mask, dst);

		cv::Mat temp1(src.size(), CV_8UC1);
		cv::Mat temp2(src.size(), CV_8UC1);
		do
		{
			//record last result
			dst.copyTo(temp1);
			cv::dilate(dst, dst, kernel);
			cv::min(dst, mask, dst);
			cv::compare(temp1, dst, temp2, CV_CMP_NE);
		} while (cv::sum(temp2).val[0] != 0);
	}
	else if (iterations == 0)
	{
		src.copyTo(dst);
	}
	else
	{
		//??????????
		cv::min(src, mask, dst);
		cv::dilate(dst, dst, kernel);
		cv::min(dst, mask, dst);
		for (int i = 1; i < iterations; i++)
		{
			cv::dilate(dst, dst, kernel);
			cv::min(dst, mask, dst);
		}
	}
}*/

//?????????????????????IPP??
/*void morphRErode(const cv::Mat src, const cv::Mat mask, cv::Mat &dst, cv::InputArray kernel = NULL, int iterations = -1)
{
	assert(src != NULL&&mask != NULL&&dst != NULL&&src != dst);

	if (iterations < 0)
	{
		//???????
		cv::max(src, mask, dst);
		cv::erode(dst, dst, kernel);
		cv::max(dst, mask, dst);

		cv::Mat temp1(src.size(), CV_8UC1);
		cv::Mat temp2(src.size(), CV_8UC1);

		do
		{
			//record last result
			dst.copyTo(temp1);
			cv::erode(dst, dst, kernel);
			cv::max(dst, mask, dst);
			cv::compare(temp1, dst, temp2, CV_CMP_NE);
		} while (cv::sum(temp2).val[0]!=0);
	}
	else if (iterations == 0)
	{
		src.copyTo(dst);
	}
	else
	{
		//????????
		cv::max(src, mask, dst);
		cv::erode(dst, dst, kernel);
		cv::max(dst, mask, dst);

		for (int i = 1; i < iterations; i++)
		{
			cv::erode(dst, dst, kernel);
			cv::max(dst, mask, dst);
		}
	}
}*/

//src??????CV_32FC1
cv::Mat morphSmooth(const cv::Mat &src, const int width)
{
	IppiMorphAdvState *pState = NULL;
	IppiSize roiSize = { src.cols, src.rows };
	IppiSize maskSize = { width, width };
	Ipp8u *pBuf = NULL, *pAdvBuf = NULL, pMask[9] = { 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	int srcStep = src.cols;
	int size = 0, specSize = 0, bufferSize = 0;     /*  working buffers size */
	IppStatus status = ippiMorphReconstructGetBufferSize(roiSize, ipp8u, 1, &size); //??????????????????????§³
	pBuf = ippsMalloc_8u(size);
	status = ippiMorphAdvGetSize_8u_C1R(roiSize, maskSize, &specSize, &bufferSize);//???????????????????????§³
	pState = (IppiMorphAdvState*)ippsMalloc_8u(specSize);
	pAdvBuf = (Ipp8u*)ippsMalloc_8u(bufferSize);
	status = ippiMorphAdvInit_8u_C1R(roiSize, pMask, maskSize, pState, pAdvBuf);

	cv::Mat tempImg;
	src.convertTo(tempImg, CV_8U, 255);
	cv::Mat Ie;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(width, width)); //?????????????????
	cv::erode(tempImg, Ie, element);
	status = ippiMorphReconstructDilate_8u_C1IR((const Ipp8u *)& tempImg.data[0], srcStep, (Ipp8u*)&Ie.data[0], srcStep, roiSize, pBuf, (IppiNorm)ippiNormL1);
	cv::Mat lobrd;
	cv::dilate(Ie, lobrd, element);
	//lobrd = 255 - lobrd;
	//Ie = 255 - Ie;
	status = ippiMorphReconstructErode_8u_C1IR((const Ipp8u *)& Ie.data[0], srcStep, (Ipp8u*)&lobrd.data[0], srcStep, roiSize, pBuf, (IppiNorm)ippiNormL1);
	//lobrd = 255 - lobrd;
	cv::Mat dstImg;
	lobrd.convertTo(dstImg, CV_32F, 1.0 / 255);
	cv::normalize(dstImg, dstImg, 1.0, 0.0, cv::NORM_MINMAX);
	ippsFree(pBuf);
	ippsFree(pState);
	ippsFree(pAdvBuf);
	return dstImg;
}

cv::Mat enhanceContrast(cv::Mat &src, const int b)
{
	cv::Mat pMap(src.size(),src.type(),cv::Scalar::all(0));
	double minv = 0.0, maxv = 0.0;
	double * minp = &minv;
	double * maxp = &maxv;
	cv::minMaxIdx(src, minp, maxp);
	float t = 0.5 * maxv;
	float v1 = 0, v2 = 0;
	float tempnum = 0;
	int countv1 = 0, countv2 = 0;
	for (int i = 0; i < src.size().height; ++i)
	{
		float * data = src.ptr<float>(i);
		for (int j = 0; j < src.size().width; ++j)
		{
			tempnum = data[j];
			if (tempnum >= t)
			{
				v1 += tempnum;
				countv1++;
			}
			else
			{
				v2 += tempnum;
				countv2++;
			}
		}
	}
	v1 = 1.0 * v1 / countv1;
	v2 = 1.0 * v2 / countv2;
	float sumv1v2 = 0.5 * (v1 + v2);
	for (int i = 0; i < src.size().height; i++)
	{
		float * datas = src.ptr<float>(i);
		float * datad = pMap.ptr<float>(i);
		for (int j = 0; j < src.size().width; j++)
		{
			datad[j] = 1.0 / (1 + exp(-1.0 * b * (datas[j] - sumv1v2)));
		}
	}
	return pMap;
}

cv::Mat BG(const cv::Mat &srcImg, const int reg, const float m_ratio)  //I?uchar3?????????
{
	cv::Mat I;
	cv::cvtColor(srcImg, I, CV_BGR2RGB);
	cv::Mat bgMapImg(I.size().width*I.size().height, 1, CV_32FC1, cv::Scalar::all(0));
	cv::Mat bgMapGrayImg(I.size().height,I.size().width, CV_32FC1, cv::Scalar::all(0));
	//???????????????????
	cv::Mat tmpMap1(I.size().width*I.size().height, 1, CV_32FC1, cv::Scalar::all(0));
	cv::Mat tmpMap2(I.size().width*I.size().height, 1, CV_32FC1, cv::Scalar::all(0));
	cv::Mat tmpMap3(I.size().width*I.size().height, 1, CV_32FC1, cv::Scalar::all(0));
	cv::Mat tmpMap4(I.size().width*I.size().height, 1, CV_32FC1, cv::Scalar::all(0));

	cv::Mat eye = (cv::Mat_<float>(3, 3) << reg*1.0, 0, 0, 0, reg*1.0, 0, 0, 0, reg*1.0);
	int rowMargin = round(I.size().height*m_ratio);
	int colMargin = round(I.size().width*m_ratio);
	cv::Mat firstRow(rowMargin*I.size().width, 3, CV_32FC1, cv::Scalar::all(0));
	cv::Mat meanfirstRow(1, 3, CV_32FC1, cv::Scalar::all(0));
	cv::Mat lastRow(rowMargin*I.size().width, 3, CV_32FC1, cv::Scalar::all(0));
	cv::Mat meanlastRow(1, 3, CV_32FC1, cv::Scalar::all(0));
	cv::Mat firstCol(colMargin*I.size().height, 3, CV_32FC1, cv::Scalar::all(0));
	cv::Mat meanfirstCol(1, 3, CV_32FC1, cv::Scalar::all(0));
	cv::Mat lastCol(colMargin*I.size().height, 3, CV_32FC1, cv::Scalar::all(0));
	cv::Mat meanlastCol(1, 3, CV_32FC1, cv::Scalar::all(0));
	float fMeanX1, fMeanX2, fMeanX3;
	float *dataMean;
	//firstRow
	dataMean = meanfirstRow.ptr<float>(0);
	for (int i = 0; i < rowMargin; ++i)
	{
		uchar * datas = I.ptr<uchar>(i);
		for (int j = 0; j < I.size().width; ++j)
		{
			float * datad = firstRow.ptr<float>(j*rowMargin + i);
			for (int k = 0; k < 3; ++k)
			{
				 datad[k] = datas[j * 3 + k];
				 dataMean[k] += datad[k];
			}
		}
	}
	meanfirstRow.convertTo(meanfirstRow, CV_32FC1, 1.0 / firstRow.size().height);

	//lastRow
	dataMean = meanlastRow.ptr<float>(0);
	for (int i = I.size().height-rowMargin; i < I.size().height; ++i)
	{
		uchar * datas = I.ptr<uchar>(i);
		for (int j = 0; j < I.size().width; ++j)
		{
			float * datad = lastRow.ptr<float>(j*rowMargin + i - I.size().height + rowMargin);
			for (int k = 0; k < 3; ++k)
			{
				datad[k] = datas[j * 3 + k];
				dataMean[k] += datad[k];
			}
		}
	}
	meanlastRow.convertTo(meanlastRow, CV_32FC1, 1.0 / lastRow.size().height);

	//firstCol
	dataMean = meanfirstCol.ptr<float>(0);
	for (int i = 0; i < I.size().height; ++i)
	{
		uchar * datas = I.ptr<uchar>(i);
		for (int j = 0; j < colMargin; ++j)
		{
			float * datad = firstCol.ptr<float>(i*colMargin + j);
			for (int k = 0; k < 3; ++k)
			{
				datad[k] = datas[j*3+k];
				dataMean[k] += datad[k];
			}
		}
	}
	meanfirstCol.convertTo(meanfirstCol, CV_32FC1, 1.0 / firstCol.size().height);

	//lastCol
	dataMean = meanlastCol.ptr<float>(0);
	for (int i = 0; i < I.size().height; ++i)
	{
		uchar * datas = I.ptr<uchar>(i);
		
		for (int j = I.size().width-colMargin; j < I.size().width; ++j)
		{
			float * datad = lastCol.ptr<float>(i*colMargin + j - I.size().width + colMargin);
			for (int k = 0; k < 3; ++k)
			{
				datad[k] = datas[j*3+k];
				dataMean[k] += datad[k];
			}
		}
	}
	meanlastCol.convertTo(meanlastCol, CV_32FC1, 1.0 / lastCol.size().height);


	cv::Mat covMat1;
	cv::Mat meanMat1;
	cv::calcCovarMatrix(firstRow, covMat1, meanMat1, CV_COVAR_NORMAL | CV_COVAR_ROWS, CV_32FC1);
	if (firstRow.size().height>1)
	{
		covMat1.convertTo(covMat1, CV_32FC1, 1.0 / (firstRow.size().height - 1));
	}
	covMat1 = covMat1 + eye;
	//displayMat(covMat1);
	cv::SVD thissvd;
	cv::Mat U;
	cv::Mat S;
	cv::Mat VT;
	cv::Mat img_mean;
	cv::Mat img_std;
	thissvd.compute(covMat1, S, U, VT, cv::SVD::FULL_UV);

	cv::Mat P(3, 3, CV_32FC1);
	for (int i = 0; i < 3; i++)
	{
		float * datas = P.ptr<float>(i);
		float * datas2 = U.ptr<float>(i);
		for (int j = 0; j < 3; j++)
		{
			datas[j] = datas2[j] * 1.0 / sqrt(S.at<float>(j, 0));
		}
	}
	cv::Mat reshapeImg(I.size().height*I.size().width,3,CV_32FC1,cv::Scalar::all(0));
	dataMean = meanfirstRow.ptr<float>(0);
	for (int i = 0; i < I.size().height; ++i)
	{
		uchar * datas = I.ptr<uchar>(i);
		for (int j = 0; j < I.size().width; ++j)
		{
			float * datad = reshapeImg.ptr<float>(j*I.size().height + i);
			for (int k = 0; k < 3; k++)
			{
				datad[k] = datas[j * 3 + k] - dataMean[k];
			}
		}
	}
	cv::Mat reshapeImgP;
	//OpenCV???????
	reshapeImgP = reshapeImg * P;
	reshapeImgP = cv::abs(reshapeImgP);

	//?????????????????
	for (int i = 0; i < reshapeImgP.size().height; ++i)
	{
		float * datas = reshapeImgP.ptr<float>(i);
		float * datad = tmpMap1.ptr<float>(i);
		for (int j = 0; j < reshapeImgP.size().width; ++j)
		{
			datad[0] += datas[j];
		}
	}
	cv::normalize(tmpMap1, tmpMap1, 1.0, 0.0, cv::NORM_MINMAX);

	cv::Mat covMat2;
	cv::Mat meanMat2;
	P.setTo(0);
	reshapeImg.setTo(0);
	reshapeImgP.setTo(0);
	cv::calcCovarMatrix(lastRow, covMat2, meanMat2, CV_COVAR_NORMAL | CV_COVAR_ROWS, CV_32FC1);
	if (lastRow.size().height>1)
	{
		covMat2.convertTo(covMat2, CV_32FC1, 1.0 / (lastRow.size().height - 1));
	}
	covMat2 = covMat2 + eye;
	//displayMat(covMat2);
	thissvd.compute(covMat2, S, U, VT, cv::SVD::FULL_UV);
	for (int i = 0; i < 3; i++)
	{
		float * datas = P.ptr<float>(i);
		float * datas2 = U.ptr<float>(i);
		for (int j = 0; j < 3; j++)
		{
			datas[j] = datas2[j] * 1.0 / sqrt(S.at<float>(j, 0));
		}
	}
	dataMean = meanlastRow.ptr<float>(0);
	for (int i = 0; i < I.size().height; ++i)
	{
		uchar * datas = I.ptr<uchar>(i);
		for (int j = 0; j < I.size().width; ++j)
		{
			float * datad = reshapeImg.ptr<float>(j*I.size().height + i);
			for (int k = 0; k < 3; ++k)
			{
				datad[k] = datas[j * 3 + k] - dataMean[k];
			}
		}
	}

	//OpenCV???????
	reshapeImgP = reshapeImg * P;
	reshapeImgP = cv::abs(reshapeImgP);

	//?????????????????
	for (int i = 0; i < reshapeImgP.size().height; ++i)
	{
		float * datas = reshapeImgP.ptr<float>(i);
		float * datad = tmpMap2.ptr<float>(i);
		for (int j = 0; j < reshapeImgP.size().width; ++j)
		{
			datad[0] += datas[j];
		}
	}
	cv::normalize(tmpMap2, tmpMap2, 1.0, 0.0, cv::NORM_MINMAX);

	
	cv::Mat covMat3;
	cv::Mat meanMat3;
	P.setTo(0);
	reshapeImg.setTo(0);
	reshapeImgP.setTo(0);
	cv::calcCovarMatrix(firstCol, covMat3, meanMat3, CV_COVAR_NORMAL | CV_COVAR_ROWS, CV_32FC1);
	if (firstCol.size().height>1)
	{
		covMat3.convertTo(covMat3, CV_32FC1, 1.0 / (firstCol.size().height - 1));
	}
	covMat3 = covMat3 + eye;
	//displayMat(covMat3);
	thissvd.compute(covMat3, S, U, VT, cv::SVD::FULL_UV);
	for (int i = 0; i < 3; ++i)
	{
		float * datas = P.ptr<float>(i);
		float * datas2 = U.ptr<float>(i);
		for (int j = 0; j < 3; j++)
		{
			datas[j] = datas2[j] * 1.0 / sqrt(S.at<float>(j, 0));
		}
	}
	dataMean = meanfirstCol.ptr<float>(0);
	for (int i = 0; i < I.size().height; ++i)
	{
		uchar * datas = I.ptr<uchar>(i);
		for (int j = 0; j < I.size().width; ++j)
		{
			float * datad = reshapeImg.ptr<float>(j*I.size().height + i);
			for (int k = 0; k < 3; ++k)
			{
				datad[k] = datas[j * 3 + k] - dataMean[k];
			}
		}
	}
	//OpenCV???????
	reshapeImgP = reshapeImg * P;
	reshapeImgP = cv::abs(reshapeImgP);
	//?????????????????
	for (int i = 0; i < reshapeImgP.size().height; ++i)
	{
		float * datas = reshapeImgP.ptr<float>(i);
		float * datad = tmpMap3.ptr<float>(i);
		for (int j = 0; j < reshapeImgP.size().width; ++j)
		{
			datad[0] += datas[j];
		}
	}
	cv::normalize(tmpMap3, tmpMap3, 1.0, 0.0, cv::NORM_MINMAX);

	cv::Mat covMat4;
	cv::Mat meanMat4;
	P.setTo(0);
	reshapeImg.setTo(0);
	reshapeImgP.setTo(0);
	cv::calcCovarMatrix(lastCol, covMat4, meanMat4, CV_COVAR_NORMAL | CV_COVAR_ROWS, CV_32FC1);
	if (lastCol.size().height>1)
	{
		covMat4.convertTo(covMat4, CV_32FC1, 1.0 / (lastCol.size().height - 1));
	}
	covMat4 = covMat4 + eye;
	//displayMat(covMat4);
	thissvd.compute(covMat4, S, U, VT, cv::SVD::FULL_UV);
	for (int i = 0; i < 3; ++i)
	{
		float * datas = P.ptr<float>(i);
		float * datas2 = U.ptr<float>(i);
		for (int j = 0; j < 3; ++j)
		{
			datas[j] = datas2[j] * 1.0 / sqrt(S.at<float>(j, 0));
		}
	}
	dataMean = meanlastCol.ptr<float>(0);
	for (int i = 0; i < I.size().height; ++i)
	{
		uchar * datas = I.ptr<uchar>(i);
		for (int j = 0; j < I.size().width; ++j)
		{
			float * datad = reshapeImg.ptr<float>(j*I.size().height + i);
			for (int k = 0; k < 3; ++k)
			{
				datad[k] = datas[j * 3 + k] - dataMean[k];
			}
		}
	}
	//OpenCV???????
	reshapeImgP = reshapeImg * P;
	reshapeImgP = cv::abs(reshapeImgP);
	//?????????????????
	for (int i = 0; i < reshapeImgP.size().height; ++i)
	{
		float * datas = reshapeImgP.ptr<float>(i);
		float * datad = tmpMap4.ptr<float>(i);
		for (int j = 0; j < reshapeImgP.size().width; ++j)
		{
			datad[0] += datas[j];
		}
	}
	cv::normalize(tmpMap4, tmpMap4, 1.0, 0.0, cv::NORM_MINMAX);

	float tempNum = 0;
	for (int i = 0; i < bgMapImg.size().height; ++i)
	{
		float * data1 = tmpMap1.ptr<float>(i);
		float * data2 = tmpMap2.ptr<float>(i);
		float * data3 = tmpMap3.ptr<float>(i);
		float * data4 = tmpMap4.ptr<float>(i);
		float * datad = bgMapImg.ptr<float>(i);
		tempNum = data1[0];
		if (tempNum < data2[0])
			tempNum = data2[0];
		if (tempNum < data3[0])
			tempNum = data3[0];
		if (tempNum < data4[0])
			tempNum = data4[0];
		datad[0] = data1[0] + data2[0] + data3[0] + data4[0] - tempNum;
		tempNum = 0;
	}
	//bgMapGrayImg = bgMapImg.reshape(0, I.size().height);
	for (int i = 0; i < bgMapGrayImg.size().height; ++i)
	{
		float * datas = bgMapGrayImg.ptr<float>(i);
		for (int j = 0; j < bgMapGrayImg.size().width; ++j)
		{
			datas[j] = bgMapImg.at<float>(j*bgMapGrayImg.size().height + i, 0);
		}
	}
	cv::normalize(bgMapGrayImg, bgMapGrayImg, 1.0, 0.0, cv::NORM_MINMAX);
	/////////////////////
	return bgMapGrayImg;
}

void doMBS(const cv::Mat &I, const param pa, cv::Mat &pMap, cv::Mat &dMap) //???????I?????????????????????uchar??
{
	cv::Mat bgMap;
	cv::Mat cmap;
	float scale = pa.MAX_DIM*1.0 / std::max(I.size().height, I.size().width);
	cv::Mat Ir;
	cv::Mat Ir3;
	cv::Mat Irr;
	cv::resize(I, Ir, cv::Size(0,0), scale, scale); //opencv??resize??Matlab??imresize?§Ö????????????
	
	//check the type of the Ir
	if (Ir.channels() == 1)
	{
		cv::cvtColor(Ir, Ir3, CV_GRAY2BGR);
		Ir3.copyTo(Irr);
	}
	else
	{
		Ir.copyTo(Irr);
	}
	
	//compute saliency
	double time1 = static_cast<double>(cv::getTickCount());
	dMap = doWork(Irr, pa.use_lab, pa.remove_border, pa.use_geodesic);
	double time2 = (static_cast<double>(cv::getTickCount()) - time1) / cv::getTickFrequency();

	double time3, time4;
	if (pa.use_backgroundness)
	{
		time3 = static_cast<double>(cv::getTickCount());
		bgMap = BG(Irr, pa.COV_REG, pa.MARGIN_RATIO);
		time4 = (static_cast<double>(cv::getTickCount()) - time3) / cv::getTickFrequency();

		dMap = dMap + bgMap;
	}
	pMap = dMap.clone();

	//postprocess
	double time5 = static_cast<double>(cv::getTickCount());
	if (pa.center_bias)
	{
		cv::resize(pa.cmap, cmap, pMap.size());
		pMap = pMap.mul(cmap);
	}
	cv::normalize(pMap, pMap, 1.0, 0.0, cv::NORM_MINMAX);
	int radius = floor(pa.smooth_alpha*sqrt(cv::mean(pMap)[0]));
	pMap = morphSmooth(pMap, cv::max(radius, 3));
	pMap = enhanceContrast(pMap, pa.contrast_b);
	double time6 = (static_cast<double>(cv::getTickCount()) - time5) / cv::getTickFrequency();

	cv::resize(pMap, pMap, I.size());
	cv::normalize(pMap, pMap, 1.0, 0.0, cv::NORM_MINMAX);
	if (pa.verbose)
	{
		printf("computing MBD map: %f\n", time2);
		if (pa.use_backgroundness)
			printf("computing BG map: %f\n", time4);
		printf("postprocessing: %f\n", time6);
	}
}
