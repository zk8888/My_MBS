/*****************************************************************************
*	Implemetation of the saliency detction method described in paper
*	"Minimum Barrier Salient Object Detection at 80 FPS", Jianming Zhang,
*	Stan Sclaroff, Zhe Lin, Xiaohui Shen, Brian Price, Radomir Mech, ICCV,
*       2015
*
*	Copyright (C) 2015 Jianming Zhang
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
*	If you have problems about this software, please contact:
*       jimmie33@gmail.com
*******************************************************************************/

#ifndef MBS_H
#define MBS_H

#ifdef IMDEBUG
#include <imdebug.h>
#endif
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>


static cv::RNG MBS_RNG;

cv::Mat getCenterMap(int dim);

struct param
{
	int MAX_DIM = 300;                        // max image dimension in computation
	bool use_lab = true;                       // use Lab color space
	bool remove_border = true;                 // detect and remove artificial image frames
	bool use_geodesic = false;                 // flag for replacing geodesic distance with MBD
	bool use_backgroundness = true;            // MB +
	int COV_REG = 50;                         // covariance regularization term for the maximum value of 255 * 255
	float MARGIN_RATIO = 0.1;                   // the boundary margion for computing backgroundness map
	cv::Mat cmap = getCenterMap(MAX_DIM);   // the center distance map for center bias
	bool center_bias = true;
	int smooth_alpha = 50;                    // see eqn. 9
	int contrast_b = 10;                      // see eqn. 11
	bool verbose = false;
};

class MBS
{
public:
	MBS(const cv::Mat& src);
	cv::Mat getSaliencyMap();
	void computeSaliency(bool use_geodesic = false);
	cv::Mat getMBSMap() const { return mMBSMap; }
private:
	cv::Mat mSaliencyMap;
	cv::Mat mMBSMap;
	int mAttMapCount;
	cv::Mat mBorderPriorMap;
	cv::Mat mSrc;
	std::vector<cv::Mat> mFeatureMaps;
	void whitenFeatMap(float reg);
	void computeBorderPriorMap(float reg, float marginRatio);
};

cv::Mat computeCWS(const cv::Mat src, float reg, float marginRatio);
cv::Mat fastMBS(const std::vector<cv::Mat> featureMaps);
cv::Mat fastGeodesic(const std::vector<cv::Mat> featureMaps);

int findFrameMargin(const cv::Mat& img, bool reverse);
bool removeFrame(const cv::Mat& inImg, cv::Mat& outImg, cv::Rect &roi);
cv::Mat doWork(const cv::Mat& src, bool use_lab, bool remove_border,
	bool use_geodesic);
#endif