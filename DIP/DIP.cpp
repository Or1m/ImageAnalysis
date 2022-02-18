#include "stdafx.h"
#include <random>
#include <iostream>

template <typename T>
void floodFill(cv::Mat& img, T& newColor, int y, int x);
template <typename T>
void floodFillUtil(cv::Mat& img, int x, int y, T& prevColor, T& newColor);

void floodFillUtilPrimitive(cv::Mat& img, int x, int y, uchar prevColor, uchar newColor);
void floodFillPrimitive(cv::Mat& img, uchar newColor, int y, int x);

void ApplyThresholding(cv::Mat& sourceImg, cv::Mat& grayScaleImg, const uchar threshold);

int main()
{
	const cv::String filename = "images/train.png";
	const uchar threshold = 128;

	cv::Mat sourceImg, grayScaleImg;
	
	// load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	sourceImg = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
	cv::cvtColor(sourceImg, grayScaleImg, cv::COLOR_BGR2GRAY);

	if (sourceImg.empty()) {
		printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
		return EXIT_FAILURE;
	}

	ApplyThresholding(sourceImg, grayScaleImg, threshold);


	//std::random_device rd; // obtain a random number from hardware
	//std::mt19937 gen(rd()); // seed the generator
	//std::uniform_int_distribution<> distr(0, 255); // define the range

	//std::vector<cv::Vec3b> colors;
	//cv::Vec3b newColor;
	uchar index = 1;
	
	for (int y = 0; y < sourceImg.rows; y++) {

		for (int x = 0; x < sourceImg.cols; x++) {

			//if (sourceImg.at<cv::Vec3b>(y, x) == cv::Vec3b(255, 255, 255)) {
			if (grayScaleImg.at<uchar>(y, x) == 255) {
			
				/*newColor = cv::Vec3b(distr(gen), distr(gen), distr(gen));
				colors.emplace_back(newColor);
				floodFill(sourceImg, newColor, y, x);*/

				floodFillPrimitive(grayScaleImg, index++, y, x);
			}
		}
	}

	// Debug output
	/*for (int y = 0; y < grayScaleImg.rows; y++) {
		for (int x = 0; x < grayScaleImg.cols; x++) {
			std::cout << (int)grayScaleImg.at<uchar>(y, x) << " ";
		}
	}*/

	std::cout << "MAX INDEX: " << (int)index - 1<< std::endl;

	// diplay image
	cv::imshow("Separated Image", grayScaleImg);
	//cv::imshow("Indexed Image", indexedImg);
	// wait until keypressed
	cv::waitKey(0); 

	return EXIT_SUCCESS;
}

void ApplyThresholding(cv::Mat& sourceImg, cv::Mat& grayScaleImg, const uchar threshold)
{
	for (int y = 0; y < sourceImg.rows; y++) {

		for (int x = 0; x < sourceImg.cols; x++) {

			if (sourceImg.at<cv::Vec3b>(y, x)[0] > threshold ||
				sourceImg.at<cv::Vec3b>(y, x)[1] > threshold ||
				sourceImg.at<cv::Vec3b>(y, x)[2] > threshold) {

				//sourceImg.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
				grayScaleImg.at<uchar>(y, x) = 255;
			}
			else {
				//sourceImg.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
				grayScaleImg.at<uchar>(y, x) = 0;
			}
		}
	}
}

template <typename T>
void floodFill(cv::Mat& img, T& newColor, int y, int x)
{
	T prevColor = img.at<T>(y, x);

	if (prevColor == newColor)
		return;

	floodFillUtil(img, x, y, prevColor, newColor);
}

template <typename T>
void floodFillUtil(cv::Mat& img, int x, int y, T& prevColor, T& newColor)
{
	// Base cases
	if (x < 0 || x >= img.cols || y < 0 || y >= img.rows)
		return;
	if (img.at<T>(y, x) != prevColor)
		return;
	if (img.at<T>(y, x) == newColor)
		return;

	// Replace the color at (x, y)
	img.at<T>(y, x) = newColor;

	// Recur for north, east, south and west
	floodFillUtil(img, x + 1, y, prevColor, newColor);
	floodFillUtil(img, x - 1, y, prevColor, newColor);
	floodFillUtil(img, x, y + 1, prevColor, newColor);
	floodFillUtil(img, x, y - 1, prevColor, newColor);

	floodFillUtil(img, x + 1, y + 1, prevColor, newColor);
	floodFillUtil(img, x - 1, y + 1, prevColor, newColor);
	floodFillUtil(img, x + 1, y - 1, prevColor, newColor);
	floodFillUtil(img, x - 1, y - 1, prevColor, newColor);
}


void floodFillPrimitive(cv::Mat& img, uchar newColor, int y, int x)
{
	uchar prevColor = img.at<uchar>(y, x);

	if (prevColor == newColor)
		return;

	floodFillUtilPrimitive(img, x, y, prevColor, newColor);
}

void floodFillUtilPrimitive(cv::Mat& img, int x, int y, uchar prevColor, uchar newColor)
{
	// Base cases
	if (x < 0 || x >= img.cols || y < 0 || y >= img.rows)
		return;
	if (img.at<uchar>(y, x) != prevColor)
		return;
	if (img.at<uchar>(y, x) == newColor)
		return;

	// Replace the color at (x, y)
	img.at<uchar>(y, x) = newColor;

	// Recur for north, east, south and west
	floodFillUtil(img, x + 1, y, prevColor, newColor);
	floodFillUtil(img, x - 1, y, prevColor, newColor);
	floodFillUtil(img, x, y + 1, prevColor, newColor);
	floodFillUtil(img, x, y - 1, prevColor, newColor);
}
