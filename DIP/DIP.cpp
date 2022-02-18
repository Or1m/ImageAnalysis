#include "stdafx.h"
#include <random>
#include <iostream>

template <typename T>
void floodFill(cv::Mat& img, T& newColor, int y, int x);
template <typename T>
void floodFillUtil(cv::Mat& img, int x, int y, T& prevColor, T& newColor);

void floodFillUtilPrimitive(cv::Mat& img, int x, int y, uchar prevColor, uchar newColor);
void floodFillPrimitive(cv::Mat& img, uchar newColor, int y, int x);

int main()
{
	const cv::Vec3b whitePoint = cv::Vec3b(255, 255, 255);
	const cv::Vec3b blackPoint = cv::Vec3b(0, 0, 0);

	const uchar whiteUchar = 255;
	const uchar blackUchar = 0;

	const cv::String filename = "images/train.png";
	const uchar treshold = 128;
	
	cv::Vec3b newColor;
	cv::Mat sourceImg, grayScaleImg;

	std::random_device rd; // obtain a random number from hardware
	std::mt19937 gen(rd()); // seed the generator
	std::uniform_int_distribution<> distr(0, 255); // define the range


	// load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	sourceImg = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
	cv::cvtColor(sourceImg, grayScaleImg, cv::COLOR_BGR2GRAY);

	if (sourceImg.empty()) {
		printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
		return EXIT_FAILURE;
	}

	for (int y = 0; y < sourceImg.rows; y++) {

		for (int x = 0; x < sourceImg.cols; x++) {

			if (sourceImg.at<cv::Vec3b>(y, x)[0] > treshold ||
				sourceImg.at<cv::Vec3b>(y, x)[1] > treshold ||
				sourceImg.at<cv::Vec3b>(y, x)[2] > treshold) {

				sourceImg.at<cv::Vec3b>(y, x) = whitePoint;
				grayScaleImg.at<uchar>(y, x) = whiteUchar;
			}
			else {
				sourceImg.at<cv::Vec3b>(y, x) = blackPoint;
				grayScaleImg.at<uchar>(y, x) = blackUchar;
			}
		}
	}

	// Asi to zjednot pod tento cyklus
	/*for (int y = 0; y < grayScaleImg.rows; y++) {

		for (int x = 0; x < grayScaleImg.cols; x++) {

			if (grayScaleImg.at<uchar>(y, x) > treshold)
				grayScaleImg.at<uchar>(y, x) = whiteUchar;
			else
				grayScaleImg.at<uchar>(y, x) = blackUchar;
		}
	}*/

	/*for (int y = 0; y < grayScaleImg.rows; y++) {
		for (int x = 0; x < grayScaleImg.cols; x++) {
			std::cout << (int)grayScaleImg.at<uchar>(y, x) << " ";
		}
	}
	return 0;*/

	std::vector<cv::Vec3b> colors;
	
	for (int y = 0; y < sourceImg.rows; y++) {

		for (int x = 0; x < sourceImg.cols; x++) {

			if (grayScaleImg.at<uchar>(y, x) == whiteUchar) {

				newColor = cv::Vec3b(distr(gen), distr(gen), distr(gen));
				colors.emplace_back(newColor);

				floodFill(sourceImg, newColor, y, x);
			}
		}
	}

	/*uchar index = 1;

	for (int y = 0; y < indexedImg.rows; y++) {

		for (int x = 0; x < indexedImg.cols; x++) {

			if (indexedImg.at<uchar>(y, x) == whiteUchar) {
				floodFillPrimitive(indexedImg, index++, y, x);
			}
		}
	}*/

	/*for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			std::cout << (int)indexedImg.at<uchar>(y, x) << " ";
		}
	}*/

	//std::cout << std::endl;
	//std::cout << "MAX INDEX: " << (int)index << std::endl;

	// diplay image
	cv::imshow("Separated Image", sourceImg);
	//cv::imshow("Indexed Image", indexedImg);
	// wait until keypressed
	cv::waitKey(0); 

	return EXIT_SUCCESS;
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
