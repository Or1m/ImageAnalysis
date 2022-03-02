#pragma once

#include <iostream>
#include <random>
#include <math.h>
#include <limits>
#include <map>

// Consts
constexpr int starIdx = 4;
constexpr int rectangleIdx = 8;

constexpr int numOfSquares = 4;
constexpr int numOfStars = 4;
constexpr int numOfRectangles = 4;

constexpr int k = 3;

// Enums & Structs
enum EClassType {
	Square, Rectangle, Star
};
enum EClassificationType {
	Ethalons, KMeans
};

struct FeatureVector {
	double f1, f2;
};

// Prototypes
// Thresholding
void ApplyThresholding(cv::Mat* coloredImg, cv::Mat& grayScaleImg, const uchar threshold, bool grayscaleOnly = false);

// Floodfill
int ApplyFloodFill(cv::Mat* coloredImg, cv::Mat& grayScaleImg, bool grayscaleOnly = false);

template <typename T>
void floodFill(cv::Mat& img, T& newColor, int y, int x);
template <typename T>
void floodFillUtil(cv::Mat& img, int x, int y, T& prevColor, T& newColor);

void floodFillUtilPrimitive(cv::Mat& img, int x, int y, uchar prevColor, uchar newColor);

void floodFillPrimitive(cv::Mat& img, uchar newColor, int y, int x);

// Features
void CalcFeatures(int objectCount, cv::Mat& grayScaleImg, std::vector<FeatureVector>& features);

inline bool IsPartOfCircumference(cv::Mat& grayScaleImg, int y, int x, int i);

inline double Mu(int p, int q, int x, double xt, int y, double yt);
inline double MuMax(double mu20, double mu02, double mu11);
inline double MuMin(double mu20, double mu02, double mu11);
inline double M(int p, int q, int x, int y);

// Labeling & ethalons
void CalcEthalons(std::vector<FeatureVector>& features, std::map<EClassType, FeatureVector>& ethalons);

void CompareFeaturesWithEthalons(const std::map<EClassType, FeatureVector>& ethalons, const std::vector<FeatureVector>& testFeatures);

// Utils
double Euklid(const FeatureVector& feature, const FeatureVector& ethalon);

std::ostream& operator<<(std::ostream& os, const EClassType type);