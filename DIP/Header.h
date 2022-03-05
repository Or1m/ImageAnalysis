#pragma once

#include <iostream>
#include <random>
#include <math.h>
#include <limits>
#include <map>

// Consts
constexpr int starIdx           = 4;
constexpr int rectangleIdx      = 8;

constexpr int numOfSquares      = 4;
constexpr int numOfStars        = 4;
constexpr int numOfRectangles   = 4;
constexpr int expectedCount     = 4;

constexpr int k                 = 3;

// Enums & Structs
enum class EClassType {
	Unknown, Square, Rectangle, Star
};
enum class EClassificationType {
	Ethalons, KMeans
};

struct FeatureVector {
    double f1, f2;

    FeatureVector() : FeatureVector(0, 0) { }

    FeatureVector(double f1, double f2) {
        this->f1 = f1;
        this->f2 = f2;
    }

    bool const operator==(const FeatureVector& f) const {
        return f1 == f.f1 && f2 == f.f2;
    }

    bool const operator<(const FeatureVector& f) const {
        return f1 < f.f1 || (f1 == f.f1 && f2 < f.f2);
    }
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
void CalcFeatures(int objectCount, cv::Mat& sourceImg, cv::Mat& grayScaleImg, std::vector<FeatureVector>& features);

void AddLabelToImage(cv::Mat& grayScaleImg, int xt, int yt, int i);

inline bool IsPartOfCircumference(cv::Mat& grayScaleImg, int y, int x, int i);

inline double Mu(int p, int q, int x, double xt, int y, double yt);
inline double MuMax(double mu20, double mu02, double mu11);
inline double MuMin(double mu20, double mu02, double mu11);
inline double M(int p, int q, int x, int y);

// Labeling & ethalons
void CalcEthalons(std::vector<FeatureVector>& features, std::map<EClassType, FeatureVector>& ethalons);

void CompareFeaturesWithEthalons(const std::map<EClassType, FeatureVector>& ethalons, const std::vector<FeatureVector>& testFeatures);

// KMeans
void DoClusteringUntilCorrectResult(std::map<FeatureVector, std::vector<FeatureVector>>& clusters, std::vector<FeatureVector>& features);

void ComputeKMeansClustering(std::map<FeatureVector, std::vector<FeatureVector>>& clusters, const std::vector<FeatureVector>& trainFeatures);

void ChooseInitialCentroids(std::vector<FeatureVector>& centroids, const std::vector<FeatureVector>& trainFeatures, int length);

void AssignFeaturesToNearestCentroids(const std::vector<FeatureVector>& trainFeatures, std::vector<FeatureVector>& centroids, std::map<FeatureVector, std::vector<FeatureVector>>& clusters);

void RecalculateCentroids(std::vector<FeatureVector>& tempCentroids, std::vector<FeatureVector>& centroids, std::map<FeatureVector, std::vector<FeatureVector>>& clusters);

void CompareFeaturesWithCentroids(std::map<FeatureVector, std::vector<FeatureVector>>& clusters, std::vector<FeatureVector>& testFeatures);

// Utils
double Euklid(const FeatureVector& feature, const FeatureVector& ethalon);

std::ostream& operator<<(std::ostream& os, const EClassType type);

template<typename T>
bool VectorsEqual(std::vector<T>& v1, std::vector<T>& v2);