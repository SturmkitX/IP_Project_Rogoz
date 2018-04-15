/* openFileDlg() does not exist in linux */
/* int may need to be transformed into unsigned int */

#include <cstdio>
#include <cstdlib>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

const int patchSize = 15;        // unused in the demo, it's the Omega
const int minThreshold = 60;    // T
const float omega = 0.95f;        // must be between 0 and 1 (used for maintaining a certain amount of haze, in order to make a natural image)
const float t0 = 0.1f;          // represents the lower bound of the transmission coefficient

int qsortcomp(const void *a, const void *b)
{
    std::pair<uchar, int> p2 = *(std::pair<uchar, int>*)b;
    std::pair<uchar, int> p1 = *(std::pair<uchar, int>*)a;
    return (p2.second - p1.second);
}

Mat negateImage(Mat src)
{
    Mat res = Mat::zeros(src.rows, src.cols, CV_8UC3);
    for(int i=0; i<src.rows; i++)
    {
        for(int j=0; j<src.cols; j++)
        {
            Vec3b pix = src.at<Vec3b>(i, j);
            pix.val[0] = 255 - pix.val[0];
            pix.val[1] = 255 - pix.val[1];
            pix.val[2] = 255 - pix.val[2];
            res.at<Vec3b>(i, j) = pix;
        }
    }

    return res;
}

uchar minPixel(Mat src, int y, int x)
{
    int offset = patchSize / 2;
    uchar res = 255;

    for(int i=y-offset; i<=y+offset; i++)
    {
        if(i < 0 || i >= src.rows) continue;
        for(int j=x-offset; j<=x+offset; j++)
        {
            if(j < 0 || j >= src.cols) continue;
            Vec3b pix = src.at<Vec3b>(i, j);
            uchar minPix = min(pix.val[0], min(pix.val[1], pix.val[2]));
            res = min(res, minPix);
        }
    }

    return res;
}

Mat computeDarkChannel(Mat src)
{
    Mat res = Mat::zeros(src.rows, src.cols, CV_8UC1);

    for(int i=0; i<src.rows; i++)
    {
        for(int j=0; j<src.cols; j++)
        {
            res.at<uchar>(i, j) = minPixel(src, i, j);
        }
    }

    return res;
}

void computeHistogram(std::pair<uchar, int> *hist, Mat src)
{
    for(int i=0; i<256; i++)
    {
        hist[i].first = i;
        hist[i].second = 0;
    }

    for(int i=0; i<src.rows; i++)
    {
        for(int j=0; j<src.cols; j++)
        {
            uchar pix = src.at<uchar>(i, j);
            hist[pix].second++;
        }
    }
}

void computeTopHistogram(Mat src, std::pair<uchar, int> *histogram, std::pair<uchar, int> *topHistogram)
{
    // select top 0.1% brightest pixels from histogram
    int selectedPixels = src.rows * src.cols / 1000;
    int total_pixels = 0;
    for(int i=0; i<256; i++)
    {
        topHistogram[i].first = i;
        topHistogram[i].second = 0;
    }

    for(int i=255; i>=0; i--)
    {
        topHistogram[i].second = min(histogram[i].second, selectedPixels - total_pixels);
        total_pixels += topHistogram[i].second;
    }
}

Vec3b computeA(uchar value, Mat dark, Mat neg)
{
    for(int i=0; i<dark.rows; i++)
    {
        for(int j=0; j<dark.cols; j++)
        {
            if(dark.at<uchar>(i, j) == value)
            {
                return neg.at<Vec3b>(i, j);
            }
        }
    }

    printf("Returned the default A\n");
    return neg.at<Vec3b>(0, 0);     // it should never reach here
}

float minPixelCoeff(Mat src, int y, int x, Vec3b A)
{
    int offset = patchSize / 2;
    float res = 1.0f;

    for(int i=y-offset; i<=y+offset; i++)
    {
        if(i < 0 || i >= src.rows) continue;
        for(int j=x-offset; j<=x+offset; j++)
        {
            if(j < 0 || j >= src.cols) continue;
            Vec3b pix = src.at<Vec3b>(i, j);
            float valB = (float)pix.val[0] / A.val[0];
            float valG = (float)pix.val[1] / A.val[1];
            float valR = (float)pix.val[2] / A.val[2];
            float minC = min(valB, min(valG, valR));
            res = min(res, minC);
        }
    }

    return res;
}

void computeCoeff(std::vector<float> &coeff, Mat neg, Vec3b A)
{
    for(int i=0; i<neg.rows; i++)
    {
        for(int j=0; j<neg.cols; j++)
        {
            float res = 1.0f - omega * minPixelCoeff(neg, i, j, A);
            coeff.push_back(res);
        }
    }
}

Mat computeRectified(Mat neg, Vec3b A, std::vector<float> &coeff)
{
    Mat res = Mat::zeros(neg.rows, neg.cols, CV_8UC3);
    for(int i=0; i<res.rows; i++)
    {
        for(int j=0; j<res.cols; j++)
        {
            float chosenT = max(coeff[i * res.rows + j], t0);
            Vec3b pix = neg.at<Vec3b>(i, j);
            Vec3b sol = pix;
            sol.val[0] = (uchar)(((float)pix.val[0] - A.val[0]) / chosenT + A.val[0]);
            sol.val[1] = (uchar)(((float)pix.val[1] - A.val[1]) / chosenT + A.val[1]);
            sol.val[2] = (uchar)(((float)pix.val[2] - A.val[2]) / chosenT + A.val[2]);
            res.at<Vec3b>(i, j) = sol;
        }
    }

    return res;
}

int main(int argc, char **argv)
{
    // char fname[100];
    Mat src;
    Mat neg;
    Mat darkChannel;
    Mat rectified;
    Mat enhanced;

    //
    std::pair<uchar, int> histogram[256];
    std::pair<uchar, int> topHistogram[256];
    Vec3b A;
    std::vector<float> coeff;

    // while(openFileDlg(fname))
    // {
    //     src = imread(fname, CV_LOAD_IMAGE_COLOR);
    //     neg = negateImage(src);
    // }


    // src = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    // neg = negateImage(src);
    // darkChannel = computeDarkChannel(neg);
    // computeHistogram(histogram, darkChannel);
    // computeTopHistogram(neg, histogram, topHistogram);
    //
    // qsort(topHistogram, 256, sizeof(std::pair<uchar, int>), qsortcomp);
    // A = computeA(topHistogram[0].first, darkChannel, neg);
    // computeCoeff(coeff, neg, A);
    // rectified = computeRectified(neg, A, coeff);
    // enhanced = negateImage(rectified);

    src = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    darkChannel = computeDarkChannel(src);
    computeHistogram(histogram, darkChannel);
    computeTopHistogram(src, histogram, topHistogram);

    qsort(topHistogram, 256, sizeof(std::pair<uchar, int>), qsortcomp);
    A = computeA(topHistogram[0].first, darkChannel, src);
    computeCoeff(coeff, src, A);
    rectified = computeRectified(src, A, coeff);

    imshow("Source", src);
    // imshow("Negated", neg);
    imshow("Dark channel", darkChannel);
    imshow("Rectified", rectified);
    // imshow("Enhanced", enhanced);

    waitKey();

    printf("A = %u %u %u\n", A.val[2], A.val[1], A.val[0]);
    for(int i=0; i<20; i++)
        printf("%f ", coeff[i]);
    printf("\n");

    float cacat = minPixelCoeff(neg, 0, 0, A);

    printf("%f %f %f\n", cacat, omega * cacat, 1.0f - omega * cacat);

    coeff.clear();
    return 0;
}
