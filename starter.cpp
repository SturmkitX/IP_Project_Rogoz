/* openFileDlg() does not exist in linux */

#include <cstdio>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

const int patchSize = 1;        // unused in the demo, it's the Omega
const int minThreshold = 20;    // T

struct compclass {
  bool operator() (std::pair<Point2i, uchar> p1, std::pair<Point2i, uchar> p2)
  {
      return (p2.second > p1.second);
  }
} compobject;

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

Mat computeDarkChannel(Mat src)
{
    Mat res = Mat::zeros(src.rows, src.cols, CV_8UC1);
    for(int i=0; i<src.rows; i++)
    {
        for(int j=0; j<src.cols; j++)
        {
            Vec3b pix = src.at<Vec3b>(i, j);
            res.at<uchar>(i, j) = min(min(pix.val[0], pix.val[1]), pix.val[2]);
        }
    }

    return res;
}

void computeHistogram(std::vector<std::pair<Point2i, uchar> > &hist, Mat src)
{
    // uint *res = (uint*) calloc(256, sizeof(uint));
    for(int i=0; i<src.rows; i++)
    {
        for(int j=0; j<src.cols; j++)
        {
            std::pair<Point2i, uchar> pix = std::make_pair(Point2i(j, i), src.at<uchar>(i, j));
            hist.push_back(pix);
        }
    }
}

uchar computeA(std::vector<std::pair<Point2i, uchar> > hist, Mat src)
{
    Point2i maxIntensity = hist[0].first;
    Vec3b res = src.at<Vec3b>(maxIntensity.y, maxIntensity.x);

    return res;
}

void computeT(std::vector<float> &coeff, Mat neg, uchar A)
{
    for(int i=0; i<neg.rows; i++)
    {
        for(int j=0; j<neg.cols; j++)
        {
            uchar p1 = neg.at<uchar>(i, j);
            float dest;

            dest = 1.0f - (float)p1 / A;
            coeff.push_back(dest);
        }
    }
}

Mat computeRectified(Mat neg, std::vector<float> coeff, uchar A)
{
    std::vector<float>::iterator it = coeff.begin();
    Mat res = Mat::zeros(neg.rows, neg.cols, CV_8UC3);

    for(int i=0; i<neg.rows; i++)
    {
        for(int j=0; j<neg.cols; j++)
        {
            Vec3b pix = neg.at<Vec3b>(i, j);
            Vec3b dest = pix;

            dest.val[0] = (uchar)((float)(pix.val[0] - A) / (*it) + A);
            dest.val[1] = (uchar)((float)(pix.val[1] - A) / (*it) + A);
            dest.val[2] = (uchar)((float)(pix.val[2] - A) / (*it) + A);
            res.at<Vec3b>(i, j) = dest;
            it++;

        }
    }
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
    std::vector<std::pair<Point2i, uchar> > histogram;
    uchar A;
    std::vector<float> coeff;

    // while(openFileDlg(fname))
    // {
    //     src = imread(fname, CV_LOAD_IMAGE_COLOR);
    //     neg = negateImage(src);
    // }
    src = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    neg = negateImage(src);
    darkChannel = computeDarkChannel(neg);
    computeHistogram(histogram, darkChannel);
    std::sort(histogram.begin(), histogram.end(), compobject);
    // A = computeA(histogram, neg);
    A = histogram[0].second;
    computeT(coeff, neg, A);
    rectified = computeRectified(neg, coeff, A);
    enhanced = negateImage(rectified);

    imshow("Original", src);
    imshow("Negated", neg);
    imshow("Enhanced", enhanced);

    waitKey();
    // free(histogram);
    histogram.clear();
    coeff.clear();

    return 0;
}
