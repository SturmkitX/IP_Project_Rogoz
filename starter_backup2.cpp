/* openFileDlg() does not exist in linux */
/* int may need to be transformed into unsigned int */

#include <cstdio>
#include <cstdlib>
#include <queue>
#include <cfloat>
#include <functional>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

class compstruct
{
public:
    bool operator() (std::pair<Point2i, uchar> p1, std::pair<Point2i, uchar> p2)
    {
        return p2.second > p1.second;
    }
};

const int patchSize = 15;        // unused in the demo, it's the Omega
const float omega = 0.95f;        // must be between 0 and 1 (used for maintaining a certain amount of haze, in order to make a natural image)
const float t0 = 0.1f;          // represents the lower bound of the transmission coefficient

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

Vec3b computeA(Mat src, Mat dark)
{
    std::priority_queue<std::pair<Point2i, uchar>, std::vector<std::pair<Point2i, uchar> >, compstruct > topVals;
    int total[3];
    int number = dark.rows * dark.cols / 1000;

    total[0] = total[1] = total[2];

    for(int i=0; i<dark.rows; i++)
    {
        for(int j=0; j<dark.cols; j++)
        {
            uchar pix = dark.at<uchar>(i, j);
            if(topVals.size() < number)
            {
                topVals.push(std::pair<Point2i, uchar>(Point2i(j, i), pix));
            }
            else
            {
                if(pix > topVals.top().second)
                {
                    topVals.pop();
                    topVals.push(std::pair<Point2i, uchar>(Point2i(j, i), pix));
                }
                else continue;
            }
        }
    }

    while(!topVals.empty())
    {
        Point2i point = topVals.top().first;
        topVals.pop();
        Vec3b pix = src.at<Vec3b>(point.y, point.x);
        total[0] += pix.val[0];
        total[1] += pix.val[1];
        total[2] += pix.val[2];
    }

    Vec3b res;
    res.val[0] = total[0] / number;
    res.val[1] = total[1] / number;
    res.val[2] = total[2] / number;

    return res;
}

Vec3b minPixelVec3(Mat src, int y, int x)
{
    int offset = patchSize / 2;
    Vec3b res;
    res.val[0] = res.val[1] = res.val[2] = 255;

    for(int i=y-offset; i<=y+offset; i++)
    {
        if(i < 0 || i >= src.rows) continue;
        for(int j=x-offset; j<=x+offset; j++)
        {
            if(j < 0 || j >= src.cols) continue;
            Vec3b pix = src.at<Vec3b>(i, j);

            res.val[0] = min(res.val[0], pix.val[0]);
            res.val[1] = min(res.val[1], pix.val[1]);
            res.val[2] = min(res.val[2], pix.val[2]);
        }
    }

    return res;
}

Mat computeRectified(Mat neg, Mat dark, Vec3b A, Mat &tmat)
{
    Mat resf = Mat::zeros(neg.rows, neg.cols, CV_32FC3);
    Mat res = Mat::zeros(neg.rows, neg.cols, CV_8UC3);
    tmat = Mat::zeros(neg.rows, neg.cols, CV_32FC1);
    Vec3f sol;
    float maxR=FLT_MIN, minR=FLT_MAX, maxG=FLT_MIN, minG=FLT_MAX, maxB=FLT_MIN, minB=FLT_MAX;
    for(int i=0; i<res.rows; i++)
    {
        for(int j=0; j<res.cols; j++)
        {
            Vec3b tmp = minPixelVec3(neg, i, j);
            float t1 = (float)tmp.val[0] / A.val[0];
            float t2 = (float)tmp.val[1] / A.val[1];
            float t3 = (float)tmp.val[2] / A.val[2];
            float chosenT = 1.0f - omega * min(t1, min(t2, t3));
            tmat.at<float>(i, j) = chosenT;

            Vec3b pix = neg.at<Vec3b>(i, j);
            sol = pix;
            if (chosenT  <  0.1) chosenT = 0.1;


            sol.val[0] =  ((float)pix.val[0] - A.val[0]) / chosenT + A.val[0];
            sol.val[1] = ((float)pix.val[1] - A.val[1]) / chosenT + A.val[1];
            sol.val[2] = ((float)pix.val[2] - A.val[2]) / chosenT + A.val[2];
            minR = minR < sol.val[2] ? minR : sol.val[2];
            minG = minG < sol.val[1] ? minG : sol.val[1];
            minB = minB < sol.val[0] ? minB : sol.val[0];

            maxR = maxR > sol.val[2] ? minR : sol.val[2];
            maxG = maxG > sol.val[1] ? maxG : sol.val[1];
            maxB = maxB > sol.val[0] ? maxB : sol.val[0];

            resf.at<Vec3f>(i, j) = sol;
        }
    }

    printf("MinR = %f\n", minR);
    printf("MaxR = %f\n", maxR);
    printf("MinG = %f\n", minG);
    printf("MaxG = %f\n", maxG);
    printf("MinB = %f\n", minB);
    printf("MaxB = %f\n", maxB);

    float maxf = maxR;
    maxf = maxf > maxB ? maxf : maxB;
    maxf = maxf > maxG ? maxf : maxG;

    float minf = minR;
    minf = minf < minB ? minf : minB;
    minf = minf < minG ? minf : minG;

    Vec3b pixel;
    for(int i=0; i<res.rows; i++)
    {
        for(int j=0; j<res.cols; j++)
        {
            sol = resf.at<Vec3f>(i,j);
            pixel.val[0] = (sol.val[0]-minf)/(maxf-minf) * 255;
            pixel.val[1] = (sol.val[1]-minf)/(maxf-minf) * 255;
            pixel.val[2] = (sol.val[2]-minf)/(maxf-minf) * 255;
            res.at<Vec3b>(i,j) = sol;
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
    Mat tmat;

    //
    Vec3b A;

    // while(openFileDlg(fname))
    // {
    //     src = imread(fname, CV_LOAD_IMAGE_COLOR);
    //     neg = negateImage(src);
    // }


    src = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    neg = negateImage(src);
    darkChannel = computeDarkChannel(neg);

    A = computeA(neg, darkChannel);
    rectified = computeRectified(neg, darkChannel, A, tmat);
    enhanced = negateImage(rectified);

    imshow("Source", src);
    // imshow("Negated", neg);
    imshow("Dark channel", darkChannel);
    // imshow("Rectified", rectified);
    imshow("Transmission", tmat);
    imshow("Enhanced", enhanced);

    // imwrite("output.jpg", enhanced);

    waitKey();

    return 0;
}
