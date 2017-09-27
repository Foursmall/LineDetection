#include <cv.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include<dirent.h>
using namespace std;
using namespace cv;

//#define IPM_GET 1          //模式选择macro

#define DIR_SRC     "/Users/zhaoxuebin/Desktop/clips/src/cordova1"
#define DIR_IPM         "/Users/zhaoxuebin/Desktop/myIPM"



#define DIR_SETUPIMG  "/Users/zhaoxuebin/Desktop/setup.png"


//获取IPM参数后的设置
#define ROIX    181
#define ROIY    210
#define ROIWIDTH  291
#define ROIHEIGHT  165
#define SRCX1  291
#define SRCX2   349

///定义一些IPM阶段用到的窗口名称

const char *winOrigin = "winOrigin";
const char *winROI = "winROI";
const char *winGray = "winGray";
const char *winIPM = "winIPM";
const char *winIPM32 = "winIPM32";

Mat imgOrigin, imgROI, imgGray, imgIPM ,imgIPM32,imgIPM_Inv,imgIPM32_Inv;
  // imgROI 是获取IPM参数的ROI   imgROI2 是车道线识别的小roi，只要占前者的百分之60左右
/// IPM阶段的参数标定

int roiX = 100;        // 几个预设值
int roiY = 100;
int roiWidth = 100;
int roiHeight = 100;
int srcX1 =roiX;
int srcX2 =roiX+roiWidth;
Mat tsfIPM, tsfIPMInv;  //变换矩阵及其  逆矩阵

Rect roiLane = Rect(roiX,roiY,roiWidth,roiHeight);  ///车道线ROI


/**
 * trackbar的事件处理函数
 */
void onROIChange(int _x, void* ptr) {
    /// IPM 仿射变换矩阵
    roiLane.x=roiX;
    roiLane.y=roiY;
    roiLane.width=roiWidth;
    roiLane.height=roiHeight;
    
    
    Point2f src[4];
    Point2f dst[4];
    //图像的坐标原点是左上！  and  默认梯形等腰！！
    //但是此处的模型建立时，以矩形左下为原点，注意区别！
    src[0].x = 1;
    src[0].y = 1;
    src[1].x = srcX1-roiX;
    src[1].y = roiHeight;
    src[2].x = srcX2-roiX;
    src[2].y = roiHeight;
    src[3].x = roiWidth;
    src[3].y = 1;
    
    dst[0].x = 1;
    dst[0].y = 1;
    dst[1].x = 1;
    dst[1].y = roiHeight;
    dst[2].x = roiWidth;
    dst[2].y = roiHeight;
    dst[3].x = roiWidth;
    dst[3].y = 1;
    
    tsfIPM = getPerspectiveTransform(dst, src);
    tsfIPMInv = tsfIPM.inv();
#ifdef IPM_GET
    Mat temp;
    imgOrigin.copyTo(temp) ;
    Point center1 =Point(srcX1,roiY);
    Point center2 =Point(srcX2,roiY);
    circle(temp, center1, 1, Scalar(255,0,0),-1);
    circle(temp, center2, 1, Scalar(255,0,0),-1);
    rectangle(temp, Point(roiLane.x, roiLane.y), Point(roiLane.x + roiLane.width, roiLane.y + roiLane.height), CV_RGB(0, 255, 0));

    
    imshow(winOrigin,temp);

    
    cout << "roiX      "<<roiLane.x<<endl;
    cout << "roiY      "<<roiLane.y<<endl;
    cout << "roiWidth  "<<roiLane.width<<endl;;
    cout << "roiHeight "<<roiLane.height<<endl;
    cout << "srcX1     ("<<center1.x<<","<<center1.y<<" )"<<endl;
    cout << "srcX2     ("<<center2.x<<","<<center2.y<<" )"<<endl;
    cout << "tsfIPM"<<tsfIPM<<"\n";
    cout << "tsfIPMInv"<<tsfIPMInv<<"\n";
#endif
    
}

/**
 * 传入原图，以及 ROI，会在指定目录下生成一系列 ROI 区域的图片
 */
void cutRegion(Mat &imgInput, Rect _roi, const char* dir ) {
    static int i = -1;
    char path[1024] = {0};
    
    
    if (_roi.x < 0 || _roi.x >= imgInput.cols) {
        _roi.x = 0;
    }
    if (_roi.y < 0 || _roi.y >= imgInput.rows) {
        _roi.y = 0;
    }
    if (_roi.x + _roi.width >= imgInput.cols) {
        _roi.width = imgInput.cols - 1 - _roi.x;
    }
    if (_roi.y + _roi.height >= imgInput.rows) {
        _roi.height = imgInput.rows - 1 - _roi.y;
    }
    
    
    snprintf(path, sizeof(path) - 1, "%s/%06d.png", dir, ++i);
    imwrite(path, Mat(imgInput, _roi));
}

/**
 *  传入目录，得到所有该目录下的文件名称   
 * NOTE : 第一个“.” 当前目录
 ＊       第二个“..” 父目录
 **/
Vector<string> getAllFilesName(string dirname) {
    Vector<string>  allfilesname ;
    string filename;
    DIR *dp;
    struct dirent *dirp;
    if((dp = opendir(dirname.c_str())) == NULL)
        cout << "Can't open " << dirname << endl;
    
    while((dirp = readdir(dp)) != NULL){
        filename =dirname+"/"+dirp->d_name;
        allfilesname.push_back(filename);
    }
    closedir(dp);
    
    return allfilesname;
}

int main( int argc, char** argv )
{
 
#ifndef IPM_GET
    ////＃＃＃＃＃＃＃＃获取参数IPM矩阵之后设置如下＃＃＃＃＃＃＃＃
    /// IPM参数获取时注意注释之
    roiX= ROIX;
    roiY= ROIY;
    roiWidth= ROIWIDTH;
    roiHeight= ROIHEIGHT;
    srcX1= SRCX1;
    srcX2= SRCX2;
    onROIChange(0, 0);//设置tsfIPM  tsfIPMInv 值
    
    cout<<tsfIPM<<endl;
#endif
    
#ifdef IPM_GET
    
    imgOrigin = imread(DIR_SETUPIMG, 1 );
    
    
    
    //四个窗口：原图    RIO    灰度图    IPM
    namedWindow(winOrigin, CV_WINDOW_KEEPRATIO | CV_WINDOW_NORMAL);
    namedWindow(winROI, CV_WINDOW_KEEPRATIO | CV_WINDOW_NORMAL);
    namedWindow(winGray, CV_WINDOW_KEEPRATIO | CV_WINDOW_NORMAL);
    namedWindow(winIPM, CV_WINDOW_KEEPRATIO | CV_WINDOW_NORMAL);
    namedWindow(winIPM32,CV_WINDOW_KEEPRATIO | CV_WINDOW_NORMAL);

    //用于自主设置ROI
    //在win上面创建trackbar，可以设置回调函数
    createTrackbar("ROI width", winOrigin, &roiWidth, imgOrigin.cols, onROIChange);
    createTrackbar("ROI height", winOrigin, &roiHeight, imgOrigin.rows, onROIChange);
    createTrackbar("ROI y", winOrigin, &roiY, imgOrigin.cols, onROIChange);
    createTrackbar("ROI x", winOrigin, &roiX, imgOrigin.rows, onROIChange);
    createTrackbar("srcX1", winOrigin, &srcX1, imgOrigin.cols,onROIChange);
    createTrackbar("srcX2", winOrigin, &srcX2, imgOrigin.rows,onROIChange);
    ///参数获取时的代码
    Mat  temp;
    imgOrigin.copyTo(temp);
    imgROI= Mat(temp,roiLane);
    rectangle(temp, Point(roiLane.x, roiLane.y), Point(roiLane.x + roiLane.width, roiLane.y + roiLane.height), CV_RGB(0, 255, 0));
    imshow(winOrigin, temp);
    
   
#endif
    
    
#ifndef IPM_GET
    int   indexFrame = 1;
    string dirFrame;
    Vector<string> allfilesname;
    allfilesname = getAllFilesName(DIR_SRC);
    while (indexFrame<allfilesname.size()-2) {
//       cout<<indexFrame<<endl;
        dirFrame =allfilesname[indexFrame+2];
//        cout<<dirFrame<<endl;
        indexFrame++;
        
        
        
        //递归处理所有图像
        imgOrigin = imread(dirFrame);
        
        cvtColor(imgOrigin, imgGray, CV_RGB2GRAY);
        
//        imgROI =imgGray(roiLane,CV_8U);
        imgROI=Mat(imgGray,roiLane);

        /// 灰度图的IPM 图
        warpPerspective(imgROI, imgIPM, tsfIPM, imgGray.size());
        
        cutRegion(imgIPM, Rect(0, 0, imgROI.cols, imgROI.rows), DIR_IPM);

        
        
      
    }
     allfilesname.clear();
    
#endif
   

    waitKey(0);
    return 0;
}
    
   
