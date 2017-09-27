//
//  main.cpp
//  LaneDetection
//
//  Created by 赵学斌 on 16/7/6.
//  Copyright (c) 2016年 Foursmall. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cvaux.hpp>
#include <fstream>
#include <dirent.h>
#include <cstdlib>
using namespace std;
using namespace cv;

#define CODELINE   cout<<endl<< "Code in "<<__LINE__<<endl;
#define DIR_IPM  "/Users/zhaoxuebin/Desktop/IPM"

/**
 *  程序运行之前必须设置好的参数（简称“必设项”）
 **/
Mat tsfIPM, tsfIPMInv;  //变换矩阵及其逆矩阵  ,程序运行之前必须设定好该Mat值

int roiX;
int roiY;
int roiWidth;
int roiHeight;
Rect roi = Rect(roiX,roiY,roiWidth,roiHeight);  ///车道线ROI


/**
 *  quantile  lineThreshol的值是设置是很讲究的，目前我是通过经验来设置的（修改值后查看hist 图）两者得配合好！
 **/
int  quantile  = 15 ;   // gaussian filter  之后的微小像素值的pixel 过滤，  threshold 值 filter后低于灰度值低于quantile值的点将被置零
  // 该值的设置会直接影响到 直线位置判断时的hist图，若小的话，hist中会出现很多的微小值

int lineThreshold=20;   //filter图中直线识别的阈值，>lineThreshold 才能被认可

Mat gaussianKernelX ;  //高斯核的设置
Mat gaussianKernelY;

float derivp[] =  {1.000000e-16, 1.280000e-14, 7.696000e-13, 2.886400e-11, 7.562360e-10, 1.468714e-08, 2.189405e-07, 2.558828e-06, 2.374101e-05, 1.759328e-04, 1.042202e-03, 4.915650e-03, 1.829620e-02, 5.297748e-02, 1.169560e-01, 1.918578e-01, 2.275044e-01, 1.918578e-01, 1.169560e-01, 5.297748e-02, 1.829620e-02, 4.915650e-03, 1.042202e-03, 1.759328e-04, 2.374101e-05, 2.558828e-06, 2.189405e-07, 1.468714e-08, 7.562360e-10, 2.886400e-11, 7.696000e-13, 1.280000e-14, 1.000000e-16};
int derivLen = 33;

float smoothp[] =  {-1.000000e-03, -2.200000e-02, -1.480000e-01, -1.940000e-01, 7.300000e-01, -1.940000e-01, -1.480000e-01, -2.200000e-02, -1.000000e-03};
int smoothLen = 9;

const char *winOrigin = "winOrigin";
const char *winIPM = "winIPM";
const char *winFilter ="winFilter";
const char *winIPMLine ="winIPMLine";
Mat imgOrigin, imgROI, imgIPM3C,imgIPM,imgFilter,imgBinary,imgIPMLine,imgBinary_scr;
//imgOrigin : src
//imgROI    :
//imgIPM    : CV_8U  IPM
//imgFilter : guassian 过滤后的图
//imgBinary : imgFiler图中把pixel值>=quantile 的值置1，否则置0 。形成的二值图
//直线的结构体    x1  x2   分别表示直线在roi上下边界的坐标

typedef struct Line {
    int  x1;
    int  x2;
} LaneLineNode;


/**
 *  传入目录，得到所有该目录下的文件名称
 * NOTE : 第一个“.” 当前目录
 *       第二个“..” 父目录
 **/
Vector<string> getAllFilesName(const string dirname) {
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




/**
 * 传入原图，以及 ROI，会在指定目录下生成一系列 ROI 区域的图片
 */
void cutRegion(const Mat &imgInput,  Rect _roi, const char* dir ) {
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
 *  根据预定的   tsfIPM  值来获取 基于 roi 的gray 图
 * img :  最原始的src
 **/
Mat  getIPM (const Mat &img ){
    return img;
}

/**
 *  Gaussian Filter  高斯核的设置属于必设项
 *  img :  gray 、基于 roi的 IPM图
 **/
Mat getFilter(const  Mat &img)   {
    //高斯核的初始化
    Mat img_filter;
    gaussianKernelX =Mat(derivLen, 1, CV_32FC1, derivp);
    gaussianKernelY =Mat(1, smoothLen,CV_32FC1, smoothp);
    ///

    filter2D(img, img_filter, -1, gaussianKernelX);
    filter2D(img_filter, img_filter, -1, gaussianKernelY);
    return img_filter;
}



/**  
 *  阈值化
 *  img：高斯filter之后的图
 **/
Mat getThreshold(const Mat &img ){
    return img;
}



/**
 *  获取车道线直线的大概位置  Local  max location for a detected line
 * 方法： 每一列相加的值
 * img :高斯filter后的灰度Mat
 * point ： 识别出来的垂直直线的坐标
 **/
void getLineLocation ( const Mat &img,vector<int> & lineScore,vector<int> & lineLocation){

    threshold(img, imgBinary_scr, 0,1, CV_THRESH_BINARY);
    
    threshold(img, img, quantile, 255, THRESH_TOZERO  );
//    cout<<"去除像素值低于quantile 的噪点后： "<<endl;
//    cout<<img<<endl;
   
    
    threshold(img, imgBinary, 0, 1, CV_THRESH_BINARY);
  
    Mat sumpoint;
    reduce(imgBinary, sumpoint, 0, CV_REDUCE_SUM,CV_32S);      //     把矩阵reduce 成一row，每一列的总和
  
//    CODELINE
//    cout<<sumpoint<<endl;

    
    //**********  画出统计直方图，并观察结果
    Mat hist (160, sumpoint.cols, CV_8U,Scalar(0));
    for(int i=0;i<hist.cols;i++)
    {
        if (sumpoint.at<int>(0,i)>lineThreshold)
         line(hist, Point(i,160), Point(i,160-sumpoint.at<int>(0,i)), Scalar(255));    //*******此处特别注意
    }                    //(0,3*i)的用法，至于原因一直没有搞懂为什么得乘3
    
//    imshow("histgram", hist);
    ///

    
    for(int i=0; i<sumpoint.cols; i++)
    {
                //get that value
                int val = sumpoint.at<int>(0,i);

                if( val > lineThreshold)
                {
                    lineScore.push_back(val);
                    lineLocation.push_back(i);
                    
                }
    }
    
//    CODELINE
//    for (int i=0; i<lineScore.size(); i++) {
//        cout<<"lineScore :" <<lineScore[i];
//        cout<<"       location  :"<<lineLocation[i]<<endl;
//    }
    
    
    // 合并相邻的直线
    vector<int> ::iterator i,j;

    int _score;
    int _location;

    if (lineLocation.size()==0) {
        return ;
    }
    
    for (i=lineLocation.begin(),j=lineScore.begin() ; i!=lineLocation.end()-1; i++,j++) {
        if (abs(*i - *(i+1))<6)
        {

                _location = (*i> *(i+1)?   (int) (*(i+1) + (*i- *(i+1))*(*j)/(*j+*(j+1))) : (int)(*i + (*(i+1)- *i)*(*(j+1))/(*j+*(j+1)))  );
                _score =(*i>*(i+1) ?*j:*(j+1));
                lineLocation.erase(i);
                lineLocation.erase(i);
                lineScore.erase(j);
                lineScore.erase(j);
                lineLocation.insert(i, _location);
                lineScore.insert(j, _score);
                i--;j--;
            }
        }
    
    
//    cout<<"+++++++++++"<<endl;
//    for (int i=0; i<lineScore.size(); i++) {
//        cout<<"lineScore :" <<lineScore[i];
//        cout<<"       location  :"<<lineLocation[i]<<endl;
//    }
    
    
       //**********  画出直线合并后的统计直方图，并观察结果
    
    Mat hist_reduce (160, sumpoint.cols, CV_8U,Scalar(0));
    
    for (int i=0; i<lineLocation.size(); i++) {
        line(hist_reduce, Point(lineLocation[i],160), Point(lineLocation[i],160-lineScore[i]), Scalar(255));
    }
    
    
//    imshow("histgram_reduce", hist_reduce);
    
    
    
    
}


/**
 *
 **/
void getHVLine(){
    
    
}



/**
 *  一条直线的拟合
 *  img : imgBinary
 *  point : getLineLocation() 之后得到的 一条直线的坐标
 *  line  : 拟合直线的上下坐标点，也就是两个int值
 *  width :  直线拟合区域，也就是垂直直线左右偏移多少像素点的区域内
 **/
void getLineFit( const Mat &img,vector<int> &line , int point ,int width) {
    Mat float_img;        //  opencv 库只有float型数据可以直接使用at<T>（）读取
    img.convertTo(float_img, CV_32FC1);
    int x1,x2,y1,y2,m,n;
    float score,bestScore;
    float k;
    int best_m,best_n;       int temp; float temp1;
    x1= max(point-width,0);
    x2 =min(point+width,img.cols);
    score =0;
    bestScore =0;
    y1 =0;
    y2 =img.rows-1;
    
//    CODELINE
    for(m=x1;m<=x2;m++)
        for (n=x1; n<=x2; n++) {
            k=(n-m)/(y2-y1+1);
            score =0;
            for (int y=y1; y<=y2; y++) {
                temp =int(m+y*k);
                temp1 =float_img.at<float>(y,temp);
                score+= temp1 ;
              
            }
//            cout<<"( "<<m<<"  ,  "<<n<<" )";
//            cout<<score<<endl;
            if (score>=bestScore) {
                bestScore=score;
                best_m=m;
                best_n=n;
            }
            
        }
    line.push_back(best_m);
    line.push_back(best_n);
    
}



/**
 *   多条直线的拟合
 *  img : imgBinary
 *  points : getLineLocation() 之后得到的 数组
 *  width :  直线拟合区域，也就是垂直直线左右偏移多少像素点的区域内
 *  lines :  多条拟合直线，表达形式是：X1,_X1, X2,_X2,X3,_X3 ......
 **/
void getLinesFit(const Mat &img,vector<int> &lines,vector<int> &points,int width)
{
    vector<int> line;
    for (int i=0; i<points.size(); i++) {
        getLineFit(img, line, points[i], width );
        lines.push_back(line[0]);
        lines.push_back(line[1]);
        line.clear();
    }
}


/**
 *   在IPM 画出 初步识别的 垂直直线
 *  img ： IPM 图
 *  lineLocation : 直线坐标集合
 **/
void   drawLinesatIPM  ( Mat & img ,const  vector<int> & lineLocation)
{
    for (int i=0; i<lineLocation.size(); i++) {
        line(img, Point(lineLocation[i],0), Point(lineLocation[i],160), Scalar(0));
    }

}
/**  在IPM上画出   拟合直线
 *   img: IPM图
 *   lines：要画的直线集合，一条直线的给出形式是两个int 坐标 ，多条直线是连续存入lines中的
 **/
void  drawFitLinesatIPM (Mat &img,const vector<int> &lines)
{
    for (int i=0; i<lines.size()/2; i++) {
        line(img, Point(lines[2*i],0), Point(lines[2*i+1],160), Scalar(255));
    }

}


/**
 *    对一张图片的车道线的提取
 *  dir :  3通道的IPM  image
 **/
void processImg(string  dir , bool show_imgIPM ,bool show_imgFilter,bool show_imgIPMLine)
{
    
    imgIPM3C = imread(dir);
    cvtColor(imgIPM3C, imgIPM, CV_BGR2GRAY );
    if (show_imgIPM ) {
         imshow(winIPM, imgIPM);
    }
    
    //根据预定的   tsfIPM  值来获取 基于 roi 的gray 图
    imgFilter=  getFilter(imgIPM);
    
    if (show_imgFilter) {
        imshow(winFilter, imgFilter);
    }
    
    vector<int>  lineLocation;
    vector<int>  lineScore;
    
    
    
    // 获取车道线直线的大概位置
    getLineLocation(imgFilter, lineScore,lineLocation);
    
    //显示识别出来的车道线的大概位置，便于直线拟合的
    CODELINE
    cout<<"一共  "<<lineLocation.size()<<"  条直线  ："<<endl;
    for (int i=0; i<lineLocation.size(); i++) {
        cout<<"Location:  "<<lineLocation[i]<<"  score  ("<<lineScore[i]<<")"<<endl;
    }
    
    
    
    imgIPM.copyTo(imgIPMLine);
    drawLinesatIPM(imgIPMLine, lineLocation);
    
    if (show_imgIPMLine) {
        imshow(winIPMLine, imgIPMLine );
    }


}

// main()    单张图片的测试



int main(int argc, const char * argv[])
{
    
    processImg("/Users/zhaoxuebin/Desktop/IPM/000099.jpg",true,true,true);
    
    
    
    
    
////    单条直线拟合的测试   
//    vector<int> fitline;
//  
//    getLineFit(imgBinary, fitline, 53, 4);
//    
//    CODELINE
//    cout<<" 拟合出来的直线 ："<<endl;
//    cout<<fitline[0]<<endl;
//    cout<<fitline[1]<<endl;
//

    
//
//
//
//    多条直线的拟合
//    vector<int> fitLines;
//    getLinesFit(imgBinary_scr, fitLines, lineLocation, 3);    //值5 之后的程序版本需要设置成macro ，值的选择需要根据实际情况
//    cout<<"FitLines :"<<endl;
//    for (int i=0; i<fitLines.size()/2; i++) {
//        cout<<fitLines[2*i] <<"     "<<fitLines[2*i+1] <<endl;
//    }
//    drawFitLinesatIPM(imgIPMLine, fitLines);
//    imshow("winFitLines", imgIPMLine);
//

    
    
    waitKey();
    return 0;
}

 


/*
// main()    多张图片的测试
int main()
{
    int indexFrame = 1;
    string  dirFrame;
    Vector<string> allfilesname;
    int waittime =int(1000/15);    //每张图片处理之后停留的时间
    allfilesname = getAllFilesName(DIR_IPM);
    while (indexFrame<allfilesname.size()-2) {
        cout<<indexFrame<< "   :";
        dirFrame =allfilesname[indexFrame+2];
        cout<<dirFrame<<endl;
        indexFrame++;
        
        
        //****可以递归处理  图片集合了
        processImg(dirFrame, false, false, true);
        

        
        waitKey(waittime);

    }
    
    
    
    
    allfilesname.clear();
    return -1;
    
}

*/







