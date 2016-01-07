// OpenCVproject.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//
#include "stdafx.h"
//#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <unordered_map>
#include <stdio.h>
#include "stdlib.h"

using namespace cv;
using namespace std;

int hough();
int sovel();
int laplacian();
int mean();
int hist();
int canny();
int canny_hough();
int ellipse();

class HashVI{
public:
	size_t operator()(const vector<int> &x) const {
		const int C = 997;
		size_t t = 0;
		for(int i = 0; i != x.size();++i){
			t = t * C + x[i];
		}
		return t;
	}
};


int _tmain(int argc, _TCHAR* argv[])
{
	int i,j,h,randIndex;
	double p,minWcv;
	const int nStart = 5;
	const int k = 3;    //クラスタ数
    Vector<Point> data;     //データ群
    double kCenterX[k];     //クラスタセンターのX座標
	double kCenterY[k];
	Vector<int> clsLabel;          //データ点がどのクラスタに属しているか
	int nData;
	Point cp, dp;
	int dx,dy, cx,cy,minIndex;
	double dis,disMin;
    double clsCount[k];            //クラスタ内のデータ数
	double totalX[k];         
	double totalY[k];
	//----------元画像-----------------------------------------
	Mat src = imread("img_thermal_3.jpg", IMREAD_GRAYSCALE);
	Mat image1(Size(600,600),CV_8UC3);
	resize(src, image1,image1.size(),0,0,INTER_LINEAR);
/*	Mat resizeThermal(image1.rows*0.5,image1.cols*0.5,image1.type());
	resize(dishes, resizeDishes,Size(),0.5,0.5);
	namedWindow("dishes");
	imshow("dishes",resizeDishes);
	*/
	namedWindow("元画像");
	imshow("元画像", image1);
	//----------二値画像（大津）-------------------------------
	Mat image2;
	threshold(image1, image2, 90, 255, THRESH_BINARY);
	namedWindow("output");
	imshow("output", image2);
	//----------クラスタリング---------------------------------
	const int width = image2.cols;
	const int height = image2.rows;
	bool changed = false;
	for(j= 0;j < height; j++){
		for(i = 0;i < width;i++){
		    p = static_cast<int>(image2.at<unsigned char>(j,i));
			if(p == 255){
			}
			else{
				Point p(j,i);
				data.push_back(p);
		//		printf("x:%d, y:%d",j,i);
			}	
		}
	}
	nData = data.size();
//---------マルチスタート-----------------
    minWcv = 100000000;
	int bestIterCount = 0;
	double bestCenterX[k];
	double bestCenterY[k];
	Mat images[nStart];
	for(h = 0 ; h < nStart; h++){
//---------初期のクラスタセンタ―を設定------------
	for(i = 0; i < k;i++){
		randIndex = (int)rand()%(nData+1);
	//	printf("[%d]%d",h,randIndex);
		kCenterX[i] = (int)data[randIndex].x;
		kCenterY[i] = (int)data[randIndex].y;
	}

//クラスタセンターが変化しなくなるまで繰り返す
	int iterCount = 0;
	do{
//----------クラスタ割り当て--------------------------------- 
		changed = false;
		for(i = 0; i < k;i++){         //初期化
    		clsCount[i] = 0;
    		totalX[i] = 0;
    		totalY[i] = 0;
	    }
		clsLabel.clear();
//----------^各データ点とクラスタセンター間との距離を計算------------------------------
     	for(i = 0; i < nData;i++){ //1つずつデータ点を取得
	    	dp = data[i];
     		dx = dp.x;
    		dy = dp.y;
    		disMin = 100000000;
    		for(j = 0; j < k;j++){    //クラスタセンターの座標を取得
	    		cx = kCenterX[j];
	    		cy = kCenterY[j];
	    		dis = sqrt((dx-cx)*(dx-cx)+(dy-cy)*(dy-cy));
	    		if(dis != 0){
	    	    	if(dis < disMin){
	    	    		disMin = dis;
						minIndex = j;
        			}
	    		}else{
	    			minIndex = j;
	    			break;
    			}
    		}
			clsLabel.push_back(minIndex);        //データ点ごとにクラスタを割り振る
			totalX[minIndex] += dx;              //クラスタのX座標の合計を計算(index自体がクラスタを表している)
			totalY[minIndex] += dy;              //    ''    Y座標の合計を計算
			clsCount[minIndex]++;                //クラスタごとの要素数を計算
    	}
//-----------新しいクラスタセンタ―を得る------------------------------
  //------------クラスタ内の平均値の算出---------------------------
		int countMatch = 0;
		double meanX[k];
		double meanY[k];
		for(i = 0; i < k;i++){
			meanX[i] = totalX[i]/clsCount[i];
			meanY[i] = totalY[i]/clsCount[i];
			if(meanX[i] == kCenterX[i] && meanY[i] == kCenterY[i]){
			     countMatch++;
			}
			kCenterX[i] = meanX[i];
			kCenterY[i] = meanY[i];
			if(h == 1){
//			printf("X2:%lf,",kCenterX[i]);
//		    printf("Y2:%lf,",kCenterY[i]);
			}
		}
  //-------------新しいクラスタセンタが同じ点であるかどうかの判定-------------
//		printf("coutMatch:%d,",countMatch);
		if(countMatch == k){
			changed = false;
		}else{
			changed = true;
		}
		iterCount++;
	}while(changed);
    
	//wcvの計算
	double var = 0.0;
	int label;

	for(int j = 0; j < data.size();j++){
		Point p = data[j];
		label = clsLabel[j];
		Point q((int)kCenterX[label], (int)kCenterY[label]);
		var += sqrt((p.x-q.x)*(p.x-q.x)+(p.y-q.y)*(p.y-q.y));
	}
	double wcv = var / data.size();
//	printf("wcv:%lf",wcv);
	if(wcv < minWcv){
		minWcv = wcv;
		bestIterCount = iterCount;
		for(i = 0 ; i < k;i++){
    		bestCenterX[i] = kCenterX[i];
			bestCenterY[i] = kCenterY[i];
	    }  
	}
/*	
	Mat newImg = Mat(Size(width,height),CV_8UC3);
	cvtColor(image2,newImg,CV_GRAY2BGR);
	for(i= 0;i < data.size(); i++){
		if(clsLabel[i] == 0){
			rectangle(newImg,Point(data[i].y,data[i].x),
				Point(data[i].y,data[i].x),Scalar(255,0,0),1,1);
		}else if(clsLabel[i] == 1){
			rectangle(newImg,Point(data[i].y,data[i].x),
				Point(data[i].y,data[i].x),Scalar(0,255,0),1,1);
		}else{
			rectangle(newImg,Point(data[i].y,data[i].x),
				Point(data[i].y,data[i].x),Scalar(0,0,255),1,1);
		}
	}
	for(i = 0; i < k;i++){
		circle(newImg,Point(kCenterX[i],kCenterY[i]),1,Scalar(0,0,0));
	}
	images[h] = newImg;
*/	
	}

//********************最終結果表示*********************

	do{
//----------クラスタ割り当て--------------------------------- 
		changed = false;
		clsLabel.clear();
		for(i = 0; i < k;i++){         //初期化
    		clsCount[i] = 0;
    		totalX[i] = 0;
    		totalY[i] = 0;
	    }
//		printf("A");
     	for(i = 0; i < nData;i++){ //1つずつデータ点を取得
//			printf("B");
	    	dp = data[i];
     		dx = dp.x;
    		dy = dp.y;
    		disMin = 100000000;
	//		printf("C");
    		for(j = 0; j < k;j++){    //クラスタセンターの座標を取得
	//    		printf("D");
	    		cx = bestCenterX[j];
	    		cy = bestCenterY[j];
	    		dis = sqrt((dx-cx)*(dx-cx)+(dy-cy)*(dy-cy));
//				printf("E");
	    		if(dis != 0){
	//				printf("F");
	    	    	if(dis < disMin){
	//					printf("G");
	    	    		disMin = dis;
						minIndex = j;
        			}
	    		}else{
	//				printf("H");
	    			minIndex = j;
	    			break;
    			}
    		}
	//		printf("I");
			clsLabel.push_back(minIndex);       //index自体がクラスタを表している
			totalX[minIndex] += dx;
			totalY[minIndex] += dy;
			clsCount[minIndex]++;
    	}
//-----------新しいクラスタセンタ―を得る------------------------------
  //------------クラスタ内の平均値の算出---------------------------
		int countMatch = 0;
		double meanX[k];
		double meanY[k];
		for(i = 0; i < k;i++){
			meanX[i] = totalX[i]/clsCount[i];
			meanY[i] = totalY[i]/clsCount[i];
			if(meanX[i] == bestCenterX[i] && meanY[i] == bestCenterY[i]){
			     countMatch++;
			}
			bestCenterX[i] = meanX[i];
			bestCenterY[i] = meanY[i];
		}
  //-------------新しいクラスタセンタが同じ点であるかどうかの判定-------------
		if(countMatch == k){
			changed = false;
		}else{
			changed = true;
		}
	}while(changed);

//-----------------結果の描画------------------------------------
	Mat newImg = Mat(Size(600,600),CV_8UC3);
	cvtColor(image2,newImg,CV_GRAY2BGR);
	int minX_1 = 10000000000;
	int minY_1 = 10000000000;
	int maxX_1 = 0;
	int maxY_1 = 0;
	int minX_2 = 10000000000;
	int minY_2 = 10000000000;
	int maxX_2 = 0;
	int maxY_2 = 0;
	int minX_3 = 10000000000;
	int minY_3 = 10000000000;
	int maxX_3 = 0;
	int maxY_3 = 0;
	for(i= 0;i < data.size(); i++){
		if(clsLabel[i] == 0){
			if(minX_1 > data[i].x){
				minX_1 = data[i].x;
			}if(minY_1 > data[i].y){
				minY_1 = data[i].y;
			}if(maxX_1 < data[i].x){
				maxX_1 = data[i].x;
			}if(maxY_1 < data[i].y){
				maxY_1 = data[i].y;
			}
			rectangle(newImg,Point(data[i].y,data[i].x),
				Point(data[i].y,data[i].x),Scalar(255,0,0),1,1);
		}else if(clsLabel[i] == 1){
			if(minX_2 > data[i].x){
				minX_2 = data[i].x;
			}if(minY_2 > data[i].y){
				minY_2 = data[i].y;
			}if(maxX_2 < data[i].x){
				maxX_2 = data[i].x;
			}if(maxY_2 < data[i].y){
				maxY_2 = data[i].y;
			}
			rectangle(newImg,Point(data[i].y,data[i].x),
				Point(data[i].y,data[i].x),Scalar(0,255,0),1,1);
		}else if(clsLabel[i] = 2){
			if(minX_3 > data[i].x){
				minX_3 = data[i].x;
			}if(minY_3 > data[i].y){
				minY_3 = data[i].y;
			}if(maxX_3 < data[i].x){
				maxX_3 = data[i].x;
			}if(maxY_3 < data[i].y){
				maxY_3 = data[i].y;
			}
			rectangle(newImg,Point(data[i].y,data[i].x),
				Point(data[i].y,data[i].x),Scalar(0,0,255),1,1);
		}
	}
	for(i = 0; i < k;i++){
		circle(newImg,Point(bestCenterX[i],bestCenterY[i]),1,Scalar(0,0,0));
	}

	namedWindow("結果");
	imshow("結果",newImg);	
//料理の元画像の読み込みとリサイズ
	Mat dishes = imread("dishes1.png");
	Mat resizeDishes(Size(600,600),dishes.type());
	resize(dishes, resizeDishes,resizeDishes.size(),0,0,INTER_LINEAR);
//料理ごとに矩形で切り抜きと表示	
	Mat roi1(resizeDishes,Rect(minY_1,minX_1,maxX_1-minX_1,maxY_1-minY_1));
    Mat roi2(resizeDishes,Rect(minY_2,minX_2,maxX_2-minX_2,maxY_2-minY_2));
	Mat roi3(resizeDishes,Rect(minY_3,minX_3,maxX_3-minX_3,maxY_3-minY_3));
	namedWindow("roi1");
	imshow("roi1",roi1);
	namedWindow("roi2");
	imshow("roi2",roi2);
	namedWindow("roi3");
	imshow("roi3",roi3);
//全体の検出結果
	rectangle(resizeDishes,Point(minY_1,minX_1),
			Point(maxY_1,maxX_1),Scalar(255,0,0),1,1);
	rectangle(resizeDishes,Point(minY_2,minX_2),
			Point(maxY_2,maxX_2),Scalar(0,255,0),1,1);
	rectangle(resizeDishes,Point(minY_3,minX_3),
			Point(maxY_3,maxX_3),Scalar(0,0,255),1,1);
	namedWindow("dishes");
	imshow("dishes",resizeDishes);

	waitKey(0);
	destroyAllWindows();
	return 0;
}

int ellipse(){
	/*********ケニーのエッジ検出******************/
	int i,j,h,k,m;
	const double PI = 3.1415926535;
	Mat src_image = imread("s-jikitama02.jpg");
//	Mat src_image = imread("2_8_main.jpg");
//	namedWindow("元画像");
 //  imshow("元画像",src_image);
	GaussianBlur(src_image,src_image,Size(5,5),0);
	Mat channels[3];
	split(src_image, channels);

	Mat canny_r, canny_g, canny_b, canny_image;

	Canny(channels[2], canny_r, 20.0, 180.0, 3);
	Canny(channels[1], canny_g, 20.0, 180.0, 3);
	Canny(channels[0], canny_b, 20.0, 180.0, 3);

	bitwise_or(canny_r, canny_g, canny_image);
	bitwise_or(canny_image, canny_b, canny_image);
	canny_image = ~canny_image;

	namedWindow("RGB分解-Canny");
	imshow("RGB分解-Canny",canny_image);

/**********ハフ変換**************************/
	const int width = canny_image.cols;
	const int height = canny_image.rows;
	double tangent,tan1,tan2,arcTan1,arcTan2,radian1,radian2,theta1,theta2;
	int theta,p,pp,xi,yi,xj,yj;
	vector<int> v;
	const int windowSize = 11;
	const int newWidth = width+windowSize-1;
	const int newHeight = height+windowSize-1;
	Mat newImg = Mat(Size(newWidth,newHeight),CV_8U);
	for(int j = 0; j < newHeight;j++){
	   for(int i = 0; i < newWidth;i++){
		   if(i < windowSize/2 || i > width || j > height || j < windowSize/2){
//			   printf("[A]x:%d y:%d \n",j,i);
			   newImg.at<unsigned char>(j,i) = 255;
		   }else{
//			   printf("[B]x:%d y:%d \n",j,i);
		       newImg.at<unsigned char>(j,i) = canny_image.at<unsigned char>((int)(j-(windowSize/2)),(int)(i-(windowSize/2)));
		   }
	   }
	}

	namedWindow("元");
	imshow("元",newImg);

//	const int nData = width*height;
	unordered_map<vector<int>,int,HashVI> data;

	int votingSpace[180];
	for(i = 0; i < 180;i++){
		votingSpace[i] = 0;
	}
	
	for(j = (int)(windowSize/2); j < (int)((windowSize/2)+height); j++){
		for(i = (int)(windowSize/2);i < (int)((windowSize/2)+width); i++){
			p = static_cast<int>(newImg.at<unsigned char>(j,i));
			if(p == 0){
				xi = i;   //注目座標の中心x座標
				yi = j;   //注目座標の中心y座標
	//			printf("%d",xi);
	//			printf("%d \n",yi);
				for(h = 0;h < windowSize;h++){      //窓内の走査
					for(k = 0;k < windowSize;k++){
					   xj = (xi-5)+k;
					   yj = (yi-5)+h;
					   if((xj == xi && yj == yi)|| xj < 0 || yj < 0 ||xj >= width||yj >= height){
						   continue;
					   }
	//				   printf("xj:%d",xj);
	//				   printf("yj:%d",yj);
			    	   pp =  static_cast<int>(newImg.at<unsigned char>(yj,xj));  
					   
					   if(pp == 0){
						   yj = height - yj;
						   if((xj-xi) == 0){
							   tangent = 0;
						   }else{
				     		   tangent = (yj-yi)/(xj-xi);
						   }
						   if(abs(tangent) > 1){
							   yj = xj;
							   xj = yj;
							   tan1 = (yj-yi-0.5)/(xj-xi);
						       tan2 = (yj-yi+0.5)/(xj-xi);
							   arcTan1 = atan(tan1);
							   arcTan2 = atan(tan2);
							   if(arcTan1 >= 0){	     
								   theta1 = (arcTan1*180/PI)+90;
								   ceil(theta1);
								   theta2 = (arcTan2*180/PI)+90;
								   floor(theta2);
							   }else{
								   theta1 = 90+(arcTan1*180/PI);
								   ceil(theta1);
								   theta2 = 90+(arcTan2*180/PI);
								   floor(theta2);
							   }
						   }
						   else{
							   tan1 = (yj-yi-0.5)/(xj-xi);
						       tan2 = (yj-yi+0.5)/(xj-xi);
							   arcTan1 = atan(tan1);
							   arcTan2 = atan(tan2);
							   if(arcTan1 >= 0){
								   theta1 = (arcTan1*180/PI);
			     				   ceil(theta1);
			     				   theta2 = (arcTan2*180/PI);
			     				   floor(theta2);
							   }else{
								   theta1 = -(arcTan1*180/PI)+90;
			     				   ceil(theta1);
			     				   theta2 = -(arcTan2*180/PI)+90;
			     				   floor(theta2);
							   }
						   }
						   
						   for(m = theta1; m <= theta2;m++){
							   votingSpace[m] += 1;
						   }
					   }
					}
				}
				int thresh = 6;
				int overThreshRad = 0;
				int overNum = 0;
				for(int a = 0; a < 180;a++){ 
					if(votingSpace[a] > thresh){
						overThreshRad += a;
						overNum++;
					}
				}
				if(overNum == 0){
				    overNum = 1;
				}

				overThreshRad = overThreshRad/overNum;
				v.push_back(i - (int)(windowSize/2));
				v.push_back(j - (int)(windowSize/2));
				data[v] = overThreshRad;
				v.clear();
			}
		}
	}
/*	
	double meanX,meanY; 
	const int N = 2;
	double H[N][N],U[N][N],E[N];
	int innerEdgeNum,info;

	for(j = (int)(windowSize/2); j < (int)((windowSize/2)+height); j++){
		for(i = (int)(windowSize/2);i < (int)((windowSize/2)+width); i++){
			p = static_cast<int>(newImg.at<unsigned char>(j,i));
			if(p == 0){
				xi = i;   //注目座標の中心x座標
				yi = j;   //注目座標の中心y座標
				for(h = 0;h < windowSize;h++){      //窓内の走査
					for(k = 0;k < windowSize;k++){
					   xj = (xi-5)+k;
					   yj = (yi-5)+h;
					   if((xj == xi && yj == yi)|| xj < 0 || yj < 0 ||xj >= width||yj >= height){
						   continue;
					   }
			    	   pp =  static_cast<int>(newImg.at<unsigned char>(yj,xj));  
					   
					   if(pp == 0){
						   meanX += xj;
						   meanY += yj;
						   innerEdgeNum++;
					   }
					}
				}

				meanX = meanX/innerEdgeNum;
				meanY = meanY/innerEdgeNum;
				for(h = 0; h < N;h++){
					for(k = 0;k <= h; k++){
						if( h == k+1){
							H[h][k] = -1.0;
						}else{
							H[h][k] = 0.0;
						}
					}
				}
			}
		}
	}
	*/
	vector<int> vv;
	for(j = 0; j < height; j++){
		for(i = 0; i < width;i++){
			p = static_cast<int>(canny_image.at<unsigned char>(j,i)); 
			if(p == 0){
     			vv.push_back(i);
	    		vv.push_back(j);
                printf("x:%d y:%d => theta:%d \n",i,j,data[vv]);
                vv.clear();
			}
		}
	}


	waitKey(0);
	destroyAllWindows();
	return 0;

}