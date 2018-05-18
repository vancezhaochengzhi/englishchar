#include "opencv2\opencv.hpp"
#include <windows.h>
#include <vector>
#include <io.h>

using namespace std;
using namespace cv;

#pragma comment(lib, "opencv_core249d.lib")
#pragma comment(lib, "opencv_highgui249d.lib")
#pragma comment(lib, "opencv_objdetect249d.lib")  
#pragma comment(lib, "opencv_imgproc249d.lib")
#pragma comment(lib, "opencv_legacy249d.lib")
#pragma comment(lib, "opencv_ml249d.lib")

#define DEST_WIDTH 10 //目标宽度
#define DEST_HEIGHT 20 //目标高度

CvANN_MLP  m_ann;//神经网络

float trainratio = 0.7;


const char strCharacters[] = {'A','B', 'C', 'D', 'E','F', 'G', 'H',\
	'J', 'K', 'L', 'M', 'N', /* 没有O */ 'P', 'Q', 'R', 'S', 'T', 'U','V', 'W', 'X', 'Y', 'Z'}; 
const int numCharacter = 24; /* 没有I和O,10个数字与24个英文字符之和 */


//这个函数遍历获得文件夹内的所有图片文件 bmp jpg png
void getPicFiles(string path, vector<string>& files)
{
	//文件句柄
	long   hFile   =   0;
	//文件信息
	struct _finddata_t fileinfo;
	string p;
	if((hFile = _findfirst(p.assign(path).append("\\*").c_str(),&fileinfo)) !=  -1)
	{
		do
		{
			//如果是目录,迭代之
			//如果不是,加入列表
			if((fileinfo.attrib &  _A_SUBDIR))
			{
				if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0)
					getPicFiles(path+"\\"+fileinfo.name, files);
			}
			else
			{

				char *pp;
				pp = strrchr(fileinfo.name,'.');//查找最后出现的位置
				if (_stricmp(pp,".bmp")==0 || _stricmp(pp,".jpg")==0 || _stricmp(pp,".png")==0 )//如果找到的是图片就行处理
				{
					files.push_back(p.assign(path).append("\\").append(fileinfo.name));//这个存储带路径的文件全名
				}
			}
		}while(_findnext(hFile, &fileinfo)  == 0);
		_findclose(hFile);
	}
}

////简单的预处理 归一大小 灰度化等
//Mat preproc(Mat image)
//{
//	Mat grayimg;
//	if (image.channels() == 3)
//	{
//		cvtColor(image,grayimg,CV_BGR2GRAY);//转为灰度图
//	}
//	else
//	{
//		grayimg = image.clone();
//	}
//
//	Mat sizeimg;
//	resize(grayimg,sizeimg,Size(DEST_WIDTH,DEST_HEIGHT));//归一到统一大小
//	
//	Mat procimg = sizeimg;
//
//	return procimg;
//}

//这个函数将图像预处理到紧紧包含。没有黑边，归一到统一大小
Mat preprocimg(Mat in)
{
	Mat grayI;
	if(in.channels() == 3)
	{
		cvtColor(in,grayI,CV_BGR2GRAY );//转为灰度图像
	}
	else
	{
		grayI = in;
	}

	Mat bwI;
	threshold(grayI,bwI,128,255,CV_THRESH_BINARY);
	
	int rows = bwI.rows;//行数
	int cols = bwI.cols;//列数
	int left,right,top,bottom;
	//左边界
	for (int i=0;i<cols;i++)//遍历每一列
	{	
		Mat data=bwI.col(i);//取得一列
		int whitenum = countNonZero(data);	//统计这一行或一列中，非零元素的个数
		if(whitenum > 0)//找到白点列了
		{
			left = i;//左边界
			break;
		}

	}

	//右边界
	for (int i=cols-1; i>=0; i--)//遍历每一列
	{	
		Mat data=bwI.col(i);//取得一列
		int whitenum = countNonZero(data);	//统计这一行或一列中，非零元素的个数
		if(whitenum > 0)//找到白点列了
		{
			right = i;//右边界
			break;
		}
	}

	//上边界
	for (int i=0;i<rows;i++)//遍历每一行
	{	
		Mat data=bwI.row(i);//取得一行
		int whitenum = countNonZero(data);	//统计这一行或一列中，非零元素的个数
		if(whitenum > 0)//找到白点行了
		{
			top = i;//左边界
			break;
		}

	}

	//下边界
	for (int i=rows-1; i>=0; i--)//遍历每一行
	{	
		Mat data=bwI.row(i);//取得一行
		int whitenum = countNonZero(data);	//统计这一行或一列中，非零元素的个数
		if(whitenum > 0)//找到白点行了
		{
			bottom = i;//下边界
			break;
		}
	}
	//组织要截取的区域
	Rect r;
	r.x = left;
	r.y = top;
	r.height = bottom-top+1;
	r.width = right-left+1;//

	Mat image_roi = bwI(r);//截取区域
	Mat result;
	resize(image_roi, result, Size(DEST_WIDTH,DEST_HEIGHT) );
	threshold(result,result,80,1,CV_THRESH_BINARY);//二值化到01区间
	return result;
}

Mat features(Mat in)
{

	Mat procI = preprocimg(in);//截取区域出来,紧紧包含

	//Low data feature
	Mat lowData = procI;
	int numCols=lowData.rows*lowData.cols;//像素总数

	Mat out=Mat::zeros(1,numCols,CV_32F);
	
	int j=0;
	//遍历每行每列的像素值，赋予到out内
	for(int x=0; x<lowData.cols; x++)
	{
		for(int y=0; y<lowData.rows; y++){
			out.at<float>(j)=(float)lowData.at<unsigned char>(y,x);
			j++;
		}
	}

	return out;
}



void mytrain()
{
	Mat trainData;  //训练的特征数据
	Mat trainLabel; //训练的类型标签
	int i = 0;
	for (i=0;i<numCharacter;i++)//遍历这些类别
	{	
		char strpicpath[260] = {0};
		sprintf(strpicpath,"..\\..\\字符集\\chars2\\%c",strCharacters[i]);//组织子文件夹名称

		vector<string> files;
		getPicFiles(strpicpath, files);//获取这个文件夹下的所有图
		int filenum = files.size();//文件个数
		int trainnum = filenum*trainratio;
		for (int j = 0; j < trainnum; j++)//遍历这些训练图
		{
			Mat image = imread(files[j].c_str());//读取这个图片
			if(image.data == NULL)
			{
				continue;//如果读取失败
			}

			//Mat procimg = preprocimg(image);//预处理图片
			
			Mat f=features(image);//计算这个图像的特征

			trainData.push_back(f);//这一行存储到traindata内
			trainLabel.push_back(i);//类别编号


		}
	

	}

	
	//遍历读完图库后，训练bp神经网络
	trainData.convertTo(trainData, CV_32FC1);
  
	int nNeruns = 30;//隐含层结点个数

	m_ann.clear();
	Mat layers(1, 3, CV_32SC1);
	layers.at<int>(0) = trainData.cols;//这是神经网络的输入维度
	layers.at<int>(1) = nNeruns;//隐含层维度
	layers.at<int>(2) = numCharacter;//输出维度

	m_ann.create(layers, CvANN_MLP::SIGMOID_SYM, 1, 1);

	CvANN_MLP_TrainParams param;  
	param.term_crit = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,1000,0.001);  //设置结束条件  
	param.train_method = CvANN_MLP_TrainParams::BACKPROP;       //训练方法采用BackProgation  
	param.bp_dw_scale=0.1;  
	param.bp_moment_scale=0.1; 

	//组织bp神经网络的输出编码，输出的不是一个数字，而是N类，就是N个数字，对应类别置1，其他都是0
	Mat trainClasses;
	trainClasses.create( trainData.rows, numCharacter, CV_32FC1 );
	for( int i = 0; i <  trainClasses.rows; i++ )//遍历每行
	{
		for( int k = 0; k < trainClasses.cols; k++ )//遍历每列，其中一位置1，其他都是0
		{
			//If class of data i is same than a k class
			if( k == trainLabel.at<int>(i) )
				trainClasses.at<float>(i,k) = 1;//对应类别位置处置1
			else
				trainClasses.at<float>(i,k) = 0;//其他都是0
		}
	}
	Mat weights( 1, trainData.rows, CV_32FC1, Scalar::all(1) );

	//Learn classifier
	m_ann.train( trainData, trainClasses, weights,Mat(),param );//训练

	m_ann.save("englishchar.xml");//保存分类器
}

void alltest()
{
	int i,j;
	int testrightnum = 0;//测试的正确个数
	int testtotalnum = 0;//测试总样本数
	//遍历测试样本去测试
	for (i=0;i<numCharacter;i++)//遍历这些类别
	{	
		char strpicpath[260] = {0};
		sprintf(strpicpath,"..\\..\\字符集\\chars2\\%c",strCharacters[i]);//组织子文件夹名称

		vector<string> files;
		getPicFiles(strpicpath, files);//获取这个文件夹下的所有图
		int filenum = files.size();//文件个数
		int trainnum = filenum*trainratio;
		for (j = trainnum; j < filenum; j++)//遍历这些训练图
		{
			Mat image = imread(files[j].c_str());//读取这个图片
			if(image.data == NULL)
			{
				continue;//如果读取失败
			}

			 Mat f=features(image);//计算这个图像的特征

			Mat output(1, numCharacter, CV_32FC1);//存放神经网络的输出
			m_ann.predict(f, output);

			int rsultindex = 0;//识别的最终类型
			
			float maxVal = -2;
			for(int j = 0; j < numCharacter; j++)
			{
				float val = output.at<float>(j);
				//cout << "j:" << j << "val:"<< val << endl;
				if (val > maxVal)
				{
					maxVal = val;
					rsultindex = j;
				}
			}
			
			

			//如果测试出来的和真实的类别一样，正确个数加1
			if (rsultindex == i )
			{
				testrightnum++;
			}
			else//如果错误，输出详细信息
			{
				printf("%s错误的识别为%c\r\n",files[j].c_str(),strCharacters[rsultindex]);
			}
			
			testtotalnum++;//不管对错，测试个数加1
		}


	}

	double rightrate = double(testrightnum)/double(testtotalnum);//准确率
	char strinfo[256] = {0};//文件名
	sprintf_s(strinfo,"测试%d张。准确%d张，准确率为:%.2f%%%%\r\n",testtotalnum,testrightnum,rightrate*100);
	printf(strinfo);

}

void main()
{
	
	DWORD T1 = GetTickCount();
	mytrain();//训练
	DWORD T2 = GetTickCount();
	printf("训练用时%dms\r\n",T2-T1);

	
	T1 = GetTickCount();
	alltest();//全部测试
	T2 = GetTickCount();
	printf("测试用时%dms\r\n",T2-T1);
	system("pause");
	////以下代码为单个图像测试
	//while(1)
	//{
	//	char strtestfilename[256] = {0};
	//	printf("\r\n请输入要测试的图片路径:\r\n");
	//	scanf("%s",strtestfilename);

	//	Mat image = imread(strtestfilename);//读取这个图片
	//	if(image.data == NULL)
	//	{
	//		printf("读取图像失败，请检查图片路径是否有误\r\n");//如果读取失败
	//		return;
	//	}

	//	//Mat procimg = preprocimg(image);//预处理图片
	//	 Mat f=features(image);//计算这个图像的特征

	//	Mat output(1, numCharacter, CV_32FC1);//存放神经网络的输出
	//	m_ann.predict(f, output);

	//	int rsultindex = 0;//识别的最终类型
	//		
	//	float maxVal = -2;
	//	for(int j = 0; j < numCharacter; j++)
	//	{
	//		float val = output.at<float>(j);
	//		//cout << "j:" << j << "val:"<< val << endl;
	//		if (val > maxVal)
	//		{
	//			maxVal = val;
	//			rsultindex = j;
	//		}
	//	}
	//		

	//	printf("%s识别为:%c\r\n",strtestfilename,strCharacters[rsultindex]);
	//}
}