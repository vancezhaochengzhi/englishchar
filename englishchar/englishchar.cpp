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

#define DEST_WIDTH 10 //Ŀ����
#define DEST_HEIGHT 20 //Ŀ��߶�

CvANN_MLP  m_ann;//������

float trainratio = 0.7;


const char strCharacters[] = {'A','B', 'C', 'D', 'E','F', 'G', 'H',\
	'J', 'K', 'L', 'M', 'N', /* û��O */ 'P', 'Q', 'R', 'S', 'T', 'U','V', 'W', 'X', 'Y', 'Z'}; 
const int numCharacter = 24; /* û��I��O,10��������24��Ӣ���ַ�֮�� */


//���������������ļ����ڵ�����ͼƬ�ļ� bmp jpg png
void getPicFiles(string path, vector<string>& files)
{
	//�ļ����
	long   hFile   =   0;
	//�ļ���Ϣ
	struct _finddata_t fileinfo;
	string p;
	if((hFile = _findfirst(p.assign(path).append("\\*").c_str(),&fileinfo)) !=  -1)
	{
		do
		{
			//�����Ŀ¼,����֮
			//�������,�����б�
			if((fileinfo.attrib &  _A_SUBDIR))
			{
				if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0)
					getPicFiles(path+"\\"+fileinfo.name, files);
			}
			else
			{

				char *pp;
				pp = strrchr(fileinfo.name,'.');//���������ֵ�λ��
				if (_stricmp(pp,".bmp")==0 || _stricmp(pp,".jpg")==0 || _stricmp(pp,".png")==0 )//����ҵ�����ͼƬ���д���
				{
					files.push_back(p.assign(path).append("\\").append(fileinfo.name));//����洢��·�����ļ�ȫ��
				}
			}
		}while(_findnext(hFile, &fileinfo)  == 0);
		_findclose(hFile);
	}
}

////�򵥵�Ԥ���� ��һ��С �ҶȻ���
//Mat preproc(Mat image)
//{
//	Mat grayimg;
//	if (image.channels() == 3)
//	{
//		cvtColor(image,grayimg,CV_BGR2GRAY);//תΪ�Ҷ�ͼ
//	}
//	else
//	{
//		grayimg = image.clone();
//	}
//
//	Mat sizeimg;
//	resize(grayimg,sizeimg,Size(DEST_WIDTH,DEST_HEIGHT));//��һ��ͳһ��С
//	
//	Mat procimg = sizeimg;
//
//	return procimg;
//}

//���������ͼ��Ԥ��������������û�кڱߣ���һ��ͳһ��С
Mat preprocimg(Mat in)
{
	Mat grayI;
	if(in.channels() == 3)
	{
		cvtColor(in,grayI,CV_BGR2GRAY );//תΪ�Ҷ�ͼ��
	}
	else
	{
		grayI = in;
	}

	Mat bwI;
	threshold(grayI,bwI,128,255,CV_THRESH_BINARY);
	
	int rows = bwI.rows;//����
	int cols = bwI.cols;//����
	int left,right,top,bottom;
	//��߽�
	for (int i=0;i<cols;i++)//����ÿһ��
	{	
		Mat data=bwI.col(i);//ȡ��һ��
		int whitenum = countNonZero(data);	//ͳ����һ�л�һ���У�����Ԫ�صĸ���
		if(whitenum > 0)//�ҵ��׵�����
		{
			left = i;//��߽�
			break;
		}

	}

	//�ұ߽�
	for (int i=cols-1; i>=0; i--)//����ÿһ��
	{	
		Mat data=bwI.col(i);//ȡ��һ��
		int whitenum = countNonZero(data);	//ͳ����һ�л�һ���У�����Ԫ�صĸ���
		if(whitenum > 0)//�ҵ��׵�����
		{
			right = i;//�ұ߽�
			break;
		}
	}

	//�ϱ߽�
	for (int i=0;i<rows;i++)//����ÿһ��
	{	
		Mat data=bwI.row(i);//ȡ��һ��
		int whitenum = countNonZero(data);	//ͳ����һ�л�һ���У�����Ԫ�صĸ���
		if(whitenum > 0)//�ҵ��׵�����
		{
			top = i;//��߽�
			break;
		}

	}

	//�±߽�
	for (int i=rows-1; i>=0; i--)//����ÿһ��
	{	
		Mat data=bwI.row(i);//ȡ��һ��
		int whitenum = countNonZero(data);	//ͳ����һ�л�һ���У�����Ԫ�صĸ���
		if(whitenum > 0)//�ҵ��׵�����
		{
			bottom = i;//�±߽�
			break;
		}
	}
	//��֯Ҫ��ȡ������
	Rect r;
	r.x = left;
	r.y = top;
	r.height = bottom-top+1;
	r.width = right-left+1;//

	Mat image_roi = bwI(r);//��ȡ����
	Mat result;
	resize(image_roi, result, Size(DEST_WIDTH,DEST_HEIGHT) );
	threshold(result,result,80,1,CV_THRESH_BINARY);//��ֵ����01����
	return result;
}

Mat features(Mat in)
{

	Mat procI = preprocimg(in);//��ȡ�������,��������

	//Low data feature
	Mat lowData = procI;
	int numCols=lowData.rows*lowData.cols;//��������

	Mat out=Mat::zeros(1,numCols,CV_32F);
	
	int j=0;
	//����ÿ��ÿ�е�����ֵ�����赽out��
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
	Mat trainData;  //ѵ������������
	Mat trainLabel; //ѵ�������ͱ�ǩ
	int i = 0;
	for (i=0;i<numCharacter;i++)//������Щ���
	{	
		char strpicpath[260] = {0};
		sprintf(strpicpath,"..\\..\\�ַ���\\chars2\\%c",strCharacters[i]);//��֯���ļ�������

		vector<string> files;
		getPicFiles(strpicpath, files);//��ȡ����ļ����µ�����ͼ
		int filenum = files.size();//�ļ�����
		int trainnum = filenum*trainratio;
		for (int j = 0; j < trainnum; j++)//������Щѵ��ͼ
		{
			Mat image = imread(files[j].c_str());//��ȡ���ͼƬ
			if(image.data == NULL)
			{
				continue;//�����ȡʧ��
			}

			//Mat procimg = preprocimg(image);//Ԥ����ͼƬ
			
			Mat f=features(image);//�������ͼ�������

			trainData.push_back(f);//��һ�д洢��traindata��
			trainLabel.push_back(i);//�����


		}
	

	}

	
	//��������ͼ���ѵ��bp������
	trainData.convertTo(trainData, CV_32FC1);
  
	int nNeruns = 30;//�����������

	m_ann.clear();
	Mat layers(1, 3, CV_32SC1);
	layers.at<int>(0) = trainData.cols;//���������������ά��
	layers.at<int>(1) = nNeruns;//������ά��
	layers.at<int>(2) = numCharacter;//���ά��

	m_ann.create(layers, CvANN_MLP::SIGMOID_SYM, 1, 1);

	CvANN_MLP_TrainParams param;  
	param.term_crit = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,1000,0.001);  //���ý�������  
	param.train_method = CvANN_MLP_TrainParams::BACKPROP;       //ѵ����������BackProgation  
	param.bp_dw_scale=0.1;  
	param.bp_moment_scale=0.1; 

	//��֯bp�������������룬����Ĳ���һ�����֣�����N�࣬����N�����֣���Ӧ�����1����������0
	Mat trainClasses;
	trainClasses.create( trainData.rows, numCharacter, CV_32FC1 );
	for( int i = 0; i <  trainClasses.rows; i++ )//����ÿ��
	{
		for( int k = 0; k < trainClasses.cols; k++ )//����ÿ�У�����һλ��1����������0
		{
			//If class of data i is same than a k class
			if( k == trainLabel.at<int>(i) )
				trainClasses.at<float>(i,k) = 1;//��Ӧ���λ�ô���1
			else
				trainClasses.at<float>(i,k) = 0;//��������0
		}
	}
	Mat weights( 1, trainData.rows, CV_32FC1, Scalar::all(1) );

	//Learn classifier
	m_ann.train( trainData, trainClasses, weights,Mat(),param );//ѵ��

	m_ann.save("englishchar.xml");//���������
}

void alltest()
{
	int i,j;
	int testrightnum = 0;//���Ե���ȷ����
	int testtotalnum = 0;//������������
	//������������ȥ����
	for (i=0;i<numCharacter;i++)//������Щ���
	{	
		char strpicpath[260] = {0};
		sprintf(strpicpath,"..\\..\\�ַ���\\chars2\\%c",strCharacters[i]);//��֯���ļ�������

		vector<string> files;
		getPicFiles(strpicpath, files);//��ȡ����ļ����µ�����ͼ
		int filenum = files.size();//�ļ�����
		int trainnum = filenum*trainratio;
		for (j = trainnum; j < filenum; j++)//������Щѵ��ͼ
		{
			Mat image = imread(files[j].c_str());//��ȡ���ͼƬ
			if(image.data == NULL)
			{
				continue;//�����ȡʧ��
			}

			 Mat f=features(image);//�������ͼ�������

			Mat output(1, numCharacter, CV_32FC1);//�������������
			m_ann.predict(f, output);

			int rsultindex = 0;//ʶ�����������
			
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
			
			

			//������Գ����ĺ���ʵ�����һ������ȷ������1
			if (rsultindex == i )
			{
				testrightnum++;
			}
			else//������������ϸ��Ϣ
			{
				printf("%s�����ʶ��Ϊ%c\r\n",files[j].c_str(),strCharacters[rsultindex]);
			}
			
			testtotalnum++;//���ܶԴ����Ը�����1
		}


	}

	double rightrate = double(testrightnum)/double(testtotalnum);//׼ȷ��
	char strinfo[256] = {0};//�ļ���
	sprintf_s(strinfo,"����%d�š�׼ȷ%d�ţ�׼ȷ��Ϊ:%.2f%%%%\r\n",testtotalnum,testrightnum,rightrate*100);
	printf(strinfo);

}

void main()
{
	
	DWORD T1 = GetTickCount();
	mytrain();//ѵ��
	DWORD T2 = GetTickCount();
	printf("ѵ����ʱ%dms\r\n",T2-T1);

	
	T1 = GetTickCount();
	alltest();//ȫ������
	T2 = GetTickCount();
	printf("������ʱ%dms\r\n",T2-T1);
	system("pause");
	////���´���Ϊ����ͼ�����
	//while(1)
	//{
	//	char strtestfilename[256] = {0};
	//	printf("\r\n������Ҫ���Ե�ͼƬ·��:\r\n");
	//	scanf("%s",strtestfilename);

	//	Mat image = imread(strtestfilename);//��ȡ���ͼƬ
	//	if(image.data == NULL)
	//	{
	//		printf("��ȡͼ��ʧ�ܣ�����ͼƬ·���Ƿ�����\r\n");//�����ȡʧ��
	//		return;
	//	}

	//	//Mat procimg = preprocimg(image);//Ԥ����ͼƬ
	//	 Mat f=features(image);//�������ͼ�������

	//	Mat output(1, numCharacter, CV_32FC1);//�������������
	//	m_ann.predict(f, output);

	//	int rsultindex = 0;//ʶ�����������
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

	//	printf("%sʶ��Ϊ:%c\r\n",strtestfilename,strCharacters[rsultindex]);
	//}
}