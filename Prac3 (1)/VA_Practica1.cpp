////////////////////////////////Cabeceras/////////////////////////////////////
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#define pi 3.1416
#define e 2.72
/////////////////////////////////////////////////////////////////////////////

///////////////////////////////Espacio de nombres////////////////////////////
using namespace cv;
using namespace std;
/////////////////////////////////////////////////////////////////////////////

Mat RefillImg(Mat original, int ksize) {

	int dif = (ksize - 1);
	Mat FilledImg(original.rows + dif, original.cols + dif, CV_8UC1);
	for (int i = 0; i < original.rows + dif;i++) {
		for (int j = 0; j < original.cols + dif;j++) {
			if (i > dif / 2 && j > dif / 2 && i <= (original.rows - (dif / 2)) && j <= (original.cols - (dif / 2))) {
				FilledImg.at<uchar>(Point(i, j)) = uchar(original.at<uchar>(Point(i - (dif / 2), j - (dif / 2))));
			}
			else {
				FilledImg.at<uchar>(Point(i, j)) = uchar(0);
			}
		}
	}
	return FilledImg;
}

vector<vector<float>> Mask(int kSize, int sigma) {
	int difb = (kSize - 1) / 2;
	vector<vector<float>> filter(kSize, vector<float>(kSize, 0));
	for (int i = -difb; i <= difb; i++)
	{
		for (int j = -difb; j <= difb; j++)
		{
			float operation = (1 / (2 * pi * sigma * sigma)) * pow(e, -((i * i + j * j) / (2 * sigma * sigma)));
			filter[i + difb][j + difb] = operation;
			cout << operation << " ";
		}
		cout << endl;
	}
	return filter;
}

float Filter(Mat original, vector<vector<float>> kernel, int kSize, int x, int y) {
	int difb = (kSize - 1) / 2;
	float sumFilter = 0;
	float sumKernel = 0;
	for (int i = -difb; i <= difb; i++)
	{
		for (int j = -difb; j <= difb; j++)
		{
			float kTmp = kernel[i + difb][j + difb];
			int tmpX = x + i;
			int tmpY = y + j;
			float tmp = 0;
			tmp = original.at<uchar>(Point(tmpX, tmpY));
			sumFilter += (kTmp * tmp);
			sumKernel += kTmp;
		}
	}
	return sumFilter / sumKernel;
}

Mat FilterImg(Mat original, vector<vector<float>> kernel, int kSize) {
	int difb = (kSize - 1) / 2;
	Mat filteredImg(original.rows, original.cols, CV_8UC1);
	for (int i = difb+1; i < original.rows-difb; i++)
	{
		for (int j = difb+1; j < original.cols-difb; j++) {
			filteredImg.at<uchar>(Point(i, j)) = uchar(Filter(original, kernel, kSize, i, j));
		}
	}
	return filteredImg;
}





/////////////////////////Inicio de la funcion principal///////////////////
int main()
{

	/********Declaracion de variables generales*********/
	char NombreImagen[] = "lena.jpg";
	Mat imagen; // Matriz que contiene nuestra imagen sin importar el formato
	/************************/

	/*********Lectura de la imagen*********/
	imagen = imread(NombreImagen);

	if (!imagen.data)
	{
		cout << "Error al cargar la imagen: " << NombreImagen << endl;
		exit(1);
	}
	/************************/

	/************Procesos*********/
	int fila_original = imagen.rows;
	int columna_original = imagen.cols;//Lectur de cuantas columnas
	int ksize = 0;
	int sigma = 0;

	Mat gris_pond(fila_original, columna_original, CV_8UC1);

	for (int i = 0; i < fila_original; i++)
	{
		for (int j = 0; j < columna_original; j++)
		{
			double azul = imagen.at<Vec3b>(Point(j, i)).val[0];  // B
			double verde = imagen.at<Vec3b>(Point(j, i)).val[1]; // G
			double rojo = imagen.at<Vec3b>(Point(j, i)).val[2];  // R

			gris_pond.at<uchar>(Point(j, i)) = uchar(0.299 * rojo + 0.587 * verde + 0.114 * azul);
		}
	}



	cout << "Tamaño de la mascara: " << endl;
	cin >> ksize;

	if (ksize % 2 == 0) {
		cout << "Tamaño de mascara invalidoo" << endl;
		exit(0);
	}


	cout << "Valor sigma: " << endl;
	cin >> sigma;

	Mat ImgRefill = RefillImg(gris_pond, ksize);
	vector<vector<float>> kernel = Mask(ksize, sigma);
	Mat filtrada = FilterImg(ImgRefill, kernel, ksize);

	namedWindow("Hola Mundo", WINDOW_AUTOSIZE);//Creación de una ventana
	imshow("Hola Mundo", imagen);

	namedWindow("Gris_Ponderada", WINDOW_AUTOSIZE);//Creación de una ventana
	imshow("Gris_Ponderada", gris_pond);

	namedWindow("Refill_Img", WINDOW_AUTOSIZE);//Creación de una ventana
	imshow("Refill_Img", ImgRefill);

	namedWindow("Filtrada", WINDOW_AUTOSIZE);//Creación de una ventana
	imshow("Filtrada", filtrada);

	/************************/

	waitKey(0); //Función para esperar
	return 1;
}
////////////////////////////////////////////////////////////////////////