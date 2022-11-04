#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#define pi 3.1416
#define e 2.72


using namespace cv;
using namespace std;
//A partir del valor del kernel y sigama => obtendremos un kernel gaussiano

vector<vector<float>> KernelGauss(int kSize, float sigma) {
	int difABorde = (kSize - 1) / 2;
	vector<vector<float>> v(kSize, vector<float>(kSize, 0));
	// consideramos que el centro es (0,0) y apartir de ello se va a determinar la totalidad del kernel
	for (int i = -difABorde; i <= difABorde; i++)
	{
		for (int j = -difABorde; j <= difABorde; j++)
		{
			float resultado = (1 / (2 * pi * sigma * sigma)) * pow(e, -((i * i +  j*j) / (2 * sigma * sigma)));
			v[i + difABorde][j + difABorde] = resultado;
		}
	}
	return v;
}
// Con la ayuda de un kernel se realizara la operacion de convolucion en una matriz x,y
float FiltroAPixel(Mat original, vector<vector<float>> kernel, int kSize, int x, int y) {
	int rows = original.rows;
	int cols = original.cols;
	//DifABorde son las casillas entre el centro de la matriz a uno de sus bordes y a su vex el excedente al aplicar un kernel sobre una matriz en una de sus esquinas
	int difABorde = (kSize - 1) / 2;
	float sumFilter = 0;
	float sumKernel = 0;
	// (0,0) se recorre para obtener nuestras coordenadas deseadas 
	for (int i = -difABorde; i <= difABorde; i++)
	{
		for (int j = -difABorde; j <= difABorde; j++)
		{
			float kTmp = kernel[i + difABorde][j + difABorde];
			int tmpX = x + i;
			int tmpY = y + j;
			float tmp = 0;
			if (!(tmpX < 0 || tmpX >= cols || tmpY < 0 || tmpY >= rows)) {
				tmp = original.at<uchar>(Point(tmpX, tmpY));
			}
			 
			sumFilter += (kTmp * tmp);
			sumKernel += kTmp;
		}
	}
	if (sumKernel == 0) { sumKernel = 1; }
	return sumFilter / sumKernel;
}

// Operacion de convolucion para una matriz con un kernel
Mat FiltroAMatriz(Mat original, vector<vector<float>> kernel) {
	// A partir de un kernel simetrico
	int kSize = kernel[0].size();
	Mat filteredImg(original.rows, original.cols, CV_8UC1);
	for (int i = 0; i < original.rows; i++)
	{
		for (int j = 0; j < original.cols; j++) {
			int pixVal = FiltroAPixel(original, kernel, kSize, i, j);
			if (pixVal < 0) { pixVal = 0; }
			if (pixVal > 255) { pixVal = 255; }
			filteredImg.at<uchar>(Point(i, j)) = uchar(pixVal);
		}
	}
	return filteredImg;
}

Mat crearBorde(Mat original, int borderSize) {
	//Se redimenciona despues de obtener el borde
	int extRows = original.rows + borderSize;
	int extCols = original.cols + borderSize;

	// Imagen con dimensiones nuevas
	Mat newImg(extRows, extCols, CV_8UC1);
	for (int i = 0; i < extRows; i++) {
		for (int j = 0; j < extCols; j++) {
			// bordes con relleno
			if (i <= borderSize || j <= borderSize || i > original.rows || j > original.cols) {
				newImg.at<uchar>(Point(i, j)) = uchar(0);
			}
			else {
				//Relleno de imagen original en el centro de la nueva imagen
				newImg.at<uchar>(Point(i, j)) = uchar(original.at<uchar>(Point(i - borderSize, j - borderSize)));
			}
		}
	}
	return newImg;
}

void MostrarTamMat(Mat mat, String name) {
	cout << name << ": [" << mat.rows << "," << mat.cols << "]" << endl;
}

// Escala de grises a NTSC
Mat GrisesANtsc(Mat img) {
	Mat grayscale(img.rows, img.cols, CV_8UC1);
	// Pasamos a escala de grises usando NTSC
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			
			double R = img.at<Vec3b>(Point(j, i)).val[2];
			double G = img.at<Vec3b>(Point(j, i)).val[1];
			double B = img.at<Vec3b>(Point(j, i)).val[0];

			grayscale.at<uchar>(Point(j, i)) = uchar(0.299 * R + 0.587 * G + 0.114 * B);
		}
	}
	return grayscale;
}
//Magnitud obtenida a partir de 2 matrices
Mat obtenerMagnitud(Mat m1, Mat m2) {
	// m1 y m2 deben tener valores iguales
	Mat res(m1.rows, m1.cols, CV_8UC1);
	int x, y = 0;
	for (int i = 0; i < m1.rows; i++) {
		for (int j = 0; j < m1.cols; j++) {

			x = m1.at<uchar>(Point(j, i));
			y = m2.at<uchar>(Point(j, i));

			res.at<uchar>(Point(j, i)) = uchar(sqrt(pow(x, 2) + pow(y, 2)));
		}
	}
	return res;
}

vector<vector<float>> mascaraGy() {
	//Se crea la mascara de la matrizGy con valores determinados
    vector<vector<float>> mascara(3, vector<float>(3, 0));

    mascara[0][0] = -1;
    mascara[0][1] = -2;
    mascara[0][2] = -1;

    mascara[1][0] = 0;
    mascara[1][1] = 0;
    mascara[1][2] = 0;

    mascara[2][0] = 1;
    mascara[2][1] = 2;
    mascara[2][2] = 1;

    return mascara;
}

vector<vector<float>> mascaraGx() {
	//Se crea la mascara de la matrizGx con valores determinados
    vector<vector<float>> mascara(3, vector<float>(3, 0));

    mascara[0][0] = -1;
    mascara[0][1] = 0;
    mascara[0][2] = 1;

    mascara[1][0] = -2;
    mascara[1][1] = 0;
    mascara[1][2] = 2;

    mascara[2][0] = -1;
    mascara[2][1] = 0;
    mascara[2][2] = 1;

    return mascara;
}


//  Muestra la direccion de los bordes para cada pixel dados los gradientes en x y y,
vector<vector<float>> getEdgeAngles(Mat gx, Mat gy) {
	//Obtener la direccion
	// m1 y m2 tienen dimensiones iguales
	vector<vector<float>> angles(gx.rows, vector<float>(gx.cols, 0));
	int x, y = 0;
	for (int i = 0; i < gx.rows; i++) {
		for (int j = 0; j < gx.cols; j++) {

			x = gx.at<uchar>(Point(j, i));
			y = gy.at<uchar>(Point(j, i));
			// el borde en grados es obtnenido en la sig
			angles[i][j] = (atan2(y, x) * 180)/pi;			
		}
	}
	return angles;
}

// Frecuencias acumuladas para cada valor de pixel
vector<int> obtenerFrecuencias(Mat img) {
	vector<int> cdf(256, 0);

	int min = 256;
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++) {
			int tmp = img.at<uchar>(Point(j, i));
			cdf[tmp] += 1;
		}
	}
	// Calculamps frecuencais acumuladas para pixeles presentes
	int acc = 0;
	for (int i = 0; i<256; i++)
	{
		int cdfVal = cdf[i];
		if (cdfVal!= 0) {
			cdf[i] += acc;
			acc = cdf[i];
		}
	}

		return cdf;
	}

// Funcion D¿de ecualizar
// Ecualizacion de Histograma
vector<int> obtenerHistograma(vector<int> cdf, int sideLength) {
	vector<int> Hv(256,0);
	int mn = sideLength * sideLength;
	for (int i = 0;i<256; i++)
	{
		int cdfVal = cdf[i];
		float val = (static_cast<float>(cdfVal) / static_cast<float>(mn)) * 255.0;
		Hv[i] = round(val);

	}
	return Hv;
}

Mat aplicarEcualizacion(Mat m1, vector<int> Hv) {
	// asumiendo que m1 y m2 tienen dimensiones iguales
	//Imagen ecualizada (aplicamos la ecualizacion)
	Mat res(m1.rows, m1.cols, CV_8UC1);
	int x = 0;
	for (int i = 0; i < m1.rows; i++) {
		for (int j = 0; j < m1.cols; j++) {

			x = static_cast<int>(m1.at<uchar>(Point(j, i)));

			res.at<uchar>(Point(j, i)) = uchar(Hv[x]);
		}
	}
	return res;
}
Mat nonMaxSupression(Mat sobel, vector<vector<float>> angulos) {
	//Identificar maximo local en el gradiente
	//Verificar si el pixel central es el maximo en su direccion 
	int filas = sobel.rows;
	int columnas = sobel.cols;

	Mat resultado(filas, columnas, CV_8UC1);

	for (int i = 0; i < filas; i++) {
		for (int j = 0; j < columnas; j++) {
			int pix1 = 255;
			int pix2 = 255;

			//si el angulo es 0 o 180 grados, obtener intensidad de izquierda y derecha
			if ((0 <= angulos[i][j] < 22.5) || (157.5 <= angulos[i][j] <= 180)) {
				pix1 = sobel.at<uchar>(Point(i, j + 1));
				pix2 = sobel.at<uchar>(Point(i, j - 1));
			}

			// esquinas a 45
			else if (22.5 <= angulos[i][j] < 67.5) {
				pix1 = sobel.at<uchar>(Point(i + 1, j - 1));
				pix2 = sobel.at<uchar>(Point(i - 1, j + 1));
			}

			// arriba y abajo a 90
			else if (67.5 <= angulos[i][j] < 112.5) {
				pix1 = sobel.at<uchar>(Point(i + 1, j));
				pix2 = sobel.at<uchar>(Point(i - 1, j));
			}

			//esquinas restantes a 135
			else if (112.5 <= angulos[i][j] < 157.5) {
				pix1 = sobel.at<uchar>(Point(i - 1, j - 1));
				pix2 = sobel.at<uchar>(Point(i + 1, j + 1));
			}

			//Si las intesidades de los lados son menores, se mantiene la intensidad, otro caso se vuelve cero
			if ((sobel.at<uchar>(Point(i, j)) >= pix1) && sobel.at<uchar>(Point(i, j)) >= pix2) {
				resultado.at<uchar>(Point(i, j)) = sobel.at<uchar>(Point(i, j));
			}
			else {
				resultado.at<uchar>(Point(i, j)) = uchar(0);
			}
		}
	}

	return resultado;
}



int intensidadMaxima(Mat img) {
	//Obtener el valor mas fuerte en toda la imagen
	int filas = img.rows;
	int columnas = img.cols;

	int max = 0;

	for (int i = 0; i < filas; i++) {
		for (int j = 0; j < columnas; j++) {
			if (img.at<uchar>(Point(i, j)) > max) {
				max = img.at<uchar>(Point(i, j));
			}
		}
	}

	return max;
}

Mat hysteresis(Mat imgNonMaxSupr, float UmbralSuperiorPor, float UmbralInferiorPor) {
	//Obtener bordes fuertes y los que no son tan notorios prescindir el valor
	int filas = imgNonMaxSupr.rows;
	int columnas = imgNonMaxSupr.cols;

	Mat imgHysteresis(filas, columnas, CV_8UC1);

	float UmbralSuperior, UmbralInferior;

	UmbralSuperior = intensidadMaxima(imgNonMaxSupr) * UmbralSuperiorPor; 
	UmbralInferior = UmbralSuperior * UmbralInferiorPor; 


	for (int i = 0; i < filas; i++) {
		for (int j = 0; j < columnas; j++) {
		// si el pixel supera el umbral
			if (imgNonMaxSupr.at<uchar>(Point(i, j)) >= UmbralSuperior) {
				imgHysteresis.at<uchar>(Point(i, j)) = 255;
			}
			
			else {
				imgHysteresis.at<uchar>(Point(i, j)) = 0;
			}
		}
	}

	return imgHysteresis;
}


int main()
{
	float sigma = 1;
	int kSize = 3;
	cout << "Ingresa el tamano del kernel" << endl;
	cin >> kSize;

	if (kSize % 2 == 0 || kSize <= 0 ) {
		cout << "Valor de kernel invalido" << endl;
		exit(0);
	}

	cout << "Ingresa sigma" << endl;
	cin >> sigma;

	if (sigma <= 0) {
		cout << "Valor de sigma invalido" << endl;
		exit(0);
	}

	char NombreImagen[] = "lena.jpg";
	Mat imagen;

	vector<vector<float>> gx = mascaraGx();
	vector<vector<float>> gy = mascaraGy();

	imagen = imread(NombreImagen);

	if (!imagen.data)
	{
		cout << "Error al cargar la imagen: " << NombreImagen << endl;
		exit(0);
	}

	// Procesamiento
	int fila_original = imagen.rows;
	int columna_original = imagen.cols;

	Mat grayscaleImg = GrisesANtsc(imagen);

	vector<vector<float>> gKernel = KernelGauss(kSize, sigma);
	Mat filtrada = FiltroAMatriz(grayscaleImg, gKernel);

	//Concatenar filtros
	vector<int> cdf2 = obtenerFrecuencias(filtrada);
	vector<int> hv2 = obtenerHistograma(cdf2, fila_original);
	Mat gaussianEq = aplicarEcualizacion(filtrada, hv2);
	Mat geqGx = FiltroAMatriz(gaussianEq, gx);
	Mat geqGy = FiltroAMatriz(gaussianEq, gy);
	Mat geqG = obtenerMagnitud(geqGx, geqGy);
	vector<vector<float>> angles2 = getEdgeAngles(geqGx, geqGy);
	Mat nonmax2 = nonMaxSupression(geqG, angles2);
	Mat hys2 = hysteresis(nonmax2, 0.9, 0.35);
	//
	namedWindow("Imagen original", WINDOW_AUTOSIZE);
	imshow("Imagen original", imagen);

	namedWindow("Escala de grises", WINDOW_AUTOSIZE);
	imshow("Escala de grises", grayscaleImg);

	namedWindow("Imagen filtrado", WINDOW_AUTOSIZE);
	imshow("Imagen filtrado", filtrada);

	namedWindow("Gradiantex", WINDOW_AUTOSIZE);
	imshow("Gradiantex", geqGx);

	namedWindow("Gradiantey", WINDOW_AUTOSIZE);
	imshow("Gradiantey", geqGy);

	namedWindow("GradianteSobel", WINDOW_AUTOSIZE);
	imshow("GradianteSobel", geqG);

	namedWindow("EcualizacionGaussiana", WINDOW_AUTOSIZE);
	imshow("EcualizacionGaussiana", gaussianEq);

	namedWindow("hysteresis", WINDOW_AUTOSIZE);
	imshow("hysteresis", hys2);
	waitKey(0);
	return 0;
}