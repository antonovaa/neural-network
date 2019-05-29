// NNUnreal.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include "pch.h"
#include <iostream>
#include "NeuralNetwork.h"
#include "ReaderParams.h"


int main()
{
	int inputCount = 3;
	int outputCount = 2;	
	int hiddenLayers[] = { 4,3 };
	int countLayers = sizeof(hiddenLayers) / sizeof(*hiddenLayers);
	vector<InputTrainParams*> inputTrain= ReaderParams::getParamsInputOutput("D:\\Development\\NNUnreal\\train.txt", inputCount, outputCount);

	
	NeuralNetwork *nn = new NeuralNetwork(outputCount, inputCount, countLayers, hiddenLayers, 0.1f);
	
	nn->trainNN(inputTrain,100000);
	cout << "---------------------------------------------" << endl;
	nn->showWeight();

	float test[3] = { 0.4,0.2,0.1 };
	nn->test(test);




    std::cout << "Hello NN!\n"; 
}