#pragma once
#include <vector>
#include "InputTrainParams.h"

class NeuralNetwork
{
public:
	NeuralNetwork(int outputCount, int inputCount, const int  countHiddenLayers, int *hiddenLayers,  float h);
	void trainNN(vector<InputTrainParams*> inputTrain, int iteration);
	~NeuralNetwork();
	
	void test(float *input);
	float sigmoid(float in) {
		return (float)(1 / (1 + exp(-in)));
	}
	float mse(float* tgt) const {
		float mse = 0.0;
		//for (int i = 0; i < neuronCnt[layerCnt - 1]; i++) {
		//	mse += (tgt[i] - Out(i)) * (tgt[i] - Out(i));
		//}
		return mse;// neuronCnt[layerCnt - 1];
	}
	void feedForward(int op, float *input);
	void backPropogation(int op, float *output);
	float Out(int op);
	void showWeight();
private:
	float sum;
	float h;
	//vector<InputTrainParams*> inputForTrain;
	//int numberOutputParam;
	int *layers;
	//int countInputParam;
	int countLayers;
	float ****weight;//op--count outputs params, l-count layers,i-current neural, j-priv neural;
	float ***delta;
	float ***outNneuron;
	int outputCount;
	int inputCount;
};

