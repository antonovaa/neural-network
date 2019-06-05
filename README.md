# neural-network
# simple neural network

#for start:))
 #set:
  int countIteration=100000;
  int inputCount = 3;
	int outputCount = 2;	
	int hiddenLayers[] = { 4,3 };
	int countLayers = sizeof(hiddenLayers) / sizeof(*hiddenLayers);
  # create vector data for train
  # file suitable :
     a b c d e
     0.3 0.2 0.1 0.6 0.4
    # 0.1 0.1 0.1 0.3 0.1
    # 0.2 0.1 0.0 0.3 0.3
    # 0.1 0.0 0.0 0.1 0.1
    # 0.1 0.2 0.2 0.5 0.1
    # 0.4 0.0 0.1 0.5 0.3
    # 0.0 0.3 0.2 0.5 0.1
    # 0.2 0.3 0.4 0.9 0.1
# a b c--input; d(a+b+c) e(a+b-c) -- output
	vector<InputTrainParams*> inputTrain= ReaderParams::getParamsInputOutput("D:\\Development\\NNUnreal\\train.txt", inputCount, outputCount);

	
	NeuralNetwork *nn = new NeuralNetwork(outputCount, inputCount, countLayers, hiddenLayers, 0.1f);
	
	nn->trainNN(inputTrain,countIteration);
	cout << "---------------------------------------------" << endl;
	nn->showWeight();

	float test[3] = { 0.4,0.2,0.1 };
	nn->test(test);
