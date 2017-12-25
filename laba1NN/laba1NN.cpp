// laba1NN.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <cmath>
#include "vector"
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <cmath>

#include "ReadData.h"
#include "NeutronNetwork.h"

int main(int argc, char* argv[])
{
	std::cout << "Write 2 paths to train and test data: " << std::endl;
	std::string PathForTrainSet;
	std::string PathForTestSet;
	if (argc < 3)
	{
		std::cout << "Error: " << std::endl;
		std::cout << "Program need to read data. Write 2 paths for train and test data." << std::endl;
	}
	else {
		PathForTrainSet = argv[1];
		PathForTestSet = argv[2];
	}
	std::cout << "You can write 1-4 arguments for network: " << std::endl;
	std::cout << "number of hidden neurons, number of epochs, speed, stop criterion" << std::endl;


	int NumberOfImages = 60000;
	int ImageSize = 28 * 28;

	std::vector<std::vector<double>> DataSet;
	std::vector<double> DataLabels;
	std::vector<std::vector <double>> TestSet;
	std::vector<double> TestLabels;

	std::string trainImageMNIST = PathForTrainSet + "train-images.idx3-ubyte";
	std::string trainLabelsMNIST = PathForTrainSet + "train-labels.idx1-ubyte";
	std::string testImageMNIST = PathForTestSet + "t10k-images.idx3-ubyte";
	std::string testLabelsMNIST = PathForTestSet + "t10k-labels.idx1-ubyte";

	int Hidden = 300;
	int NumberOfEpochs = 15;
	double Speed = 0.01;
	double StopCriterion = 0.005;
	switch (argc){
	case 3:
		std::cout << "number of hidden neurons = 300, number of epochs = 15, speed = 0.01, stop criterion = 0.005" << std::endl;
		break;
	case 4:
		Hidden = atoi(argv[3]);
		break;
	case 5:
		Hidden = atoi(argv[3]);
		NumberOfEpochs = atoi(argv[4]);
		break;
	case 6:
		Hidden = atoi(argv[3]);
		NumberOfEpochs = atoi(argv[4]);
		Speed = atof(argv[5]);
		break;
	case 7:
		Hidden = atoi(argv[3]);
		NumberOfEpochs = atoi(argv[4]);
		Speed = atof(argv[5]);
		StopCriterion = atof(argv[6]);
		break;
	}

	ReadData rd;
	rd.ReadMnistData(trainImageMNIST, DataSet);
	rd.ReadMnistLabels(trainLabelsMNIST, DataLabels);
	rd.ReadMnistData(testImageMNIST, TestSet);
	rd.ReadMnistLabels(testLabelsMNIST, TestLabels);

	NNetwork net(Hidden, NumberOfEpochs, Speed, StopCriterion);
	net.TrainingAndLookingForAccuracy(DataSet, DataLabels, TestSet, TestLabels);

	return 0;
}