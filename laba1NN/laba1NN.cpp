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

	std::cout << "You can write 1-4 arguments for network: " << std::endl;
	std::cout << "number of hidden neurons, number of epochs, speed, stop criterion" << std::endl;


	std::string trainImageMNIST = "train-images.idx3-ubyte";
	std::string trainLabelsMNIST = "train-labels.idx1-ubyte";
	std::string testImageMNIST = "t10k-images.idx3-ubyte";
	std::string testLabelsMNIST = "t10k-labels.idx1-ubyte";

	int NumberOfImages = 60000;
	int ImageSize = 28 * 28;



	std::vector<std::vector<double>> DataSet;
	std::vector<double> DataLabels;
	std::vector<std::vector <double>> TestSet;
	std::vector<double> TestLabels;

	ReadData rd;
	rd.ReadMnistData(trainImageMNIST, DataSet);
	rd.ReadMnistLabels(trainLabelsMNIST, DataLabels);
	rd.ReadMnistData(testImageMNIST, TestSet);
	rd.ReadMnistLabels(testLabelsMNIST, TestLabels);

	int Hidden = 300;
	int NumberOfEpochs = 15;
	double Speed = 0.01;
	double StopCriterion = 0.005;
	switch (argc){
	case 1:
		std::cout << "number of hidden neurons = 300, number of epochs = 15, speed = 0.01, stop criterion = 0.005" << std::endl;
		break;
	case 2:
		Hidden = atoi(argv[1]);
		break;
	case 3:
		Hidden = atoi(argv[1]);
		NumberOfEpochs = atoi(argv[2]);
		break;
	case 4:
		Hidden = atoi(argv[1]);
		NumberOfEpochs = atoi(argv[2]);
		Speed = atof(argv[3]);
		break;
	case 5:
		Hidden = atoi(argv[1]);
		NumberOfEpochs = atoi(argv[2]);
		Speed = atof(argv[3]);
		StopCriterion = atof(argv[4]);
		break;
	default:
		Hidden = atoi(argv[1]);
		NumberOfEpochs = atoi(argv[2]);
		Speed = atof(argv[3]);
		StopCriterion = atof(argv[4]);
		break;
	}

	NNetwork net(Hidden, NumberOfEpochs, Speed, StopCriterion);
	net.TrainingAndLookingForAccuracy(DataSet, DataLabels, TestSet, TestLabels);

	return 0;
}