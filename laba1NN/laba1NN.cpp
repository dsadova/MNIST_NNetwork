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

int main(int argc, char** argv[])
{
	if (argc < 1) {
		std::cout << "Number of arguments is not right. You should write: " << std::endl;
		std::cout << "1: number of hidden neurons: " << std::endl;
		std::cout << "2: number of epochs: " << std::endl;
		std::cout << "3: speed: " << std::endl;
		std::cout << "4: stop criterion: " << std::endl;
		return 0;
	}


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
	if (argc > 1) {
		Hidden = atoi(*argv[1]);
		NumberOfEpochs = atoi(*argv[2]);
		Speed = atof(*argv[3]);
		StopCriterion = atof(*argv[4]);
	}
	NNetwork net(Hidden, NumberOfEpochs, Speed, StopCriterion);
	net.Training(DataSet, DataLabels, TestSet, TestLabels);
	
	/*double trainingAccuracy = net.Accuracy(DataSet, DataLabels);
	double testAccuracy = net.Accuracy(TestSet, TestLabels);
	std::cout << "Accuracy of training set: " << trainingAccuracy << "Accuracy of test set: " << testAccuracy << std::endl;*/
	return 0;
}

