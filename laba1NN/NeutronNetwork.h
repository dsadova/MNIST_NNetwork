#pragma once
#include "vector"
#include <iostream>
class NNetwork
{

public:
	NNetwork();
	NNetwork(int hidden, int epochs, double speed, double stop);

	~NNetwork();
	void Training(std::vector<std::vector <double>> DataSet, std::vector<double> DataLabels, const std::vector<std::vector <double>> TestSet, const std::vector<double> TestLabels);
	double Accuracy(const std::vector<std::vector<double>>DataSet, const std::vector<double> TestLabels);

private:
	const int m_Input = 28 * 28;
	const int m_Output = 10;

	std::vector<std::vector<double>> hiddenWeights;
	std::vector<std::vector<double>> outputWeights;
	std::vector<double> m_b1;
	std::vector<double> m_b2;

	int m_hidden = 200, m_epochs = 15;
	double m_speed = 0.008, m_stop = 0.005;

	std::vector <double> PrepareDataAtFirstLayer;
	std::vector <double> m_tmpOutput;

	double hyperbTan(double x);
	std::vector<double> softmax(std::vector<double> z);
	std::vector<double> hGradient, oGradient;
	void NewBias(std::vector <double> labels);
	void NewWeights(std::vector <double> labels);
	void Gradient(std::vector <double> labels);
	void Shuffle(std::vector <std::vector <double>> Dataset, std::vector <double> Labels);

	void InitWeights();
	std::vector<double> InputForHiddenLayer(std::vector<double> MainInput);
	std::vector <double> MainOutput();
	int Max(std::vector<double> vec);
	double CrossEntropy(const std::vector<std::vector<double>>DataSet, const std::vector<double> TestLabels);
	std::pair<std::vector<double>, std::vector<double>> inputFirstLayer; //пара: один - входная картинка; второй - цифра на пикче
};

