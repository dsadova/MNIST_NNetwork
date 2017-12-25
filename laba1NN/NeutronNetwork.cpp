#include "stdafx.h"
#include "NeutronNetwork.h"

NNetwork::NNetwork()
{
}
NNetwork::NNetwork(int hidden, int epochs, double speed, double stop)
{
	m_hidden = hidden;
	m_epochs = epochs;
	m_speed = speed;
	m_stop = stop;


	hiddenWeights.resize(m_Input);
	for (int i = 0; i != hiddenWeights.size(); i++)
		hiddenWeights[i].resize(m_hidden);

	outputWeights.resize(m_hidden);
	for (int i = 0; i != outputWeights.size(); i++) {
		outputWeights[i].resize(m_Output);
	}

	InitWeights();

	hGradient.resize(hidden);
	oGradient.resize(m_Output);
}
NNetwork::~NNetwork()
{
}

double NNetwork::Accuracy(const std::vector<std::vector<double>>DataSet, const std::vector<double> TestLabels) 
{
	std::vector<double> picture;
	std::vector<double> labels;
	int numTrue = 0, numFalse = 0;
	for (size_t i = 0; i < DataSet.size(); i++){ //для каждой картинки

		picture.resize(m_Input);
		labels.resize(m_Output);
		for (size_t j = 0; j != m_Input; j++)
			picture[j] = DataSet[i][j]; //сформировали входной слой

		for (int j = 0; j < m_Output; j++){
			double l = 0.0;
			if (TestLabels[i] == j)
				l = 1.0;
			labels[j] = l;
		}
		inputFirstLayer = std::make_pair(picture, labels);
		picture.clear();
		labels.clear();

		InputForHiddenLayer(inputFirstLayer.first);
		MainOutput();
		int answer = Max(m_tmpOutput);
		if (inputFirstLayer.second[answer] == 1.0)
			numTrue++;
		else
			numFalse++;
		PrepareDataAtFirstLayer.clear();
		m_tmpOutput.clear();
	}
	return (double)numTrue / (double)(numTrue + numFalse);
}

double NNetwork::CrossEntropy(const std::vector<std::vector<double>>DataSet, const std::vector<double> TestLabels)

{

	double err = 0.0;

	std::vector<double> picture;
	std::vector<double> labels;
	for (size_t i = 0; i < DataSet.size(); i++){ //для каждой картинки

		picture.resize(m_Input);
		labels.resize(m_Output);
		for (size_t j = 0; j != m_Input; j++)
			picture[j] = DataSet[i][j]; //сформировали входной слой

		for (int j = 0; j < m_Output; j++){
			double l = 0.0;
			if (TestLabels[i] == j)
				l = 1.0;
			labels[j] = l;
		}
		inputFirstLayer = std::make_pair(picture, labels);
		picture.clear();
		labels.clear();

		InputForHiddenLayer(inputFirstLayer.first);
		MainOutput();

		for (int j = 0; j < m_Output; j++)
			err += log(m_tmpOutput[j]) * inputFirstLayer.second[j];

	}
	return -1.0 * err / DataSet.size();
}

int NNetwork::Max(std::vector<double> vec) {

	int num = 0;
	double tmp = vec[0];
	for (int j = 0; j != vec.size(); j++)
		if (vec[j] > tmp - 0.0001){
			tmp = vec[j];
			num = j;
		}
	return num;
}

double NNetwork::hyperbTan(double x) {
	if (x < -20.0) 
		return -1.0;
	else if (x > 20.0)
		return 1.0;
	return tanh(x);
}

std::vector<double> NNetwork::softmax(std::vector<double> z)
{
	std::vector<double> rez;
	rez.resize(z.size());
	double sum = 0.;

	for (size_t i = 0; i != z.size(); i++)
		sum += exp(z[i]);

	for (size_t j = 0; j != z.size(); j++)
		rez[j] = exp(z[j]) / sum;

	return rez;
}

void NNetwork::InitWeights()
{
	for (int i = 0; i < m_Input; i++)
		for (int j = 0; j < m_hidden; j++)
			hiddenWeights[i][j] = (double)((rand() / double(RAND_MAX)) / 100.0);


	for (int i = 0; i < m_hidden; i++)
		for (int j = 0; j < m_Output; j++)
			outputWeights[i][j] = (double)((rand() / double(RAND_MAX)) / 100.0);

	m_b1.resize(m_hidden);
	for (size_t b = 0; b != m_b1.size(); b++)
		m_b1[b] = (double)((rand() / double(RAND_MAX)) / 100.0);

	m_b2.resize(m_Output);
	for (size_t b = 0; b != m_b2.size(); b++)
		m_b2[b] = (double)((rand() / double(RAND_MAX)) / 100.0);
}

std::vector<double> NNetwork::InputForHiddenLayer(std::vector<double> hiddenInput)
{

	PrepareDataAtFirstLayer.resize(m_hidden, 0.);

	for (int j = 0; j != m_hidden; j++)
		for (size_t i = 0; i != hiddenInput.size(); i++)
			PrepareDataAtFirstLayer[j] += hiddenInput[i] * hiddenWeights[i][j];

	for (int i = 0; i < m_hidden; i++)
		PrepareDataAtFirstLayer[i] += m_b1[i];

	return PrepareDataAtFirstLayer;
}

void NNetwork::Gradient(std::vector <double> labels) {
	for (int i = 0; i != oGradient.size(); i++)
		oGradient[i] = (labels[i] - m_tmpOutput[i]);

	for (int i = 0; i != hGradient.size(); i++) {
		double derivative = (1 - PrepareDataAtFirstLayer[i]) * (1 + PrepareDataAtFirstLayer[i]);
		double sum = 0.0;
		for (int j = 0; j < m_Output; j++) {
			sum += oGradient[j] * outputWeights[i][j];
		}
		hGradient[i] = derivative * sum;
	}
}

std::vector <double> NNetwork::MainOutput()
{
	std::vector <double> GlobalOutput;
	GlobalOutput.resize(m_Output, 0.);
	for (auto& fl : PrepareDataAtFirstLayer)
		fl = hyperbTan(fl);
	
	for (int j = 0; j < m_Output; j++)
		for (int i = 0; i < m_hidden; i++)
			GlobalOutput[j] += PrepareDataAtFirstLayer[i] * outputWeights[i][j];

	for (int i = 0; i < m_Output; i++)
		GlobalOutput[i] += m_b2[i];
	m_tmpOutput = softmax(GlobalOutput);
	return m_tmpOutput;

}

void NNetwork::TrainingAndLookingForAccuracy(std::vector<std::vector <double>> DataSet, std::vector<double> DataLabels, const std::vector<std::vector <double>> TestSet, const std::vector<double> TestLabels)
{
	
	std::vector<double> picture;
	std::vector<double> labels;
	int epoch = 0, numTrue = 0, numFalse = 0;
	for (epoch = 0; epoch < m_epochs; epoch++) {
		std::cout << "epoch = " << epoch << "\n";

		Shuffle(DataSet, DataLabels);
		for (size_t i = 0; i < DataSet.size(); i++){ //для каждой картинки

			picture.resize(m_Input);
			labels.resize(m_Output);
			for (size_t j = 0; j != m_Input; j++)
				picture[j] = DataSet[i][j]; //сформировали входной слой

			for (int j = 0; j < m_Output; j++){
				double l = 0.0;
				if (DataLabels[i] == j)
					l = 1.0;
				labels[j] = l;
			}

			inputFirstLayer = std::make_pair(picture, labels);
			picture.clear();
			labels.clear();

			InputForHiddenLayer(inputFirstLayer.first);
			MainOutput();

			Gradient(inputFirstLayer.second);
			NewWeights();
			NewBias();
			PrepareDataAtFirstLayer.clear();
			m_tmpOutput.clear();
		}

		double trainingAccuracy = Accuracy(DataSet, DataLabels);
		double testAccuracy = Accuracy(TestSet, TestLabels);
		std::cout << "Accuracy of training set: " << trainingAccuracy << " Accuracy of test set: " << testAccuracy << std::endl; 
		double mcee = CrossEntropy(DataSet, DataLabels);
		if (mcee < m_stop)
			break;
		PrepareDataAtFirstLayer.clear();
		m_tmpOutput.clear();
	}
}

void NNetwork::NewBias()
{
	for (int i = 0; i != m_b1.size(); i++) 
		m_b1[i] += m_speed * hGradient[i] * 1.0;

	for (int i = 0; i != m_b2.size(); i++)
		m_b2[i] += m_speed * oGradient[i] * 1.0;
}

void NNetwork::NewWeights() {
	for (int i = 0; i != hiddenWeights.size(); ++i)
		for (int j = 0; j != hiddenWeights[0].size(); ++j)
			hiddenWeights[i][j] += (m_speed * hGradient[j]) * inputFirstLayer.first[i];

	for (int i = 0; i != outputWeights.size(); ++i)
		for (int j = 0; j != outputWeights[0].size(); ++j)
			outputWeights[i][j] += m_speed * oGradient[j] * PrepareDataAtFirstLayer[i];
}

void NNetwork::Shuffle(std::vector <std::vector <double>> Dataset, std::vector <double> Labels)
{
	for (int i = 0; i != Dataset.size(); i++)
	{
		int nom1 = rand() % Dataset.size();
		int nom2 = rand() % Dataset.size();
		std::swap(Dataset[nom1], Dataset[nom2]);
		std::swap(Labels[nom1], Labels[nom2]);
	}
}