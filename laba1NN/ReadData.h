#pragma once

#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <math.h>
#include <iostream>

class ReadData
{
public:
	ReadData();
	~ReadData();
	void ReadMnistData(std::string fileName, std::vector<std::vector<double>> &dataSet);
	void ReadMnistLabels(std::string fileName, std::vector<double> &dataSet);
private:
	int ReverseInt(int i);

};

