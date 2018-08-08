#include <iostream>
#include <string.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <algorithm>
#include <math.h>
#include <fstream>

using namespace std;

class clsWeka
{
    public:
        clsWeka();
        string* classify(vector<int>, char*, int);
    protected:
    private:
        string classifiers[9];
        string clsAcc[11];
	double acc[9];
};
