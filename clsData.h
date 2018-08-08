#include <iostream>
#include <string.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cstring>

using namespace std;

class clsData
{
    private:
        int r;
        int c;
        vector< vector<double> > dataset;
        vector<double> v;
        vector<int> clsC;
        int nClsC;

    public:
        clsData ();
        ~clsData ();
        void initialize (char*);
        void initialize (vector< vector<double> >&, int, int);
        int getRow () { return r; }
        int getCol () { return c; }
        vector< vector<double> > getData () { return dataset; }
        vector<double> getVari () { return v; }
        vector<int> getCls () { return clsC; }
        int getnCls () { return nClsC; }
};
