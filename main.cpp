#include <iostream>
#include <string.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include "clsData.h"
#include <algorithm>
#include <math.h>
#include <iomanip>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <pthread.h>
#include "clsWeka.h"
#include <stdint.h>

using namespace std;
using namespace std::chrono;
#define _USE_MATH_DEFINES

int NUM_THREADS = 4;

//https://software.intel.com/en-us/articles/fast-random-number-generator-on-the-intel-pentiumr-4-processor/
static unsigned int g_seed;
//Used to seed the generator.
inline void fast_srand(int seed)
{
    g_seed = seed;
}
//fastrand routine returns one integer, similar output value range as C lib.
inline int fastrand(int bound)
{
    g_seed = (214013*g_seed+2531011);
    return (g_seed>>16)&bound;//&0x7FFF;
}

struct threadData {
  int r, c, nCls, lF, tID;
  vector<int>* cls;
  vector<double>* v;
  vector<vector<double> >* data;
  vector<int>* maxF;
};

struct threadEvo {
  int q, m, n, Smax, nF, N, r, c, nCls, stalled, im, tID;
  vector<int>* cls;
  vector<double>* v;
  vector<vector<double> >* data;
  vector< vector<double> >* Y;
  vector<double>* Pj;
};

vector< vector<double> > generatePop(int m, int n, int r, int c, int nCls, vector<int>& cls, vector<double>& v, vector<vector<double> >& data, vector<double>& normThreshold);
float sqrt7(float x); //http://www.codeproject.com/Articles/69941/Best-Square-Root-Method-Algorithm-Function-Precisi
double dependency(int r, int c, int nCls, vector<int>& cls, vector<double>& v, vector<vector<double> >& data, vector<int>& maxF, int lF);
void *ddWrapper(void *threadArg);
void *evoWrapper(void *threadArg);
double dependency(int r, int c, int nCls, vector<int>& cls, vector<double>& v, double *data, vector<int>& maxF, int lF);
int linear_search(int arr[], int n, int val); //http://www.quepublishing.com/articles/article.aspx?p=25281
vector< vector<double> > rankFrogs(vector< vector<double> >& F, int m, int n);
double meanDD(vector< vector<double> >& F, int idx);
vector< vector<double> > partitionFrogs(vector< vector<double> >& X, int m, int n, int c);
vector< vector<double> > memEvolution(vector< vector<double> >& Y, int q, int m, int n, int Smax, int nF, int N, vector<vector<double> >& data, int r, int c, int nCls, vector<int>& cls, vector<double>& v, int stalled);
vector<int> vXor (vector<double>& vec1, vector<double>& vec2);

bool compDDxRed(const vector<double>& p1, const vector<double>& p2)
{
    int ddPos = p1.size();
    ddPos -= 2;
    int lenPos = ddPos + 1;

    if (p1[ddPos]*(1-p1[lenPos]/ddPos) > p2[ddPos]*(1-p2[lenPos]/ddPos))
        return true;

    return false;
}

bool compDDLen(const vector<double>& p1, const vector<double>& p2)
{
    int ddPos = p1.size();
    ddPos -= 2;
    int lenPos = ddPos + 1;

    if (p1[ddPos]/p1[lenPos] > p2[ddPos]/p2[lenPos]) {return true;}

    return false;
}

bool compDD(const vector<double>& p1, const vector<double>& p2)
{
    int ddPos = p1.size();
    ddPos -= 2;

    if (p1[ddPos] > p2[ddPos]) {return true;}

    return false;
}

bool compLen(const vector<double>& p1, const vector<double>& p2)
{
    int ddPos = p1.size();
    ddPos -= 2;
    int lenPos = ddPos + 1;

    if (p1[lenPos] < p2[lenPos]) {return true;}

    return false;
}

ostream& operator<<(ostream& os, vector<double>& vec)
{
    for (double b : vec)
    {
        os << b << ", ";
    }
    return os;
}

ostream& operator<<(ostream& os, vector<int>& vec)
{
    for (int b : vec)
    {
        os << b << ", ";
    }

    os << '\b' << '\b';

    return os;
}

ostream& operator<<(ostream& os, vector< vector<double> >& vec)
{
    for (vector<double> a : vec)
    {
        for (double b : a)
        {
            os << b << ", ";
        }
        os << endl;
    }
    return os;
}

ostream& operator<<(ostream& os, vector< vector<int> >& vec)
{
    for (vector<int> a : vec)
    {
        for (int b : a)
        {
            os << b << ", ";
        }
        os << endl;
    }
    return os;
}

int main(int argc, char* argv[])
{
    srand (time(NULL));
    fast_srand(rand());
    time_t currentTime;
    struct tm *localTime;
    time( &currentTime );
    localTime = localtime(&currentTime);
    int Hour = localTime->tm_hour;
    int Min = localTime->tm_min;
    int Sec = localTime->tm_sec;
    int maxIter = 2;
    int classSize = 9;
    string *classRes;
    char *filename;
    filename = argv[1];
    int stalled = atof(argv[2]);
    double mu, var = 0.0;
    clsWeka *classWeka = new clsWeka;
	vector< vector<double> > F;
	vector< vector<double> > X;
	vector< vector<double> > Y;

    if (argv[3]!="")
        NUM_THREADS = atof(argv[3]);

    ofstream finalRes;

    cout << "Reading " << filename << " dataset" << endl;
    clsData *spec = new clsData;
    spec->initialize(filename);
    int r = spec->getRow();
    int c = spec->getCol();
    vector< vector<double> > dataset = spec->getData();
    vector<double> v = spec->getVari();
    vector<int> cls = spec->getCls();
    int nCls = spec->getnCls();

	//For small datasets use these parameters
    //int Smax = floor((c-2)/2);
    //int m = floor((c-2)*2.2);
    //int n = floor((c-2)*.7);
    //int q = floor((c-2)*.45);
    //int N = floor((c-2)*.5);

    int Smax = floor((c-1)*.45);
    int m = 30;
    int n = 30;
    int q = 5;
    int N = 15;

    if ((q<1)||(q>n))
    {
        cout << "Invalid value for q!" << endl;
        return 0;
    }

    //Generating normal distribution threshold for generating individuals
    mu = (m*n+1) / 2.0;
    for (int i=0; i<m*n; i++)
    {
        var += pow(i*+1.0-mu, 2.0);
    }
    var /= (m*n);

    vector<double> normThreshold(m*n);
    for (int i=0; i<m*n; i++)
    {
        normThreshold[i] = 1.0/(sqrt(var*2.0*M_PI)) * exp(-pow((i+1.0-mu), 2.0)/(2.0*var));
    }

    double minThreshold = *min_element(normThreshold.begin(), normThreshold.end());
    double maxThreshold = *max_element(normThreshold.begin(), normThreshold.end());

    for (int i=0; i<m*n; i++)
    {
        normThreshold[i] = (normThreshold[i]-minThreshold)/(maxThreshold-minThreshold);
    }
    //-------------------------------------------------------------

    if (argv[4]) {
        maxIter = atof(argv[4]);
	}
	double converg[maxIter][3]; //iteration, best, mean

    string name = "./results/" + ((string)filename).substr(7, ((string)filename).length()-4) + "_" + to_string(Hour) + to_string(Min) + to_string(Sec) + ".txt";
cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<< " <<name;
    finalRes.open (name, ios_base::out);

    for(int iter=0; iter<maxIter; iter++)
    {
        cout << "------------------------- " << "Iteration " <<to_string(iter+1) << "-------------------------" << endl;
        finalRes  << "------------------------- " << "Iteration " <<to_string(iter+1) << "-------------------------" << endl;

        high_resolution_clock::time_point tStart = high_resolution_clock::now();

        cout << "Running SFLA..." << endl;
        cout << " Generating population..." << endl;
        F = generatePop(m, n, r, c, nCls, cls, v, dataset, normThreshold);
        cout << endl;

        cout << " Ranking frogs..." << endl;
        X = rankFrogs(F, m, n);
        vector<double> Px(1);
        Px = X[0];

        cout << " Partition frogs..." << endl;
        Y = partitionFrogs(X, m, n, c);

        cout << " Meme evolution..." << endl;
        Y = memEvolution(Y, q, m, n, Smax, c-1, N, dataset, r, c, nCls, cls, v, stalled);
        cout << endl;

        cout << " Ranking frogs..." << endl;
        Y = rankFrogs(Y, m, n);
		converg[iter][0] = iter;
		converg[iter][1] = Y[0][c-1];
		converg[iter][2] = meanDD(Y, c-1);

        high_resolution_clock::time_point tEnd = high_resolution_clock::now();

        int j = 0;
        vector<int> _selF(Y[0][c]);

        for (int i=0; i<(c-1); i++)
        {
            if (Y[0][i] > 0)
            {
                _selF[j] = i;
                _selF[j]++;
                j++;
            }
        }

        double tDuration = duration_cast<milliseconds>(tEnd-tStart).count()/1000.0;

        cout << "Best subset = [" << _selF << "] DD = " << Y[0][c-1] << ", Len = " << Y[0][c] << endl;
        cout << "Time taken: " << tDuration << " seconds" << endl;

        finalRes << "Best subset = [" << _selF << "] DD = " << Y[0][c-1] << ", Len = " << Y[0][c] << endl;
        finalRes << "Time taken: " << tDuration << " seconds" << endl;

        cout << "Running WEKA..." << endl;
        classRes = classWeka->classify(_selF, filename, c);
        for(j=0; j<classSize+1; j++)
        {
            cout << *(classRes+j) << endl;
            finalRes << *(classRes+j) << endl;
        }
        cout << *(classRes+j) << endl;
        finalRes << *(classRes+j) << endl;
    }

    finalRes  << "------------------------- Convergence -------------------------" << endl;
	finalRes << "Iter" << "," << "DD" << endl;
	for(int iter=0; iter<Y.size(); iter++) {
		finalRes << iter+1 << "," << Y[iter][c-1] << endl;
	}

    finalRes.close();
    delete(spec);
    delete(classWeka);
}

vector< vector<double> > memEvolution(vector< vector<double> >& Y, int q, int m, int n, int Smax, int nF, int N, vector<vector<double> >& data, int r, int c, int nCls, vector<int>& cls, vector<double>& v, int stalled)
{
    int i = 0, j = 0;
    pthread_t threads[NUM_THREADS];
    threadEvo trdEvo[NUM_THREADS];
    vector<double> Pj(n);

    for (i=0; i<n; i++)
    {
        Pj[i] = (2 * (n + 1 - (n - (double)i + 1))) / (n * (n + 1));
    }

    for (i=0;i<NUM_THREADS;i++)
    {
        trdEvo[i].tID = i;
        trdEvo[i].r = r;
        trdEvo[i].c = c;
        trdEvo[i].q = q;
        trdEvo[i].m = m;
        trdEvo[i].n = n;
        trdEvo[i].Smax = Smax;
        trdEvo[i].nF = nF;
        trdEvo[i].N = N;
        trdEvo[i].nCls = nCls;
        trdEvo[i].stalled = stalled;
        trdEvo[i].cls = &cls;
        trdEvo[i].v = &v;
        trdEvo[i].data = &data;
        trdEvo[i].Y = &Y;
        trdEvo[i].Pj = &Pj;
    }

    i = 0;
    while ((i+NUM_THREADS)<m)
    {
        cout << setw(5) << setprecision(5) << "  Memeplex = " << i+1 << " of " << m << "                    "<< '\r';
        cout.flush();
        for (j=0;j<NUM_THREADS;j++)
        {
            trdEvo[j].im = i+j;
            pthread_create(&threads[j], NULL, evoWrapper, (void *)&trdEvo[j]);
        }

        for (j=0;j<NUM_THREADS;j++)
        {
            pthread_join(threads[j], NULL);
        }

        i+=NUM_THREADS;
    }

    for (i;i<m;i++)
    {
        cout << setw(5) << setprecision(5) << "  Memeplex = " << i+1 << " of " << m << "                    "<< '\r';
        cout.flush();
        trdEvo[i%NUM_THREADS].im = i;
        pthread_create(&threads[i%NUM_THREADS], NULL, evoWrapper, (void *)&trdEvo[i%NUM_THREADS]);
        pthread_join(threads[i%NUM_THREADS], NULL);
    }

    return Y;
}

vector<int> vXor (vector<double>& vec1, vector<double>& vec2)
{
    vector<int> out(2*(vec1.size()-1)-1);
    int nOnes = 0, i = 0, nF = vec1.size()-2, k = nF;

    for (i=0; i<nF; i++)
    {
        out[i] = abs(vec1[i]-vec2[i]);
        nOnes += out[i];
        if (out[i] > 0)
        {
            out[k+1] = i;
            k++;
        }
    }

    out[i] = nOnes;

    return out;
}

vector< vector<double> > partitionFrogs(vector< vector<double> >& X, int m, int n, int c)
{
    int k = 0;
    vector<double> yTmp(c+1);
    vector< vector<double> > Y(m*n, yTmp);

    for (int i=0; i<m; i++)
    {
        Y[k] = X[i];

        for (int j=1; j<n; j++)
        {
            k++;
            Y[k] = X[j*m+i];
        }

        k++;
    }

    return Y;
}

vector< vector<double> > generatePop(int m, int n, int r, int c, int nCls, vector<int>& cls, vector<double>& v, vector<vector<double> >& data, vector<double>& normThreshold)
{
    int i = 0, j = 0, len = 0;
    vector< vector<int> > selF(m*n, vector<int>(c-1));
    vector< vector<double> > rndTemp(m*n, vector<double>(c+1));
    pthread_t threads[NUM_THREADS];
    threadData trdData[NUM_THREADS];
    void *dd;
    double rnd;

    for (i=0; i<m*n; i++)
    {
        for (j=0; j<c-1; j++)
        {
            rnd = abs((double) rand() / (RAND_MAX) - 1.0);
            if (rnd > normThreshold[i])
            {
                rndTemp[i][j] = 1.0;
                len++;
                selF[i][len-1] = j;
            }
		}

        rndTemp[i][c] = len;
        len = 0;
    }

    for (i=0;i<NUM_THREADS;i++)
    {
        trdData[i].tID = i;
        trdData[i].r = r;
        trdData[i].c = c;
        trdData[i].nCls = nCls;
        trdData[i].cls = &cls;
        trdData[i].v = &v;
        trdData[i].data = &data;
    }

    i=0;
    while ((i+NUM_THREADS)<m*n)
    {
        cout << setw(5) << setprecision(5) << "  Individual = " << i+1 << " of " << m*n << "                    "<< '\r';
        cout.flush();
        for (j=0;j<NUM_THREADS;j++)
        {
            trdData[j].maxF = &selF[i+j];
            trdData[j].lF = rndTemp[i+j][c];
            pthread_create(&threads[j], NULL, ddWrapper, (void *)&trdData[j]);
        }

        for (j=0;j<NUM_THREADS;j++)
        {
            pthread_join(threads[j], &dd);
            rndTemp[i+j][c-1] = *(double *)dd;
        }

        i+=NUM_THREADS;
    }

    for (i;i<m*n;i++)
    {
        cout << setw(5) << setprecision(5) << "  Individual = " << i+1 << " of " << m*n << "                    "<< '\r';
        cout.flush();
        trdData[i%NUM_THREADS].maxF = &selF[i];
        trdData[i%NUM_THREADS].lF = rndTemp[i][c];
        pthread_create(&threads[i%NUM_THREADS], NULL, ddWrapper, (void *)&trdData[i%NUM_THREADS]);
        pthread_join(threads[i%NUM_THREADS], &dd);
        rndTemp[i][c-1] = *(double *)dd;
    }

    return rndTemp;
}

vector< vector<double> > rankFrogs(vector< vector<double> >& F, int m, int n)
{
    sort(F.begin(), F.end(), compDDxRed);
    return F;
}

double meanDD(vector< vector<double> >& F, int idx)
{
	double out = 0;
    for(int i = 0; i < F.size(); i++) {
		out += F[i][idx];
	}
    return (out/F.size());
}

void *ddWrapper(void *threadArg)
{
    struct threadData *args = (struct threadData *)threadArg;
    int r = args->r;
    int c = args->c;
    int nCls = args->nCls;
    vector<int> cls = *args->cls;
    vector<double> v = *args->v;
    vector<vector<double> > data = *args->data;
    vector<int> maxF = *args->maxF;
    int lF = args->lF;
    double dd;
    double *DD = &dd;

    dd = dependency(r, c, nCls, cls, v, data, maxF, lF);

    return (void *)DD;
}

void *evoWrapper(void *threadArg)
{
    struct threadEvo *args = (struct threadEvo *)threadArg;
    int r = args->r;
    int c = args->c;
    int q = args->q;
    int m = args->m;
    int im = args->im;
    int n = args->n;
    int Smax = args->Smax;
    int nF = args->nF;
    int N = args->N;
    int nCls = args->nCls;
    int stalled = args->stalled;
    vector<int> cls = *args->cls;
    vector<double> v = *args->v;
    vector<vector<double> > data = *args->data;
    vector<vector<double> > Y = *args->Y;
    vector<double> Pj = *args->Pj;

    int i = 0, j = 0, _nCls, traped = 0, nSel, nChanges, nOnes, nChanged = 0, _nOnes = 0, _n;
    double rnd, simDep, depOld, depNew;
    bool found = false;
    bool change = true;
    vector<int> selF(1);
    vector<int> _selF(nF);
    vector<double> tmp1(2);
    vector< vector<double> > pData(nF, tmp1);
    vector< vector<double> > _dataset;
    vector<double> _v(2);
    vector<int> _cls(2);
    vector<double> Pw(c+1);
    vector<double> Pb(c+1);
    vector<double> PwTemp(c+1);
    vector< vector<double> > yTemp(q, PwTemp);
    vector<int> cPlaces(2*c-1);
    vector<double> rndTmp(nF);
    vector<double> rd(q);
    selF[0] = 0;

    clsData *_spec = new clsData;

   _n = 0;
    nSel = 0;
    while (nSel < q)
    {
        for (int k=(im*n); k<((im+1)*n); k++)
        {
            rnd = abs((double) rand() / (RAND_MAX) - 1.0);
            _n = rand() % n;
                if ((rnd < Pj[_n]) && (nSel < q))
                {
                    rd[nSel] = k;
                    yTemp[nSel] = Y[k];
                    nSel++;
                }

                if (nSel == q) {break;}
        }
    }

    for (int iN=0; iN<N; iN++)
    {
        found = false;
        change = true;
        nChanged = 0;

        sort(yTemp.begin(), yTemp.end(), compDDxRed);

        Pw = yTemp[q-1];
        Pb = yTemp[0];
        PwTemp = Pw;

        for (int i=0; i<(c-1); i++)
        {
            pData[i][0] = Pb[i];
            pData[i][1] = PwTemp[i];
        }

        while(!found)
        {
            while (change)
            {
                for (int i=0; i<(c-1); i++)
                {
                    pData[i][1] = PwTemp[i];
                }
                _spec->initialize(pData, c-1, 2);
                _dataset = _spec->getData();
                _v = _spec->getVari();
                _cls = _spec->getCls();
                _nCls = _spec->getnCls();
                change = false;
            }

            cPlaces = vXor(Pb, PwTemp);

            if (cPlaces[c-1] != 0)
                simDep = nF - dependency(c-1, 2, _nCls, _cls, _v, _dataset, selF, 1);

            else
            {
                simDep = 0;
                traped ++;

                if (traped >= stalled)
                {
                    break;
                }
            }

            rnd = abs((double) rand() / (RAND_MAX) - 1.0);
            nChanges = min(floor(rnd*simDep), Smax*1.0);

            if (nChanges > 0)
            {
                nOnes = cPlaces[c-1];

                for (int i=0; i<nOnes; i++)
                    rndTmp[i] = abs((double) rand() / (RAND_MAX) - 1.0);


                for (int i=0; i<nOnes; i++)
                {
                    rnd = abs((double) rand() / (RAND_MAX) - 1.0);

                    if ((rnd < rndTmp[i]) && (nChanged <= nChanges))
                    {
                        PwTemp[cPlaces[i+c+1]] = Pb[cPlaces[i+c+1]];
                        nChanged++;
                        change = true;
                    }
                }

                j = 0;
                _nOnes = 0;

                for (int i=0; i<(c-1); i++)
                {
                    if (PwTemp[i] > 0)
                    {
                        _selF[j] = i;
                        j++;
                        _nOnes++;
                    }
                }

                PwTemp[c] = _nOnes;

                if (_nOnes != 0)
                {
                    depOld = PwTemp[c-1];
                    depNew = dependency(r, c, nCls, cls, v, data, _selF, _nOnes);
                }

                else
                {
                    traped ++;

                    if (traped >= stalled)
                    {
                        break;
                    }
                }

                if (depNew > depOld)
                {
                    PwTemp[c-1] = depNew;
                    traped = 0;
                    found = true;
                }

                else if (depNew == depOld)
                {
                    if (Pw[c] > PwTemp[c])
                    {
                        PwTemp[c-1] = depNew;
                        traped = 0;
                        found = true;
                    }

                    if (Pw[c] == PwTemp[c])
                    {
                        traped ++;
                    }
                }

                else if ((depNew < depOld) || ((depNew == depOld) && Pw[c] < PwTemp[c]))
                {
                    j = 0;
                    _nOnes = 0;

                    for (int i=0; i<(c-1); i++)
                    {
                        PwTemp[i] = fastrand(0x0001) * fastrand(0x0001);///rand() % 2;
                        _selF[i] = 0;

                        if (PwTemp[i] > 0)
                        {
                            _selF[j] = i;
                            j++;
                            _nOnes++;
                        }

                    }

                    if (_nOnes != 0)
                    {
                        PwTemp[c] = _nOnes;
                        depNew = dependency(r, c, nCls, cls, v, data, _selF, _nOnes);
                    }

                    else
                    {
                        traped ++;

                        if (traped >= stalled)
                        {
                            break;
                        }
                    }

                    if (depNew > depOld)
                    {
                        PwTemp[c-1] = depNew;
                        found = true;
                        traped = 0;
                    }

                    if (depNew == depOld)
                    {
                        if (Pw[c] > PwTemp[c])
                        {
                            PwTemp[c-1] = depNew;
                            found = true;
                            traped = 0;
                        }

                        if (Pw[c] == PwTemp[c])
                        {
                            PwTemp[c-1] = depNew;
                            traped ++;
                        }
                    }
                    change = true;
                }
            }
            else
            {
                traped ++;
            }

            if (traped >= stalled)
            {
                traped = 0;
                break;
            }
        }
        if (found) {Y[rd[q-1]] = PwTemp;}
    }
    traped = 0;

    delete(_spec);
}

double dependency(int r, int c, int nCls, vector<int>& cls, vector<double>& v, vector<vector<double> >& data, vector<int>& maxF, int lF)
{
    int mF = 1, h = 0, k = 0, s = 0, i = 0, j = 0, nF = 0;

    int lMoD = nCls;
    double fterm1 = 0.0, fterm2 = 0.0, moRp = 0.0, out = 0.0;
    vector<double> tmp(c), moRa(lF), supMat(r);
    vector<vector<double> > moX(r, vector<double>(2 * lMoD));

    if (lF <1)
        return 0.0;

    for (nCls=0;nCls<lMoD;++nCls)
    {
        for (s;s<cls[nCls];++s) //s
        {
            moX[s][nCls] = 1.0;
        }

        s = cls[nCls];
    }

    for (int x1=0;x1<r;++x1)
    {
        for (int x2=0;x2<r;++x2)
        {
            for (h=0;h<lF;++h)
            {
                mF = maxF[h];
                fterm1 = (data[x2][mF] - (data[x1][mF] - v[mF])) / (data[x1][mF] - (data[x1][mF] - v[mF]));
                fterm2 = ((data[x1][mF] + v[mF]) - data[x2][mF]) / (data[x1][mF] + v[mF] - data[x1][mF]);
                moRa[h] = max(min(fterm1, fterm2), 0.0);
            }

            if (lF > 1)
            {
                moRp = max(moRa[0] + moRa[1] - 1.0, 0.0);
                for(int nMoRmF=2;nMoRmF<lF;++nMoRmF)
                {
                    moRp = max(moRp + moRa[nMoRmF] - 1.0, 0.0);
                }
            }
            else
            {
                moRp = moRa[0];
            }

            for (int j=lMoD;j<(2*lMoD);++j)
            {
                moX[x2][j] = min(1- moRp + moX[x2][j-lMoD], 1.0);
            }
        }


        k = 0;

        for (j=lMoD;j<(2*lMoD);++j)
        {
            tmp[k] = moX[0][j];

            for (i=1;i<r;++i)
            {
                tmp[k] = min(moX[i][j], tmp[k]);
            }

            k++;
        }

        supMat[x1] = tmp[0];

        for (k=1;k<lMoD;++k)
        {
            supMat[x1] = max(tmp[k], supMat[x1]);
        }
    }

    for (k=0;k<r;++k)
    {
        out += supMat[k];
        supMat[k] = 0.0;
    }

    if (c > 2)
        out = out / r;

    return out;
}

double dependency(int r, int c, int nCls, vector<int>& cls, vector<double>& v, double *data, vector<int>& maxF, int lF)
{
    int mF = 1, h = 0, k = 0, s = 0, i = 0, j = 0, nF = 0;

    int lMoD = nCls;
    double fterm1 = 0.0, fterm2 = 0.0, moRp = 0.0, out = 0.0;
    double tmp[c], moRa[lF], supMat[r];
    double moX[r][2 * lMoD];

    for (h=0;h<lF;++h)
    {
        nF += maxF[h];
    }
    if (nF == 0)
        return 0.0;


    for (int a=0;a<r;++a)
    {
        for (int b=0;b<(2*lMoD);++b)
        {
            moX[a][b] = 0.0;
        }
    }

    for (nCls=0;nCls<lMoD;++nCls)
    {
        for (s=cls[nCls];s<cls[nCls];++s)
        {
            moX[s][nCls] = 1.0;
        }

        s = cls[nCls];
    }

    for (int x1=0;x1<r;++x1)
    {
        for (int x2=0;x2<r;++x2)
        {
            for (h=0;h<lF;++h)
            {
                mF = maxF[h];
                fterm1 = (data[x2*c+mF] - (data[x1*c+mF] - v[mF])) / (data[x1*c+mF] - (data[x1*c+mF] - v[mF]));
                fterm2 = ((data[x1*c+mF] + v[mF]) - data[x2*c+mF]) / (data[x1*c+mF] + v[mF] - data[x1*c+mF]);
                moRa[h] = max(min(fterm1, fterm2), 0.0);
            }

            if (lF > 1)
            {
                moRp = max(moRa[0] + moRa[1] - 1.0, 0.0);
                for(int nMoRmF=2;nMoRmF<lF;++nMoRmF)
                {
                    moRp = max(moRp + moRa[nMoRmF] - 1.0, 0.0);
                }
            }
            else
            {
                moRp = moRa[0];
            }

            for (int j=lMoD;j<(2*lMoD);++j)
            {
                moX[x2][j] = min(1- moRp + moX[x2][j-lMoD], 1.0);
            }
        }


        k = 0;

        for (j=lMoD;j<(2*lMoD);++j)
        {
            tmp[k] = moX[0][j];

            for (i=1;i<r;++i)
            {
                tmp[k] = min(moX[i][j], tmp[k]);
            }

            k++;
        }

        supMat[x1] = tmp[0];

        for (k=1;k<lMoD;++k)
        {
            supMat[x1] = max(tmp[k], supMat[x1]);
        }
    }

    for (k=0;k<r;++k)
    {
        out += supMat[k];
        supMat[k] = 0.0;
    }

    if (c > 2)
        out = out / r;

    return out;
}

int linear_search(int arr[], int n, int val)
{
    for(int i = 0; i < n; i++)
        if (arr[i] == val) return i;
    return -1;
}

float sqrt7(float x)
 {
   unsigned int i = *(unsigned int*) &x;
   // adjust bias
   i  += 127 << 23;
   // approximation of square root
   i >>= 1;
   return *(float*) &i;
 }
