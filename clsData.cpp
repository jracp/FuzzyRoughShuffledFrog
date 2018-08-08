#include "clsData.h"

clsData::clsData ()
{

}

clsData::~clsData ()
{
    this->dataset.clear();
    this->v.clear();
    this->clsC.clear();
}

bool comp(const vector<double>& p1, const vector<double>& p2)
{
    int sortCol = p1.size();
    sortCol--;
    return p1[sortCol] < p2[sortCol];
}

void handle_error(const char* msg)
{
    perror(msg);
    exit(255);
}

//http://stackoverflow.com/questions/17925051/fast-textfile-reading-in-c
const char* map_file(const char* fname, size_t& length)
{
    int fd = open(fname, O_RDONLY);
    if (fd == -1)
        handle_error("open");

    // obtain file size
    struct stat sb;
    if (fstat(fd, &sb) == -1)
        handle_error("fstat");

    length = sb.st_size;

    const char* addr = static_cast<const char*>(mmap(NULL, length, PROT_READ, MAP_PRIVATE, fd, 0u));
    if (addr == MAP_FAILED)
        handle_error("mmap");

    // TODO close fd at some point in time, call munmap(...)
    return addr;
}

//void clsData::initialize (ifstream& dataset)
void clsData::initialize (char* filename)
{
    int row = 0, _col = 0, i = 0, j = 0, k = 0, pStart = 0, pEnd = 0, pos = 0, nCls = 1;
    string value, elmnt, temp;
    double clsTmp = 0.0;
    ifstream dataset (filename);
    size_t length;
    auto f = map_file(filename, length);
    auto l = f + length;
    auto rawData = f;
    auto tmpF = f;

    //Calcultating number of rows and columns
    while (f && f!=l)
        if (f = static_cast<const char*>(memchr(f, '\r', l-f)))
            row++, f++;
    if (row == 0)
    {
        while (tmpF && tmpF!=l)
            if (tmpF = static_cast<const char*>(memchr(tmpF, '\n', l-tmpF)))
                row++, tmpF++;
    }

    while ((rawData[k] != '\r') && (rawData[k] != '\n'))
    {
        k++;
    }

    while(i < k)
    {
        if (rawData[i] == ',')
            _col++;
        i++;
    }
    _col++;

    vector<double> tempData(_col);
    vector< vector<double> > data(row, tempData);

    k = 0;

    for (i=0;i<row;++i)
    {
        for (j=0;j<_col;++j)
        {
            while ((rawData[k] != ',') && (rawData[k] != '\r') && (rawData[k] != '\n'))
            {
                value += rawData[k];
                k++;
            }

            if (value != "")
            {
                data[i][j] = atof(value.c_str());
                value = "";
            }
            k++;
        }
    }

    //Finding number of classes in dataset
    j = 0;
    vector<int> tmpCls(row);

    for (i=1;i<row;++i)
    {
        if (clsTmp != data[i][_col-1])
        {
            j++;
            tmpCls[j] = i;
            clsTmp = data[i][_col-1];
            nCls++;
        }
    }

    vector<int> _cls(nCls);

    for (j=0;j<nCls-1;++j)
    {
        _cls[j] = tmpCls[j+1];
    }

    _cls[j] = row;

    //Calculating variance of feature
    double sum = 0.0;
    double mean = 0.0;
    vector<double> vrnc(_col-1);

    for (j=0;j<_col-1;++j)
    {
        sum = 0.0;
        for (i=0;i<row;++i)
        {
            sum += data[i][j];
        }
        mean = sum / row;

        vrnc[j] = 0.0;

        for (i=0;i<row;i++)
        {
            vrnc[j] = vrnc[j] + pow((data[i][j] - mean), 2.0);
        }

        vrnc[j] = pow(vrnc[j]/(row - 1), 0.5);
    }

    this->r = row;
    this->c = _col;
    this->dataset = data;
    this->v = vrnc;
    this->clsC = _cls;
    this->nClsC = nCls;
}

void clsData::initialize (vector< vector<double> >& data, int row, int col)
{
    int i = 0, j = 0;
    string value, elmnt, temp;

    //Sort dataset based on class value
    sort(data.begin(), data.end(), comp);

    //Finding number of classes in dataset
    vector<int> cls(col);
    double clsTmp;
    int nCls = 1;
    j = 0;
    clsTmp = data[j][col-1];
    cls[j] = 0;

    for (i=1;i<row;++i)
    {
        if (clsTmp != data[i][col-1])
        {
            j++;
            cls[j] = i;
            clsTmp = data[i][col-1];
            nCls++;
        }
    }

    for (j=0;j<nCls-1;++j)
    {
        cls[j] = cls[j+1];
    }
    cls[nCls-1] = row;

    //Calculating variance of feature
    double sum = 0.0;
    double mean = 0.0;
    j = 0;
    vector<double> v(2);

    sum = 0.0;
    for (i=0;i<row;++i)
    {
        sum += data[i][j];
    }
    mean = sum / row;

    for (i=0;i<row;i++)
    {
        v[j] = v[j] + pow((data[i][j] - mean), 2.0);
    }
    v[j] = pow(v[j]/(row - 1), 0.5);
    //}

    this->r = row;
    this->c = col;
    this->dataset = data;
    this->v = v;
    this->clsC = cls;
    this->nClsC = nCls;
}
