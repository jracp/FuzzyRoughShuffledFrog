#include "clsWeka.h"

clsWeka::clsWeka()
{
    //ctor
}

string* clsWeka::classify(vector<int> maxF, char* input, int col)
{
    string attSelIn, attSelOut, samSelIn, samSelOut, rmWEKA, selF = "", clWEKA, line, token = "%";
    string filename(input);
    int i, j, maxFSize, classSize, pos = 0, res;
    double meanAcc = 0.0, bestAcc = 0.0, stdAcc = 0.0;

    maxFSize = maxF.size();
    for (i=0; i < maxFSize; i++)
    {
        selF = selF + to_string(maxF[i]) + ",";
    }
    selF = selF + to_string(col);

    classifiers[0] = "weka.classifiers.rules.PART -M 2 -C 0.25 -Q 1";
    classifiers[1] = "weka.classifiers.rules.JRip -F 3 -N 2.0 -O 2 -S 1";
    classifiers[2] = "weka.classifiers.bayes.NaiveBayes";
    classifiers[3] = "weka.classifiers.bayes.BayesNet -D -Q weka.classifiers.bayes.net.search.local.K2";
    classifiers[4] = "weka.classifiers.trees.J48 -C 0.25 -M 2";
    classifiers[5] = "weka.classifiers.trees.BFTree -S 1 -M 2 -N 5 -C 1.0 -P POSTPRUNED";
    classifiers[6] = "weka.classifiers.trees.FT -I 15 -F 0 -M 15 -W 0.0";
    classifiers[7] = "weka.classifiers.trees.NBTree";
    classifiers[8] = "weka.classifiers.functions.RBFNetwork -B 2 -S 1 -R 1.0E-8 -M -1 -W 0.1";

    clsAcc[0] = " PART = ";
    clsAcc[1] = " JRip = ";
    clsAcc[2] = " NaiveBayes = ";
    clsAcc[3] = " BayesNet = ";
    clsAcc[4] = " J48 = ";
    clsAcc[5] = " BFTree = ";
    clsAcc[6] = " FT = ";
    clsAcc[7] = " NBTree = ";
    clsAcc[8] = " RBFNetwork = ";
    clsAcc[9] = " Mean = ";
    clsAcc[10] = " Best = ";

    attSelIn = filename.substr(0, filename.length()-6) + "W.csv"; //attibute selection input filename
    attSelOut = filename.substr(0, filename.length()-6)+ ".arff"; //attibute selection outou filename
    samSelIn = filename.substr(0, filename.length()-6)+ ".arff"; //sample selection input filename
    samSelOut = filename.substr(0, filename.length()-6)+ "S.arff"; //sample selection outou filename

    rmWEKA = "java -Xmx1024m -classpath weka.jar weka.filters.unsupervised.attribute.Remove -V -R"; //remove command
    rmWEKA = rmWEKA + " " + selF + " -i " + attSelIn + " -o " + attSelOut;
	res = system(rmWEKA.c_str());


    classSize = sizeof(classifiers)/sizeof(classifiers[0]);

    for(j=0; j<classSize; j++)
    {
        clWEKA = "java -Xmx1024m -classpath weka.jar";

        if (j!=3) //BayesNet
            clWEKA = clWEKA + " " + classifiers[j] + " -t " + samSelIn + " > ./classification/classifiers" + to_string(j) + ".txt";
        else
            clWEKA = clWEKA + " " + classifiers[j] + " -t " + samSelIn + " -- -P 1 -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5" + " > ./classification/classifiers" + to_string(j) + ".txt";

        res = system(clWEKA.c_str());

        ifstream classFiles("./classification/classifiers" + to_string(j) + ".txt");
        pos = 0;
        i = 0;
        line = "";
        while (getline(classFiles, line))
        {
            pos = line.find(token);
            if (pos > 0)
                i++;
            if (i == 5)
                    break;
        }

        clsAcc[j] = clsAcc[j] + line.substr(pos-8, pos-2);
	acc[j] = atof(line.substr(pos-8, pos-2).c_str());
        meanAcc += atof(line.substr(pos-8, pos-2).c_str());

        if (acc[j] > bestAcc) {bestAcc = acc[j];}

    }

    meanAcc /= classSize;

    for(j=0; j<classSize; j++)
    {
	stdAcc = stdAcc + pow((acc[j] - meanAcc), 2.0);
    }

    stdAcc /= classSize;
    stdAcc = pow(stdAcc, 0.5);

    clsAcc[j] = clsAcc[j] + to_string(meanAcc) + " ± " + to_string(stdAcc);
    clsAcc[j+1] = clsAcc[j+1] + to_string(bestAcc) + "  %";

    return clsAcc;
}
