import pandas as pd
import numpy as np
import glob
from sklearn.metrics import matthews_corrcoef
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from minepy import MINE
from machine_learning_models import  logisticregression
from machine_learning_models import  randomforest
from machine_learning_models import  decisiontree
from machine_learning_models import  LightGBM
from machine_learning_models import  adaboost
from machine_learning_models import  XGBoost
from machine_learning_models import  bayes
from machine_learning_models import  Svm,svmOpt
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler


def process(datadic):
    index = []
    for i in datadic:
        data = datadic[i]
        index.append(i)
        print(i)
        normalizer = preprocessing.Normalizer(copy=True, norm='l2').fit(data[0])
        data[0] = normalizer.transform(data[0])
        data[2] = normalizer.transform(data[2])
        lda = LinearDiscriminantAnalysis()
        lda.fit(data[0],data[1])
        data[0] = lda.transform(data[0])
        data[2] = lda.transform(data[2])
        datadic[i] = data

    return datadic

def find_best_SVM(c,gamma,traindata,trainlabel,testdata,testlabel):
    model = SVC(C=c, gamma=gamma, probability=True)
    model.fit(traindata,trainlabel)
    prediction=model.predict(testdata)
    acc=matthews_corrcoef(testlabel, prediction)
    print(acc)
    return acc


def SVMpara(data):
    best_par_acc = 0
    c = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    gamma = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]

    for C in c:
        for Gamma in gamma:
            accuracy = find_best_SVM(C, Gamma, data[0], data[1], data[2], data[3])
            print(C, " ", Gamma, " ", accuracy)
            if (accuracy > best_par_acc):
                best_par_acc = accuracy
                best_C = C
                best_gamma = Gamma

    print("the highest acc is ", best_par_acc)
    print("the best C is ", best_C)
    print("the best gamma is ", best_gamma)
    return best_C, best_gamma

def dataprocessing(filepath):
    print ("Loading feature files")

    dataset1 = pd.read_csv(filepath[0],header=None,low_memory=False)
    dataset2 = pd.read_csv(filepath[1],header=None,low_memory=False)
    dataset3 = pd.read_csv(filepath[2],header=None,low_memory=False)
    dataset4 = pd.read_csv(filepath[3],header=None,low_memory=False)

    print ("Feature processing")



    dataset1 = dataset1.apply(pd.to_numeric, errors="ignore")
    dataset2 = dataset2.apply(pd.to_numeric, errors="ignore")
    dataset3 = dataset3.apply(pd.to_numeric, errors="ignore")
    dataset4 = dataset4.apply(pd.to_numeric, errors="ignore")

    dataset1.dropna(inplace = True)
    dataset2.dropna(inplace = True)
    dataset3.dropna(inplace = True)
    dataset4.dropna(inplace = True)


    traindata = pd.concat([dataset2, dataset4],axis=0)
    testdata = pd.concat([dataset1, dataset3],axis=0)


    smo = RandomUnderSampler(random_state=42)
    smo2= RandomUnderSampler(random_state=42)


    negtraintags = [0]*dataset2.shape[0]
    postraintags= [1]*dataset4.shape[0]
    traintags = negtraintags+postraintags

    negtesttags = [0]*dataset1.shape[0]
    postesttags= [1]*dataset3.shape[0]
    testtags = negtesttags+postesttags

    traindata,traintags = smo.fit_resample(traindata, traintags)
    testdata, testtags = smo2.fit_resample(testdata, testtags)

    data = [traindata,traintags,testdata,testtags]


    print("after the  balance")
    print(pd.DataFrame(data[0]).shape)
    print(pd.DataFrame(data[1]).shape)
    print(pd.DataFrame(data[2]).shape)
    print(pd.DataFrame(data[3]).shape)
    return data


def datadic(filegroup):
    method = [
        "feature-DT.csv", "-PDT-Profile.csv", "-Top-n-gram.csv", "-PSSM-RT.csv", "-PSSM-DT.csv", "-CC-PSSM.csv",
        "-AC-PSSM.csv", "ACC-PSSM.csv", "kmer", "feature-AC.csv", "feature-ACC.csv", "feature-CC.csv",
        "DP.csv", "DR.csv", "PC-PseAAC.csv", "PC-PseAAC-General.csv", "PDT.csv", "SC-PseAAC.csv",
        "SC-PseAAC-General.csv",

        "-One_hot-10.csv", "-One_hot-12.csv", "-One_hot-14.csv", "-One_hot-16.csv", "-One_hot-18.csv",
        "-One_hot_6_bit-10.csv", "-One_hot_6_bit-12.csv", "-One_hot_6_bit-14.csv", "-One_hot_6_bit-16.csv",
        "-One_hot_6_bit-18.csv",

        "-Binary_5_bit-10", "-Binary_5_bit-12", "-Binary_5_bit-14", "-Binary_5_bit-16", "-Binary_5_bit-18",
        "-Hydrophobicity_matrix-10.csv", "-Hydrophobicity_matrix-12.csv", "-Hydrophobicity_matrix-14.csv",
        "-Hydrophobicity_matrix-16.csv", "-Hydrophobicity_matrix-18.csv",
        "-Acthely_factors-10.csv", "-Acthely_factors-12.csv", "-Acthely_factors-14.csv", "-Acthely_factors-16.csv",
        "-Acthely_factors-18.csv",
        "-Meiler_parameters-10.csv", "-Meiler_parameters-12.csv", "-Meiler_parameters-14.csv",
        "-Meiler_parameters-16.csv", "-Meiler_parameters-18.csv",
        "-PAM250-10.csv", "-PAM250-12.csv", "-PAM250-14.csv", "-PAM250-16.csv", "-PAM250-18.csv",
        "-BLOSUM62-10.csv", "-BLOSUM62-12.csv", "-BLOSUM62-14.csv", "-BLOSUM62-16.csv", "-BLOSUM62-18.csv",
        "-Miyazawa_energies-10.csv", "-Miyazawa_energies-12.csv", "-Miyazawa_energies-14.csv",
        "-Miyazawa_energies-16.csv", "-Miyazawa_energies-18.csv",
        "-Micheletti_potentials-10.csv", "-Micheletti_potentials-12.csv", "-Micheletti_potentials-14.csv",
        "-Micheletti_potentials-16.csv", "-Micheletti_potentials-18.csv",
        "-AESNN3-10.csv", "-AESNN3-12.csv", "-AESNN3-14.csv", "-AESNN3-16.csv", "-AESNN3-18.csv",
        "-ANN4D-10.csv", "-ANN4D-12.csv", "-ANN4D-14.csv", "-ANN4D-16.csv", "-ANN4D-18.csv"]

    postrain = filegroup["postrain"]
    negtrain = filegroup["negtrain"]
    postest = filegroup["postest"]
    negtest = filegroup["negtest"]

    file_method = {}
    filepath = []
    postrain_method = negtrain_method = postest_method = negtest_method = ''
    for methodname in method:
        for i in postrain:
            if methodname in i:
                postrain_method = i
                break

        for j in negtrain:
            if methodname in j:
                negtrain_method = j
                break

        for k in postest:
            if methodname in k:
                postest_method = k
                break

        for l in negtest:
            if methodname in l:
                negtest_method = l
                break

        filepath = [negtest_method, negtrain_method, postest_method, postrain_method]

        file_method[methodname] = dataprocessing(filepath)

    print('the type is :', type(file_method))
    return file_method

def main():
    # tmpdir = "method-feature_extension"
    # postrain = glob.glob(tmpdir + '/train_pos*')
    # negtrain = glob.glob(tmpdir + '/train_neg*')
    # postest = glob.glob(tmpdir + '/test_pos*')
    # negtest = glob.glob(tmpdir + '/test_neg*')
    #
    # filegroup = {}
    # filegroup['postrain'] = postrain
    # filegroup['negtrain'] = negtrain
    # filegroup['postest'] = postest
    # filegroup['negtest'] = negtest
    # datadics = datadic(filegroup)
    # np.save("indenp/Independent_JMC.npy", datadics, allow_pickle=True)
    # datadics = np.load("indenp/Independent_JMC.npy", allow_pickle=True).item()
    # processed = process(datadics)
    #
    # jojo = 0
    # tr = pd.DataFrame(processed['-ANN4D-18.csv'][0])
    # for i in processed:
    #     jojo = jojo + 1
    #     tr = pd.concat([tr, pd.DataFrame(processed[i][0])], axis=1)
    #     if jojo == 78:
    #         break
    # print(tr.shape)
    #
    # tg = pd.DataFrame(processed["DR.csv"][1])
    # print(tg.shape)
    #
    # jojo = 0
    # ta = pd.DataFrame(processed['-ANN4D-18.csv'][2])
    # for i in processed:
    #     jojo = jojo + 1
    #     ta = pd.concat([ta, pd.DataFrame(processed[i][2])], axis=1)
    #     if jojo == 78:
    #         break
    # print(ta.shape)
    #
    # tsg = pd.DataFrame(processed["DR.csv"][3])
    # print(tsg.shape)
    #
    # data3 = [tr.values, tg.values.ravel(), ta.values, tsg.values.ravel()]
    #
    # a, b = data3[0].shape
    # list = []
    # mine = MINE(alpha=0.6, c=15)
    # for i in range(b):
    #     mine.compute_score(data3[0][:, i], data3[1])
    #     list.append(mine.mic())
    # mic = np.array(list)
    # list = []
    #
    # for i in range(len(mic)):
    #     if mic[i] >= 0.2:
    #         list.append(i)
    #
    # data2 = [data3[0][:, list], data3[1], data3[2][:, list], data3[3]]
    # np.save("indenp/dataInden.npy",allow_pickle=True)

    data2 = np.load("indenp/dataInden.npy",allow_pickle=True)
    model_SVC = SVC(probability=True, C=1e-05, gamma=0.01)
    model_SVC.fit(data2[0], data2[1])
    indemetric = svmOpt(data2[2], data2[3], model_SVC)
    random_inde = randomforest(data2[0], data2[1], data2[2], data2[3])
    bayes_inde = bayes(data2[0], data2[1], data2[2], data2[3])
    logisticregression_inde = logisticregression(data2[0], data2[1], data2[2], data2[3])
    adaboost_inde = adaboost(data2[0], data2[1], data2[2], data2[3])
    decisiontree_inde = decisiontree(data2[0], data2[1], data2[2], data2[3])
    xg = XGBoost(data2[0], data2[1], data2[2], data2[3])
    li = LightGBM(data2[0], data2[1], data2[2], data2[3])

    metric = pd.DataFrame(indemetric)
    random_inde = pd.DataFrame(random_inde)
    bayes_inde = pd.DataFrame(bayes_inde)
    logisticregression_inde = pd.DataFrame(logisticregression_inde)
    adaboost_inde = pd.DataFrame(adaboost_inde)
    decisiontree_inde = pd.DataFrame(decisiontree_inde)
    xg = pd.DataFrame(xg)
    li = pd.DataFrame(li)

    col = ['acc', 'auc', 'sen', 'spec', 'mcc', 'f1_score', 'sp', 'sn']
    piece = metric.loc[0, col]
    random_inde_piece = random_inde.loc[0, col]
    bayes_inde_piece = bayes_inde.loc[0, col]
    logisticregression_inde_piece = logisticregression_inde.loc[0, col]
    adaboost_inde_piece = adaboost_inde.loc[0, col]
    decisiontree_inde_piece = decisiontree_inde.loc[0, col]
    piece1 = xg.loc[0, col]
    piece2 = li.loc[0, col]

    piece.name = 'Svm'
    random_inde_piece.name = 'Randomforest'
    bayes_inde_piece.name = 'Bayes'
    logisticregression_inde_piece.name = 'logisticregression'
    adaboost_inde_piece.name = 'Adaboost'
    decisiontree_inde_piece.name = 'Decision_tree'
    piece1.name = 'XGBoost'
    piece2.name = 'lightGBM'

    outCome = pd.concat([piece, random_inde_piece, bayes_inde_piece, logisticregression_inde_piece, adaboost_inde_piece,
                         decisiontree_inde_piece, piece1, piece2], axis=1)
    filename = "INDEPENDENT" + '.csv'
    filepath = "indenp//" + filename
    # outCome.to_csv(filepath)
    print(outCome)


if __name__ == '__main__':
    main()


