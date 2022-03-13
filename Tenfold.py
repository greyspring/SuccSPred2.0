import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import KFold
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
from imblearn.over_sampling import SMOTE


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

    pos = filegroup["pos"]
    neg = filegroup["neg"]

    file_method = {}
    filepath = []
    pos_method = neg_method = ''
    for methodname in method:
        for i in pos:
            if methodname in i:
                pos_method = i
                break
        for j in neg:
            if methodname in j:
                neg_method = j
                break

        filepath = [neg_method, pos_method]

        file_method[methodname] = dataprocessing(filepath)

    print(file_method)
    return file_method


def dataprocessing(filepath):
    print("Loading feature files")
    print(filepath)

    dataset1 = pd.read_csv(filepath[0], header=None, low_memory=False)
    dataset2 = pd.read_csv(filepath[1], header=None, low_memory=False)

    print(dataset1.shape, dataset2.shape)

    print("Feature processing")

    dataset1 = dataset1.apply(pd.to_numeric, errors="ignore")
    dataset2 = dataset2.apply(pd.to_numeric, errors="ignore")

    dataset1.dropna(inplace=True)
    dataset2.dropna(inplace=True)

    datas = pd.concat([dataset1, dataset2], axis=0)

    smo = SMOTE(random_state=42)

    negtags = [0] * dataset1.shape[0]
    postags = [1] * dataset2.shape[0]
    tags = negtags + postags

    datas, tags = smo.fit_resample(datas, tags)
    data = [datas, tags]

    print(pd.DataFrame(data[0]).shape)

    return data

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

def main():
    # tmpdir = "method-feature1/"
    # tmpdir2 = "method-feature-extend/"
    #
    # pos = glob.glob(tmpdir + 'suc_pos*')
    # neg = glob.glob(tmpdir + 'suc_neg*')
    #
    # pos2 = glob.glob(tmpdir2 + 'suc_pos.fastafeature*')
    # neg2 = glob.glob(tmpdir2 + 'suc_neg.fastafeature*')
    #
    # pos.extend(pos2)
    # neg.extend(neg2)
    # filegroup = {}
    # filegroup['pos'] = pos
    # filegroup['neg'] = neg
    # datadics = datadic(filegroup)
    # np.save("tenfold3/SMOTE_79.npy", datadics, allow_pickle=True)

    # datadics = np.load("tenfold3/SMOTE_79.npy", allow_pickle=True).item()

    # kf = KFold(n_splits=10, shuffle=True)
    # train_all = datadics['-AC-PSSM.csv'][0]
    # label_all = datadics['-AC-PSSM.csv'][1]
    # data_new = {}
    # j = 0

    # for train_index, test_index in kf.split(train_all):
    #     for i in datadics:
    #         data = datadics[i]
    #         X_train, X_test = np.array(data[0])[train_index], np.array(data[0])[test_index]
    #         y_train, y_test = np.array(data[1])[train_index], np.array(data[1])[test_index]
    #         a = [X_train, y_train, X_test, y_test]
    #         data_new[i] = a
    #     filename = "tenfold3//data//data_new_" + str(j) + ".npy"
    #     np.save(filename, data_new, allow_pickle=True)
    #     processed = process(data_new)
    #
    #     jojo = 0
    #     tr = pd.DataFrame(processed['-ANN4D-18.csv'][0])
    #     for i in processed:
    #         jojo = jojo + 1
    #         tr = pd.concat([tr, pd.DataFrame(processed[i][0])], axis=1)
    #         if jojo == 78:
    #             break
    #     print(tr.shape)
    #     tg = pd.DataFrame(processed["DR.csv"][1])
    #     print(tg.shape)
    #
    #     jojo = 0
    #     ta = pd.DataFrame(processed['-ANN4D-18.csv'][2])
    #     for i in processed:
    #         jojo = jojo + 1
    #         ta = pd.concat([ta, pd.DataFrame(processed[i][2])], axis=1)
    #         if jojo == 78:
    #             break
    #     print(ta.shape)
    #
    #     tsg = pd.DataFrame(processed["DR.csv"][3])
    #     print(tsg.shape)
    #     data3 = [tr.values, tg.values.ravel(), ta.values, tsg.values.ravel()]
    #
    #     filename = "tenfold3//beforeselect//data_" + str(j) + ".npy"
    #     np.save(filename, data3, allow_pickle=True)
    #
    #     a, b = data3[0].shape
    #     list = []
    #     mine = MINE(alpha=0.6, c=15)
    #     for i in range(b):
    #         mine.compute_score(data3[0][:, i], data3[1])
    #         list.append(mine.mic())
    #     mic = np.array(list)
    #     list = []
    #
    #     for i in range(len(mic)):
    #         if mic[i] >= 0.2:
    #             list.append(i)
    #
    #     filename = "tenfold3//mic//mic_" + str(j) + ".npy"
    #     np.save(filename, mic, allow_pickle=True)
    #
    #     data2 = [data3[0][:, list], data3[1], data3[2][:, list], data3[3]]
    #
    #     model_SVC = SVC(probability=True)
    #     model_SVC.fit(data2[0], data2[1])
    #     indemetric = svmOpt(data2[2], data2[3], model_SVC)
    #     random_inde = randomforest(data2[0], data2[1], data2[2], data2[3])
    #     bayes_inde = bayes(data2[0], data2[1], data2[2], data2[3])
    #     logisticregression_inde = logisticregression(data2[0], data2[1], data2[2], data2[3])
    #     adaboost_inde = adaboost(data2[0], data2[1], data2[2], data2[3])
    #     decisiontree_inde = decisiontree(data2[0], data2[1], data2[2], data2[3])
    #     xg = XGBoost(data2[0], data2[1], data2[2], data2[3])
    #     li = LightGBM(data2[0], data2[1], data2[2], data2[3])
    #
    #     metric = pd.DataFrame(indemetric)
    #     random_inde = pd.DataFrame(random_inde)
    #     bayes_inde = pd.DataFrame(bayes_inde)
    #     logisticregression_inde = pd.DataFrame(logisticregression_inde)
    #     adaboost_inde = pd.DataFrame(adaboost_inde)
    #     decisiontree_inde = pd.DataFrame(decisiontree_inde)
    #     xg = pd.DataFrame(xg)
    #     li = pd.DataFrame(li)
    #
    #     col = ['acc', 'auc', 'sen', 'spec', 'mcc', 'f1_score', 'sp', 'sn']
    #     piece = metric.loc[0, col]
    #     random_inde_piece = random_inde.loc[0, col]
    #     bayes_inde_piece = bayes_inde.loc[0, col]
    #     logisticregression_inde_piece = logisticregression_inde.loc[0, col]
    #     adaboost_inde_piece = adaboost_inde.loc[0, col]
    #     decisiontree_inde_piece = decisiontree_inde.loc[0, col]
    #     piece1 = xg.loc[0, col]
    #     piece2 = li.loc[0, col]
    #
    #     piece.name = 'Svm'
    #     random_inde_piece.name = 'Randomforest'
    #     bayes_inde_piece.name = 'Bayes'
    #     logisticregression_inde_piece.name = 'logisticregression'
    #     adaboost_inde_piece.name = 'Adaboost'
    #     decisiontree_inde_piece.name = 'Decision_tree'
    #     piece1.name = 'XGBoost'
    #     piece2.name = 'lightGBM'
    #
    #     outCome = pd.concat(
    #         [piece, random_inde_piece, bayes_inde_piece, logisticregression_inde_piece, adaboost_inde_piece,
    #          decisiontree_inde_piece, piece1, piece2], axis=1)
    #     filename = "ten_" + str(j) + '.csv'
    #     filepath = "tenfold3//result//" + filename
    #     outCome.to_csv(filepath)
    #     print(outCome)
    #     j = j + 1

    for j in range(10):

        filename = "tenfold3//beforeselect//data_" + str(j) + ".npy"
        data3 = np.load(filename, allow_pickle=True)

        filename = "tenfold3//mic//mic_" + str(j) + ".npy"
        mic = np.load(filename, allow_pickle=True)
        list = []

        for i in range(len(mic)):
            if mic[i] >= 0.2:
                list.append(i)
        print(len(list))

        data2 = [data3[0][:, list], data3[1], data3[2][:, list], data3[3]]

        model_SVC = SVC(probability=True)
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

        outCome = pd.concat(
            [piece, random_inde_piece, bayes_inde_piece, logisticregression_inde_piece, adaboost_inde_piece,
             decisiontree_inde_piece, piece1, piece2], axis=1)
        print(outCome)

if __name__ == '__main__':
    main()


