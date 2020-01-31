import numpy as np
import csv
import pandas as pd

from scipy import sparse

import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

types = {
    "id" : "int",
    "amount_tsh" : "float",
    "date_recorded" : "datetime64",
    "funder" : "str",
    "gps_height" : "float",
    "installer" : "str",
    "longitude" : "float",
    "latitude" : "float",
    "wpt_name" : "str",
    "num_private" : "int",
    "basin" : "str",
    "subvillage" : "str",
    "region" : "str",
    "region_code" : "int",
    "district_code" : "int",
    "lga" : "str",
    "ward" : "str",
    "population" : "int",
    "public_meeting" : "bool",
    "recorded_by" : "str",
    "scheme_management" : "str",
    "scheme_name" : "str",
    "permit" : "bool",
    "construction_year" : "int",
    "extraction_type" : "str",
    "extraction_type_group" : "str",
    "extraction_type_class" : "str",
    "management" : "str",
    "management_group" : "str",
    "payment" : "str",
    "payment_type" : "str",
    "water_quality" : "str",
    "quality_group" : "str",
    "quantity" : "str",
    "quantity_group" : "str",
    "source" : "str",
    "source_type" : "str",
    "source_class" : "str",
    "waterpoint_type" : "str",
    "waterpoint_type_group" : "str"
}

categorical_columns = ["funder",
                        "installer",
                        "wpt_name",
                        "basin",
                        "subvillage",
                        "region",
                        "lga",
                        "ward",
                        "recorded_by",
                        "scheme_management",
                        "scheme_name",
                        "extraction_type",
                        "extraction_type_group",
                        "extraction_type_class",
                        "management",
                        "management_group",
                        "payment",
                        "payment_type",
                        "water_quality",
                        "quality_group",
                        "quantity",
                        "quantity_group",
                        "source",
                        "source_type",
                        "source_class",
                        "waterpoint_type",
                        "waterpoint_type_group"]

def getCategoriesAndNumbers(route = "../data/train_values.csv"):
    data = None
    with open(route) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",", quotechar="|")
        i=1
        names = []
        d = []
        for row in spamreader:
            if i==1:
                d = {"id": []}
                names.append("id")
                for r in row[1:]:
                    d[r] = []
                    names.append(r)
            else:
                for r,n in zip(row, names):
                    d[n].append(r)
            i+=1

    data = pd.DataFrame(d)
    data = data.astype(types)

    for cat in categorical_columns:
        data[cat] = pd.Categorical(data[cat])

    aglomerated = dict()

    for cat in categorical_columns:
        if len(data[cat].value_counts())<100:
            continue

        threshold = data[cat].value_counts().iloc[0]*0.1
        new_cat = "aglomerate"

        bad_categories = np.array(data[cat].value_counts().index[np.where(data[cat].value_counts()<=threshold)[0]].astype("str"))
        aglomerated[cat] = bad_categories
        data[cat] = data[cat].cat.add_categories(new_cat)

        for bc in bad_categories:
            data[cat][data[cat]==bc] = new_cat
        data[cat] = data[cat].cat.remove_unused_categories()

    categories = dict()

    for cat in categorical_columns:
        categories[cat] = dict()
        string_categories = data[cat].cat.categories
        data[cat] = pd.Categorical(data[cat].cat.codes)
        numeric_categories = data[cat].cat.categories
        for s,n in zip(string_categories, numeric_categories):
            if s=="aglomerate":
                for bc in aglomerated[cat]:
                    categories[cat][bc]=n
            else:
                categories[cat][s]=n
            categories[cat][s]=n
    return categories

def preprocessingTest(route = "../data/test_values.csv"):
    d=None
    data = None
    with open(route) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",", quotechar="|")
        i=1
        names = []
        d = []
        for row in spamreader:
            if i==1:
                d = {"id": []}
                names.append("id")
                for r in row[1:]:
                    d[r] = []
                    names.append(r)
            else:
                for r,n in zip(row, names):
                    d[n].append(r)
            i+=1

    data = pd.DataFrame(d)
    data = data.astype(types)

    # Convertimos la fecha a float
    min_date = data["date_recorded"].min()
    for i in range(len(data)):
        # In days
        data["date_recorded"][i] = (data["date_recorded"][i]-min_date).total_seconds()/86400

    categories = getCategoriesAndNumbers()

    for cat in categorical_columns:
        print(cat)
        for i in range(len(data)):
            if data[cat][i] not in categories[cat]:
                data[cat][i]="aglomerate"
            data[cat][i] = categories[cat][data[cat][i]]
        data[cat]=pd.Categorical(data[cat])

    data = pd.DataFrame(KNNImputer().fit_transform(data), columns=data.keys())

    data = pd.get_dummies(data,prefix=categorical_columns, columns=categorical_columns)

    return data

###############################################################################


def preprocessing(route = "../data/train_values.csv"):
    labels = None
    d=None
    with open("../data/train_labels.csv") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",", quotechar="|")
        i=1
        names = []
        d = []
        for row in spamreader:
            if i==1:
                d = {"id": []}
                names.append("id")
                for r in row[1:]:
                    d[r] = []
                    names.append(r)
            else:
                for r,n in zip(row, names):
                    d[n].append(r)
            i+=1
    labels = pd.DataFrame(d)
    labels = np.array(labels["status_group"])

    data = None
    with open(route) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",", quotechar="|")
        i=1
        names = []
        d = []
        for row in spamreader:
            if i==1:
                d = {"id": []}
                names.append("id")
                for r in row[1:]:
                    d[r] = []
                    names.append(r)
            else:
                for r,n in zip(row, names):
                    d[n].append(r)
            i+=1

    data = pd.DataFrame(d)
    data = data.astype(types)

    # Convertimos la fecha a float
    min_date = data["date_recorded"].min()
    for i in range(len(data)):
        # In days
        data["date_recorded"][i] = (data["date_recorded"][i]-min_date).total_seconds()/86400

    for cat in categorical_columns:
        data[cat] = pd.Categorical(data[cat])

    for cat in categorical_columns:
        print("Filtering the column " + cat + " before One Hot Encoding, num. categories: " + str(len(data[cat].cat.categories)))
        if len(data[cat].value_counts())<100:
            continue

        threshold = data[cat].value_counts().iloc[0]*0.1
        new_cat = "aglomerate"

        bad_categories = np.array(data[cat].value_counts().index[np.where(data[cat].value_counts()<=threshold)[0]].astype("str"))
        data[cat] = data[cat].cat.add_categories(new_cat)

        for bc in bad_categories:
            data[cat][data[cat]==bc] = new_cat
        data[cat] = data[cat].cat.remove_unused_categories()

    for cat in categorical_columns:
        data[cat] = pd.Categorical(data[cat].cat.codes)

    data = pd.DataFrame(KNNImputer().fit_transform(data), columns=data.keys())

    data = pd.get_dummies(data,prefix=categorical_columns, columns=categorical_columns)

    return data,labels


def savePreprocessed(route_values = "../data/preprocessed_values.txt", route_labels = "../data/preprocessed_labels.txt"):
    X,y = preprocessing()
    X,y = reduceDim(X,y,100)
    names = np.array(X.keys())
    X = np.array(X)

    fich_values = open(route_values, "w")
    for n in names[:-1]:
        fich_values.write(str(n) + ",")
    fich_values.write(str(names[-1]) + "\n")

    for row in X:
        for r in row[:-1]:
            fich_values.write(str(r) + ",")
        fich_values.write(str(r) + "\n")
    fich_values.close()

    fich_labels = open(route_labels, "w")
    for l in y:
        fich_labels.write(str(l) + "\n")
    fich_labels.close()
