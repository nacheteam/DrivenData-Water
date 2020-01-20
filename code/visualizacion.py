import numpy as np
from sklearn.manifold import TSNE
import csv
import pandas as pd

import plotly.express as px
import plotly

from scipy import sparse

import matplotlib.pyplot as plt

from random import randint
colors = []

for i in range(6):
    colors.append('#%06X' % randint(0, 0xFFFFFF))

data = None

with open("../data/train_values.csv") as csvfile:
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
data = data.astype({
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
})

import pdb; pdb.set_trace()

# En primer lugar pasamos los strings a variables categoricas y luego a enteros


# funder
data["funder"] = pd.Categorical(data["funder"])
#data["funder"] = pd.Categorical(data["funder"].cat.codes)

# installer
data["installer"] = pd.Categorical(data["installer"])
#data["installer"] = pd.Categorical(data["installer"].cat.codes)

# wpt_name
data["wpt_name"] = pd.Categorical(data["wpt_name"])
#data["wpt_name"] = pd.Categorical(data["wpt_name"].cat.codes)

# basin
data["basin"] = pd.Categorical(data["basin"])
#data["basin"] = pd.Categorical(data["basin"].cat.codes)

# subvillage
data["subvillage"] = pd.Categorical(data["subvillage"])
#data["subvillage"] = pd.Categorical(data["subvillage"].cat.codes)

# region
data["region"] = pd.Categorical(data["region"])
#data["region"] = pd.Categorical(data["region"].cat.codes)

# lga
data["lga"] = pd.Categorical(data["lga"])
#data["lga"] = pd.Categorical(data["lga"].cat.codes)

# ward
data["ward"] = pd.Categorical(data["ward"])
#data["ward"] = pd.Categorical(data["ward"].cat.codes)

# recorded_by
data["recorded_by"] = pd.Categorical(data["recorded_by"])
#data["recorded_by"] = pd.Categorical(data["recorded_by"].cat.codes)

# scheme_management
data["scheme_management"] = pd.Categorical(data["scheme_management"])
#data["scheme_management"] = pd.Categorical(data["scheme_management"].cat.codes)

# scheme_name
data["scheme_name"] = pd.Categorical(data["scheme_name"])
#data["scheme_name"] = pd.Categorical(data["scheme_name"].cat.codes)

# extraction_type
data["extraction_type"] = pd.Categorical(data["extraction_type"])
#data["extraction_type"] = pd.Categorical(data["extraction_type"].cat.codes)

# extraction_type_group
data["extraction_type_group"] = pd.Categorical(data["extraction_type_group"])
#data["extraction_type_group"] = pd.Categorical(data["extraction_type_group"].cat.codes)

# extraction_type_class
data["extraction_type_class"] = pd.Categorical(data["extraction_type_class"])
#data["extraction_type_class"] = pd.Categorical(data["extraction_type_class"].cat.codes)

# management
data["management"] = pd.Categorical(data["management"])
#data["management"] = pd.Categorical(data["management"].cat.codes)

# management_group
data["management_group"] = pd.Categorical(data["management_group"])
#data["management_group"] = pd.Categorical(data["management_group"].cat.codes)

# payment
data["payment"] = pd.Categorical(data["payment"])
#data["payment"] = pd.Categorical(data["payment"].cat.codes)

# payment_type
data["payment_type"] = pd.Categorical(data["payment_type"])
#data["payment_type"] = pd.Categorical(data["payment_type"].cat.codes)

# water_quality
data["water_quality"] = pd.Categorical(data["water_quality"])
#data["water_quality"] = pd.Categorical(data["water_quality"].cat.codes)

# quality_group
data["quality_group"] = pd.Categorical(data["quality_group"])
#data["quality_group"] = pd.Categorical(data["quality_group"].cat.codes)

# quantity
data["quantity"] = pd.Categorical(data["quantity"])
#data["quantity"] = pd.Categorical(data["quantity"].cat.codes)

# quantity_group
data["quantity_group"] = pd.Categorical(data["quantity_group"])
#data["quantity_group"] = pd.Categorical(data["quantity_group"].cat.codes)

# source
data["source"] = pd.Categorical(data["source"])
#data["source"] = pd.Categorical(data["source"].cat.codes)

# source_type
data["source_type"] = pd.Categorical(data["source_type"])
#data["source_type"] = pd.Categorical(data["source_type"].cat.codes)

# source_class
data["source_class"] = pd.Categorical(data["source_class"])
#data["source_class"] = pd.Categorical(data["source_class"].cat.codes)

# waterpoint_type
data["waterpoint_type"] = pd.Categorical(data["waterpoint_type"])
#data["waterpoint_type"] = pd.Categorical(data["waterpoint_type"].cat.codes)

# waterpoint_type_group
data["waterpoint_type_group"] = pd.Categorical(data["waterpoint_type_group"])
#data["waterpoint_type_group"] = pd.Categorical(data["waterpoint_type_group"].cat.codes)


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
                        "waterpoint_type"]

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


dummy = pd.get_dummies(data,prefix=categorical_columns, columns=categorical_columns)

############################
# Representación de los datos con TSNE
labels = np.array(dummy["waterpoint_type_group"])

dummy = dummy.drop(columns = ["waterpoint_type_group", "date_recorded"])

reduced = TSNE(n_components=2).fit_transform(np.array(dummy))

cl0 = np.array([reduced[i] for i in range(len(reduced)) if labels[i]=="cattle trough"])
cl1 = np.array([reduced[i] for i in range(len(reduced)) if labels[i]=="communal standpipe"])
cl2 = np.array([reduced[i] for i in range(len(reduced)) if labels[i]=="dam"])
cl3 = np.array([reduced[i] for i in range(len(reduced)) if labels[i]=="hand pump"])
cl4 = np.array([reduced[i] for i in range(len(reduced)) if labels[i]=="improved spring"])
cl5 = np.array([reduced[i] for i in range(len(reduced)) if labels[i]=="other"])

plt.scatter(cl0[:,0], cl0[:,1], color = colors[0], label = "Clase 0")
plt.scatter(cl1[:,0], cl1[:,1], color = colors[1], label = "Clase 1")
plt.scatter(cl2[:,0], cl2[:,1], color = colors[2], label = "Clase 2")
plt.scatter(cl3[:,0], cl3[:,1], color = colors[3], label = "Clase 3")
plt.scatter(cl4[:,0], cl4[:,1], color = colors[4], label = "Clase 4")
plt.scatter(cl5[:,0], cl5[:,1], color = colors[5], label = "Clase 5")
plt.legend()
plt.show()


'''
d = pd.DataFrame({"x": reduced[:,0], "y": reduced[:,1], "z": reduced[:,2], "labels": labels})
fig = px.scatter_3d(d, x="x", y="y", z="z", color="labels")
fig.update_traces(marker=dict(size=5,
                              line=dict(width=1,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
plotly.offline.plot(fig, filename="plot_ohe_3d.html", auto_open=True)
'''
