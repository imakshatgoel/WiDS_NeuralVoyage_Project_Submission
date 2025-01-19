import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("C:\WiDS-Neural-Voyage\Week 1\iris_dataset.csv")
#rint(df.to_string())
#rint(df.sort_values(['sepal length (cm)'], ascending = False))
#rint(df[['petal length (cm)', 'petal width (cm)']])
#rint(df.iloc[0:20])
#rint(df.loc[df['sepal length (cm)'] > 6])
index = [1 + x for x in range(150)]
sepal_length = df['sepal length (cm)']
sepal_width = df['sepal width (cm)']
petal_length = df['petal length (cm)']
petal_width = df['petal width (cm)']
target = df['target']
plt.plot(index, sepal_length, label = "Sepal Length")
plt.plot(index, sepal_width, label = "Sepal Width")
plt.plot(index, petal_length, label = "Petal Length")
plt.plot(index, petal_width, label = "Petal Width")
plt.plot(index, target, label = "Target")
plt.xlabel("Index")
plt.ylabel("Length(cm)")
plt.legend()
plt.show()
setosa_x = df.loc[df['target'] == 0, ['petal length (cm)']]
setosa_y = df.loc[df['target'] == 0, ['petal width (cm)']]
versicolor_x = df.loc[df['target'] == 1, ['petal length (cm)']]
versicolor_y = df.loc[df['target'] == 1, ['petal width (cm)']]
virginica_x = df.loc[df['target'] == 2, ['petal length (cm)']]
virginica_y = df.loc[df['target'] == 2, ['petal width (cm)']]
plt.scatter(setosa_x, setosa_y, c = 'red', label = 'Setosa')
plt.scatter(versicolor_x, versicolor_y, c = 'blue', label = 'Versicolor')
plt.scatter(virginica_x, virginica_y, c = 'green', label = 'Virginica')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend()

plt.show()


