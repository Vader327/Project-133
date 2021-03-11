import csv
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

rows = []

with open("data.csv", "r", encoding="UTF-8") as f:
  reader = csv.reader(f)

  for row in reader:
    rows.append(row)

headers = rows[0]
planet_data_rows = rows[1:]

names = []
masses = []
radiuses = []
gravity = []

for planet_data in planet_data_rows:
  if len(planet_data) == 13:
    names.append(planet_data[1])
    masses.append(planet_data[3])
    radiuses.append(planet_data[4])
    gravity.append(planet_data[12])

X = []
wcss = []

for index, planet_mass in enumerate(masses):
  temp_list = [radiuses[index], planet_mass]
  X.append(temp_list)

for i in range(1, 11):
  k_means = KMeans(n_clusters=i, init='k-means++', random_state=42)
  k_means.fit(X)
  wcss.append(k_means.inertia_)

plt.figure(figsize=(10,5))
sns.lineplot(range(1, 11), wcss, marker='o', color='red')
plt.title("The Elbow Method")
plt.xlabel('No. of Clusters')
plt.ylabel('WCSS')
plt.show()
