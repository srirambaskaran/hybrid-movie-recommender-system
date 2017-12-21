import csv
import numpy as np
import scipy
import matplotlib.pyplot as plt

from collections import defaultdict
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial.distance import pdist, squareform

ratings_file = "ratings.csv"
movies = set(map(lambda x: int(x[1]), map(lambda x:x.split(","), open(ratings_file).readlines() [1:])))
genre_file = "movies.csv"
genre = filter(lambda x: x[0] in movies , \
                    map(lambda x: (int(x[0]), x[2].split("|")), \
                        filter(lambda x: x[2] != '(no genres listed)', \
                         [row for row in csv.reader(open(genre_file, 'r'))][1:]
                         )
                        )
                    )
labels = map(lambda x: x[0], genre)
genre = map(lambda x: x[1], genre)
encoder = preprocessing.MultiLabelBinarizer()
encoded_genre = encoder.fit_transform(genre)

np.savetxt("encoded.txt", encoded_genre, fmt='%d')
np.savetxt("labels.txt", labels, fmt='%d')

clusters = fclusterdata(encoded_genre, criterion='distance', metric='jaccard',t=0.38, method='complete')
np.savetxt("clusters.txt", np.array(list(enumerate(clusters.tolist()))), fmt="%d")
score = silhouette_score(encoded_genre, clusters, metric='jaccard')


cluster_genres = defaultdict(lambda: defaultdict(int))
cluster_total = defaultdict(int)
for i, cluster_id in enumerate(clusters):
    for g in genre[i]:
        cluster_genres[cluster_id][g] += 1

    cluster_total[cluster_id] += 1

for cluster in cluster_genres:
    print str(cluster) +"\tCluster count: "+str(cluster_total[cluster])
    for g in cluster_genres[cluster]:
        # if cluster_genres[cluster][g] > 0.65 * cluster_total[cluster]:
        print "\t"+str(g)
print str(score), len(set(clusters)), len(movies)
