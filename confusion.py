import cPickle
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
import matplotlib.pyplot as plt

best = cPickle.load(open("best_models_no_cutoff.pickle", "r"))
for i in range(len(best)):
	for j in range(len(best[i])):
		if (best[i][j] == 3): best[i][j] = 4
		elif (best[i][j] == 4): best[i][j] = 3

#indeterminate = np.ones((100)) * -1
#y_pred = np.array([best[0], best[6], best[12], best[24], best[18], best[30], best[36]])
#y_pred = np.array([best[1], best[7], best[13], best[19], best[25], best[31], best[37], indeterminate])
y_pred = np.array([best[2], best[8], best[14], best[26], best[20], best[32], best[38]])
#y_pred = np.array([best[3], best[9], best[15], best[21], best[27], best[33], best[39], indeterminate])
#y_pred = np.array([best[4], best[10], best[16], best[22], best[28], best[34], best[40], indeterminate])
#y_pred = np.array([best[5], best[11], best[17], best[23], best[29], best[35], best[41], indeterminate])


y_pred = y_pred.flatten()

#print y_pred
#print y_pred.shape
#y_pred = [best[0], best[6], best[12], best[18], best[24], best[30], best[36]]

#print y_pred.shape

#y_true = [0, 1, 2, 4, 3, 5, -1]
#y_true = []
y_true = np.zeros((700))

y_true = np.array([0, 1, 2, 3, 4, 5, 6])
y_true = np.repeat(y_true, 100)


#print y_true
#print y_true.shape

#print len(y_true)
#print len(y_true[0])

#labels = ["unrelated", "full-sibs", "parent-child", "3rd Degree", "2nd Degree", "4th Degree", "double-first-cousins"]
matrix =  confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
plt.figure()
plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Simulations with 1.0x Coverage")
plt.colorbar()
thresh = matrix.max()/2
#line, = plt.plot(["unrelated", "full-sibs", "parent-child", "3rd Degree", "2nd Degree", "4th Degree", "double-first-cousins"], label = 'Inline label')
#plt.legend()
#plt.ylabel("Relationships")
#ymarks = ["unrelated", "full-sibs", "parent-child", "2nd Degree", "3rd Degree", "4th Degree", "double-first-cousins"]
#plt.xticks(ymarks, matrix)
locs, labels = plt.yticks()
plt.rcParams.update({'font.size': 14})
plt.yticks(np.arange(0,7, step =1))
plt.yticks(np.arange(8), ('Unrelated', 'Full Siblings', 'Parent/Offspring', '2nd Degree', '3rd Degree', '4th Degree', 'Double-First-Cousins'))
for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
	plt.text(j, i, format(matrix[i,j], '.2f'),
		horizontalalignment = "center",
		color="white"if matrix[i,j] > thresh else "black")
plt.tight_layout()
#plt.show()
plt.savefig('cov1.png')

#print matrix
#pfile = open("confusion_matrix_cov_8.pickle", "w")
#cPickle.dump(matrix, pfile)
#pfile.close()
#np.savetxt("confusion_matrix_no_cutoff_cov0.1.tab", matrix, fmt = '%i', newline = '\n')


