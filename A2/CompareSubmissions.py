import numpy as np

def find_majority(k):
    myMap = {}
    maximum = ( -1, 0 ) # (occurring element, occurrences)
    for n in k:
        if n in myMap: myMap[n] += 1
        else: myMap[n] = 1

        # Keep track of maximum on the go
        if myMap[n] > maximum[1]: maximum = (n,myMap[n])

    return maximum

submissionNames = [
# "KNN6_downsampled.csv",
# "KNN6_small.csv",
# "baggingKNN_small.csv",
"KNN6_deskewed_normalized.csv",
"baggingKNN.csv",
"KNN6_deskewed.csv",
"KNN6.csv",
"KNN6_L3.csv",
"KNN6_L3_gaussian_blurred.csv",
]

submissionVals = [-1]*10000

for sub in submissionNames:
	with open("submissions/"+sub, "r") as subfile:
		for line in subfile:
			try:
				index, val = line.split(",")
				index = int(index) - 1
				val = int(val)
				if submissionVals[index] == -1:
					submissionVals[index] = [val]
				else:
					submissionVals[index].append(val)
			except:
				continue

for i, vals in enumerate(submissionVals):
	if isinstance(vals, list):
		elm, count = find_majority(vals)
		if count > 0.5*len(vals):
		# if count > 1:
			submissionVals[i] = elm
		else:
			print(i, vals, elm, vals[-1])
			submissionVals[i] = vals[-1]


with open("submission_combined.csv", "w+") as submission:
	submission.write("ImageID,Digit\n")
	for i,y in enumerate(submissionVals):
		submission.write(str(int(i+1))+','+str(int(y))+'\n')

