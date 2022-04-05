from operator import indexOf
import numpy as np;
import random;

Allnumbers =[];
file = open("smalldigits.csv");

vecWords = [0,1,2,3,4,5,6,7,8,9];


line = file.readline();
while (line != "" ):
    Allnumbers.append(line);
    line = file.readline();

file.close();

numbers = np.random.choice(Allnumbers, size=(int(len(Allnumbers)*0.8)), replace=False);


pTotalNumbers = [0,0,0,0,0,0,0,0,0,0];

temp = np.zeros((len(numbers), 65));

for pos,i in enumerate(numbers):
    noTemp = i.split(",");
    pTotalNumbers[int(noTemp[-1].replace("\n",""))] += 1;
    temp[pos] = noTemp;

numbers = temp;

temp = np.zeros((len(Allnumbers), 65));

for pos,i in enumerate(Allnumbers):
    noTemp = i.split(",");
    temp[pos] = noTemp;

Allnumbers = temp;

def setdiff2d_list(arr1, arr2):
    delta = set(map(tuple, arr2))
    return np.array([x for x in arr1 if tuple(x) not in delta])




# vectors = np.zeros((12,len(vecWords)));
# reviews = np.random.choice(allReviews, size=12, replace=False);

# i =0;
# nGood=0;
# nBad=0;

# for review in reviews:
#     review = review.split();
#     if (review[0] == "-1"): 
#         nBad += 1;
#     else:
#         nGood += 1;

Probability = np.zeros((65,10));

for no in numbers:
    i = int(no[-1]);

    for j, bi in enumerate(no):
        if (bi == 1):
            Probability[j][i] += 1/pTotalNumbers[i]
    
test = setdiff2d_list(Allnumbers, numbers)

print(len(test))

# def encodeVector(s):
#     s = s.split();
#     temp = np.zeros(np.size(vecWords))
#     for w in s:
#         if (w in vecWords):
#             temp[vecWords.index(w)] = 1;
    
#     return temp

# def calcAgivenB(a, b):
#     total = 1;
#     for index,j in enumerate(a):
#         if (b[index] == 0):
#             b[index] = 1/(nGood+2)
#         if (j == 1):
            
#             total = total*b[index]
#         if (j == 0):
#             total = total*(1-b[index])
        
#     return total

# testcase = encodeVector(test[0])

# def calcEnd(a, pA, b, pB):
#     return (a*pA/12)/((a*pA/12)+(b*pB/12))

# def printClassModel():
#     print("word\tGood\tBad\t")
#     for i,w in enumerate(vecWords):
#         print(w+"\t"+str(pGood[i])+"\t"+str(pBad[i]))

#     print("Probability good:"+str(nGood))
#     print("Probability bad:"+str(nBad))

def determineNumber(no):

    prob = calcEnd(calcAgivenB(no, pGood),nGood,calcAgivenB(vector,pBad),nBad)
    print("Probability good: "+str(prob))
    if (prob >= 0.5):
        print("This review is most likely good")
        print("\n")
        return 1
    else:
        print("this review is most likely bad")
        print("\n")
        return 0

    



# def confusionMatrix():
#     temp = np.zeros((2,2));
#     for t in test:
#         actualType = t.split()[0]
#         determinedType = determineReview(t)

#         if ((actualType == "1") and (determinedType == 1)):
#             temp[0,0]+=1;
#         if ((actualType == "1") and (determinedType == 0)):
#             temp[0,1]+=1; 
#         if ((actualType == "-1" and determinedType == 1)):
#             temp[1,0]+=1;
#         if ((actualType == "-1" and determinedType == 0)):
#             temp[1,1]+=1;
#     return temp


# def calcAccuracy(arr):
#     accuracy = (arr[0,0]+arr[1,1])/(arr[0,0]+arr[0,1]+arr[1,0]+arr[1,1]);
#     return accuracy;


# cm = confusionMatrix()
# print(cm)
# print(calcAccuracy(cm))