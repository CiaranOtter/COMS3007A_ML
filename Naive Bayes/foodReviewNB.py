from operator import indexOf
import numpy as np;
import random;

allReviews =[];
file = open("simple-food-reviews.txt");

vecWords = [];


line = file.readline();
while (line != "" ):
    allReviews.append(line);
    line = file.readline();

for i in range(len(allReviews)):
    temp = allReviews[i].split();
    for w in temp:
        if (w == "1") or (w == "-1"):
            continue;
        if w not in vecWords:
            vecWords.append(w);



file.close();
pGood = np.zeros(len(vecWords))
pBad = np.zeros(len(vecWords))

vectors = np.zeros((12,len(vecWords)));
reviews = np.random.choice(allReviews, size=12, replace=False);

i =0;
nGood=0;
nBad=0;

for review in reviews:
    review = review.split();
    if (review[0] == "-1"): 
        nBad += 1;
    else:
        nGood += 1;

        

for review in reviews:
    review = review.split();
    gb = False;
    if (review[0] == "1"):
        gb = True;
    temp = np.zeros(np.size(vecWords));
    for w in review:
        if (w in vecWords):
            pos = vecWords.index(w);
            temp[pos] = 1

    for pos,t in enumerate(temp):
        if (t == 1):
            if (gb):
                pGood[pos] += 1/nGood
            else:
                pBad[pos] += 1/nBad
   
    vectors[i] = temp;
    i += 1

     
test = list(set(allReviews)-set(reviews))

def encodeVector(s):
    s = s.split();
    temp = np.zeros(np.size(vecWords))
    for w in s:
        if (w in vecWords):
            temp[vecWords.index(w)] = 1;
    
    return temp

def calcAgivenB(a, b):
    total = 1;
    for index,j in enumerate(a):
        if (b[index] == 0):
            b[index] = 1/(nGood+2)
        if (j == 1):
            
            total = total*b[index]
        if (j == 0):
            total = total*(1-b[index])
        
    return total

testcase = encodeVector(test[0])

def calcEnd(a, pA, b, pB):
    return (a*pA/12)/((a*pA/12)+(b*pB/12))

def printClassModel():
    print("word\tGood\tBad\t")
    for i,w in enumerate(vecWords):
        print(w+"\t"+str(pGood[i])+"\t"+str(pBad[i]))

    print("Probability good:"+str(nGood))
    print("Probability bad:"+str(nBad))

def determineReview(review):
    print(review)
    vector = encodeVector(review)

    prob = calcEnd(calcAgivenB(vector, pGood),nGood,calcAgivenB(vector,pBad),nBad)
    print("Probability good: "+str(prob))
    if (prob >= 0.5):
        print("This review is most likely good")
        print("\n")
        return 1
    else:
        print("this review is most likely bad")
        print("\n")
        return 0

    



def confusionMatrix():
    temp = np.zeros((2,2));
    for t in test:
        actualType = t.split()[0]
        determinedType = determineReview(t)

        if ((actualType == "1") and (determinedType == 1)):
            temp[0,0]+=1;
        if ((actualType == "1") and (determinedType == 0)):
            temp[0,1]+=1; 
        if ((actualType == "-1" and determinedType == 1)):
            temp[1,0]+=1;
        if ((actualType == "-1" and determinedType == 0)):
            temp[1,1]+=1;
    return temp


def calcAccuracy(arr):
    accuracy = (arr[0,0]+arr[1,1])/(arr[0,0]+arr[0,1]+arr[1,0]+arr[1,1]);
    return accuracy;


cm = confusionMatrix()
print(cm)
print(calcAccuracy(cm))
# print("All words:")
# print(vecWords)
# print("\n")

# print("reviews:")
# print(reviews)
# print("\n")

# printClassModel()
# print("\n")

# print("new review test vector:")
# print(testcase)
# print("\n")

# print("Probability of review given good:")
# print(calcAgivenB(testcase, pGood))
# print("\n")

# print("Probability of review given bad:")
# print(calcAgivenB(testcase,pBad))
# print("\n")

# print()


# print(vecWords);
# print(reviews)
# print(pGood);
# print(pBad);

# print(reviews);

