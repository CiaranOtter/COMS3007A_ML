import random
import re

with open("simple-food-reviews.txt") as f:
    lines = f.readlines();

for i in range(len(lines)):
    lines[i] = lines[i].replace("\n","");

good = 0;
bad = 0;
conf = [];

trainingData = random.sample(lines, int(len(lines)*0.6));
testing_data = list(set(lines) - set(trainingData))


def findStats(line,conf, good, bad):
    lineArr = line.split(" ")

    if (lineArr[0] == "1"):
        good += 1
        isGood = True;
    else:
        bad += 1
        isGood = False;

    

    for i in range(len(lineArr)):
        if (lineArr[i] == "-1" or lineArr[i] == "1"):
            continue
        
        found = False
        for j in range(len(conf)):
            if (conf[j]["word"] == lineArr[i]):
                found = True;
                conf[j] = incGoodBad(conf[j], isGood)
                conf[j]["count"] += 1;
        
        if (not found):
            obj = {
                "word": lineArr[i],
                "good": 0,
                "bad": 0,
                "count": 0
            }
            obj = incGoodBad(obj, isGood)
            obj["count"] += 1;

            conf.append(obj)
        
    return good, bad


def incGoodBad(obj, isGood):
    if (isGood):
        obj["good"] += 1;
    else:
        obj["bad"] += 1;

    return obj;

for i in range(len(trainingData)):
    good, bad = findStats(trainingData[i], conf,good, bad)


def testALine(line, conf):

    line = line.replace("\n", "");
    line = line.split(' ');

    for i in range(len(line)):
        isFound = False;
        for j in range(len(conf)):
            if (line[i] == conf[j]["word"]):
                isFound = True;
                break;
        if (not isFound):
            obj = {
                "word": line[i],
                "good": 1,
                "bad": 1,
                "count": 1
            };
            conf.append(obj);
        

    binary = [0 for i in range(len(conf))];

    for i in range(len(conf)):
        isTrained = False;
        if (conf[i]["word"] in line):
            binary[i] = 1;

    findProbability(binary)
        

def laplaceSmooth(conf):
    for i in conf:
        if (i["good"] == 0 or i["bad"] == 0):
            i["good"] += 1;
            i["bad"] += 1;
            i["count"] += 2;

    return conf;

def findProbability(binaryStr):
    probGood = 1;
    probBad = 1;

    # conf = laplaceSmooth(conf);
    for i in range(len(conf)):
        if (binaryStr[i] == 1):
            probGood = probGood*(conf[i]["good"]/conf[i]["count"]);
            probBad = probBad*(conf[i]["bad"]/conf[i]["count"]);
        else:
            probGood = probGood * (1 - (conf[i]["good"]/conf[i]["count"]));
            probBad = probBad * (1 - (conf[i]["bad"]/conf[i]["count"]));

    givenGood = (probGood *good)/( (probGood*good) + (probBad*bad) )
    givenBad = 1 - givenGood;
    # Ciaran you are a a dumb-ass sometimes

    results = False;
    if (givenGood > givenBad):
        results = True;

    return results;

conf = laplaceSmooth(conf);

print(testing_data[1])

if (testALine(testing_data[1], conf)):
    print("good review")
else:
    print("bad reveiw");