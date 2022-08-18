import random

with open("simple-food-reviews.txt") as f:
    lines = f.readlines();

for i in range(len(lines)):
    lines[i] = lines[i].replace("\n","");

good = 0;
bad = 0;
conf = [];

trainingData = random.sample(lines, int(len(lines)*0.6));
print(lines)
print(len(trainingData))
testing_data = list(set(lines) - set(trainingData))
print(len(testing_data))


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
                "bad":0,
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


def testALine(line):
    print("running a test for a line on the models")

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

    print(line)
    print(binary)
    findProbability(binary)
        

def findProbability(binaryStr):
    probGood = 1;
    probBad = 1;
    for i in range(len(conf)):
        if (binaryStr[i] == 1):
            probGood = probGood*(conf[i]["good"]/conf[i]["count"]);
            probBad = probBad*(conf[i]["bad"]/conf[i]["count"]);
        else:
            probGood = probGood * (1 - (conf[i]["good"]/conf[i]["count"]));
            probBad = probBad * (1 - (conf[i]["bad"]/conf[i]["count"]));

    print(good, bad)
    print(probGood, probBad)
    givenGood = (probGood *good)/( (probGood*good) + (probBad*bad) )

    print(givenGood)

print(conf)
testALine(testing_data[1])