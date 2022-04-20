import numpy as np

#add files to an array 

words = []
allPages = []
file = open("smalldigits.csv");
 # add unseen words to array
def add_to_array(book):
    temp = book.split(" ");
    for w in temp:
        if (w not in words):
            words.append(w);

# reading a file into the array of words
def readIn(file, i):
    count = 0;
    bookTemp = ""
    line = file.readline();
    while (line != "" ):
        bookTemp += line;
        allPages.append(str(i)+" "+line)
        line = file.readline(); 
        count += 1
    add_to_array(bookTemp);

def encodeVector(s):
    s = s.split();
    temp = np.zeros(np.size(words))
    for w in s:
        if (w in words):
            temp[words.index(w)] = 1;
    
    return temp

def countBooks(data):
    for s in data:
        index = int(s.split(" ")[0]);
        nBook[index] += 1;

def addValue():


def calcProbability(data):
    for s in data:
        s = s.split(" ")
        index = int(s[0])
        s.pop(0)
        temp = encodeVector(s)

        for pos,t in enumerate(temp):
        if (t == 1):
            if (index == 0){
                
            }

nBook = [0,0,0,0,0,0,0];
 
readIn(open("hp_books/HP1.txt"), 0)
readIn(open("hp_books/HP2.txt"), 1)
readIn(open("hp_books/HP3.txt"), 2)
readIn(open("hp_books/HP4.txt"), 3)
readIn(open("hp_books/HP5.txt"), 4)
readIn(open("hp_books/HP6.txt"), 5)
readIn(open("hp_books/HP7.txt"), 6)



# print(allPages)

trainingPages = np.random.choice(allPages, size=(int(len(allPages)*0.8)), replace=False);

countBooks(trainingPages)
print(nBook)


# for page in trainingPages:
#     print(encodeVector(page));



# initialize the probability of a word being in a book arrays

pH = np.zeros((7,len(words))



# print(words);