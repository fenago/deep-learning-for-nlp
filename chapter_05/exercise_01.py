# load text
filename = 'exercise.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words by comma
words = text.split(',')
print(words[:50])