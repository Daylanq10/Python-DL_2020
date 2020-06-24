#1

# Two major differences between Pyton 2 and Python 3 are that in using
# the division operator in Python 2 if you do not use numbers of type float and
# use type integer then the division result will be a rounded whole number and not
# entirely accurate where as in Python 3 the divided result of integers will be
# more precise and return a float as needed. Another difference is the print function
# where in Python 2 you do not need things like parenthesis around a desired
# string and in Python 3 you need parenthesis around a desired string.

#2

sample_input = input("Enter a string -> ")

#removes what is at last two indexes
sample_input = sample_input[0:-2]

#reverses the string
sample_input = sample_input[::-1]

print(sample_input)


x = int(input("\nEnter a number to preform arithmetic operations on -> "))
y = int(input("Enter the other number -> "))

print(x, "+", y, "=", x + y)
print(x, "-", y, "=", x - y)
print(y, "-", x, "=", y - x)
print(x, "*", y, "=", x * y)
print(x, "/", y, "=", x / y)

#3

sentence = str(input("\nEnter a sentence. Occurence of 'python' will be replaced with 'pythons' -> "))

#replaces python with pythons
sentence = sentence.replace('python','pythons')

print(sentence)

