def lbs_to_klg() -> list:
    """
    Takes list of weights in lbs and multiplies by 0.453592 to convert to kilograms and adds to new list
    """
    # ASKS FOR DESIRED NUMBER OF WEIGHTS TO ENTER AND ADDS TO LIST
    length = int(input("Enter how many weights in lbs you will be entering -> "))
    weight_lst = []
    # ADDS WEIGHTS GIVEN TO LIST
    for i in range(length):
        weight = int(input("Enter a weight in lbs -> "))
        weight_lst.append(weight)

    # CONVERTS WEIGHTS AND ADDS TO NEW LIST
    convert = []
    for item in weight_lst:
        new = item * 0.453592
        new = round(new, 2)
        convert.append(new)

    return convert


def string_alternative(original: str) -> str:
    """
    Takes string and returns new string that takes every two characters of original string
    """
    new = ""
    # LOOPS THROUGH GIVEN STRING AND CONCATENATES EVERY SECOND CHARACTER
    for element in original[::2]:
        new += element

    return new


def file_word_counter(filename: str) -> dict:
    """
    Takes in a file and adds contents to a list, uses the list to create a dict,
    and then uses the dict to return a wordcount for each word in the file the contents were pulled from
    """

    # TAKES IN FILE CONTENTS AND CREATES LIST OF WORDS USED
    file = open(filename, 'r+')  # USED 'r+' TO READ FROM A FILE AND THEN ALLOW UPDATES TO EXISTING FILE
    words_lst = []
    for line in file:
        temp = line.split(' ')  # CREATES LIST OF SEPARATE WORDS
        for item in temp:
            item = item.strip('\n')  # GETS RID OF NEW LINE INPUT
            if item != '':  # DID THIS TO GET RID OF UNNECESSARY SPACES TAKEN IN AS WORDS
                words_lst.append(item)

    # CREATES DICTIONARY FOR WORD USAGE WITH INPUTS FROM LIST
    words_dict = {}

    for item in words_lst:
        if item in words_dict.keys():  # IF THE KEY ALREADY EXIST, INCREMENTS THE VALUE
            temp = int(words_dict.get(item))
            words_dict.update({item: temp + 1})
        elif item not in words_dict.keys():  # IF THE KEY DOES NOT EXIST, CREATES AND ADDS 1 TO VALUE
            words_dict.update({item: 1})

    # PLACES CONTENTS FROM DICTIONARY BACK INTO FILE
    file.write('\n')
    for item in words_dict:
        file.write(item)
        file.write(":")
        file.write(str(words_dict[item]))
        file.write('\n')

    file.close()

    return words_dict

if __name__ == "__main__":
    print(lbs_to_klg())  # PROBLEM 1
    print()
    print(string_alternative('good evening'))   # PROBLEM 2
    print()
    print(file_word_counter('test.txt'))    # PROBLEM 3
