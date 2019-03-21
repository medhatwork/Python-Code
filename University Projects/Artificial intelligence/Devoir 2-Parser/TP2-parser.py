#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Presenté par
Nom : Ahmed Mohamed                  Matricule : 20024603
Nom : Boumediene Boukharouba         Matricule : 20032279
'''


def start(number_of_words):
    line_res, all_words, all_types, indices = [], [], [], []

    lines = open("./interest.acl94.txt", "r")

    with open(('interest-' + str(number_of_words) + '-words.arff'), "w") as fp:
        fp.write("@RELATION interest")

        for line in lines:

            words = {}

            line = line.replace('.', "X")
            line = line.replace('[', "")
            line = line.replace(']', "")
            line = line.replace(',', "")
            line = line.replace(':', "")
            line = line.replace('``', "")

            ind = 0
            line_arr = line.split(' ')

            for i, f in enumerate(line_arr):
                if 'XXX' in f:
                    line_arr = line_arr[i].replace('XXX/', '/XXX')

            line_arr = [x for x in line_arr if x != '' and '*interest' not in x]

            for i, w in enumerate(line_arr):
                if any(c.isalpha() for c in w) and '/' not in w:
                    line_arr[i] = w + '/ZZZ'

                elif w[-1] == '/':
                    line_arr[i] = w + "ZZZ"

            for i, x in enumerate(line_arr):

                if any(c.isalpha() for c in x) and 'interest_' not in x:
                    x_arr = x.split('/')

                    if len(x_arr) > 1:
                        x_0, x_1 = x_arr[0], x_arr[1]

                        if any(c.isalpha() for c in x_0) and (x_0 not in all_words or x_1 not in all_types):
                            if "'" in x_0:
                                x_0 = x_0.replace("'", "\\'")
                                x_0 = "'" + x_0 + "'"
                            if "$" in x_0:
                                x_0 = x_0.replace("$", "\\S")
                                x_0 = "'" + x_0 + "'"

                            if x_0 not in all_words:
                                all_words.append(x_0)

                            if "$" in x_1:
                                x_1 = x_1.replace("$", "\\S")
                                x_1 = "'" + x_1 + "'"

                            if len(x_1) > 0 and x_1 not in all_types and any(c.isalpha() for c in x_1):
                                all_types.append(x_1)

            getwords = 0
            interest_class = ''

            for index, x in enumerate(line_arr):
                if 'interest' in x and any(c.isnumeric() for c in x):
                    interest_class = x.split('/')[0][-1]
                    ind = index
                    getwords = 1

            if getwords == 1:
                for i in range(1, number_of_words + 1):
                    words['previous_word' + str(i)], words['previous_word' + str(i) + '_type'] = "NULL", "NULL"
                    words['next_word' + str(i)], words['next_word' + str(i) + '_type'] = "NULL", "NULL"

                i = x = 1

                while i <= number_of_words:

                    if ind - x >= 0:
                        words['previous_word' + str(i)], words['previous_word' + str(i) + '_type'] = \
                            get_word_and_type(line_arr, ind - x)
                        x += 1
                    i += 1

                i = x = 1
                while i <= number_of_words:
                    if ind + x < len(line_arr):
                        words['next_word' + str(i)], words['next_word' + str(i) + '_type'] = \
                            get_word_and_type(line_arr, ind + x)
                        x += 1

                    i += 1

            if interest_class != '':
                previous_words, next_words = '', ''
                for i in range(number_of_words, 0, -1):
                    previous_words += words[('previous_word' + str(i))] + "," + words[
                        ('previous_word' + str(i) + "_type")] + ","

                for i in range(1, number_of_words + 1):
                    next_words += words[('next_word' + str(i))] + "," + words[('next_word' + str(i) + "_type")] + ","
                line_res.append(previous_words + next_words + interest_class + "\n")

        all_words.append("NULL")
        all_types.append("NULL")

        get_words(number_of_words, all_words, all_types, fp, line_res)


def get_word_and_type(line_arr, ind):
    take_word = 1
    arr = line_arr[ind].split('/')
    word, word_type = "NULL", "NULL"

    for y in arr:
        if not any(c.isalpha() for c in y):
            take_word = 0
        break

    if take_word == 1:
        if "'" in arr[0]:
            arr[0] = arr[0].replace("'", "\\'")
            arr[0] = "'" + arr[0] + "'"
        if "$" in arr[0]:
            arr[0] = arr[0].replace("$", "\\S")
            arr[0] = "'" + arr[0] + "'"
        word = arr[0]
        if len(arr) > 1:
            if "$" in arr[1]:
                arr[1] = arr[1].replace("$", "\\S")
                arr[1] = "'" + arr[1] + "'"
            word_type = arr[1]
    return word, word_type


def get_words(num_words, all_words, all_types, fp, line_res):
    fp.write("\n\n")
    for i in range(1, num_words + 1):
        fp.write("@ATTRIBUTE previous_word" + str(i) + " {" + ','.join(all_words) + "}\n"
                 + "@ATTRIBUTE previous_word" + str(i) + "_type {" + ','.join(all_types) + "}\n")

    for i in range(1, num_words + 1):
        fp.write("@ATTRIBUTE next_word" + str(i) + " {" + ','.join(all_words) + "}\n"
                 + "@ATTRIBUTE next_word" + str(i) + "_type {" + ','.join(all_types) + "}\n")

    fp.write("@ATTRIBUTE class {1,2,3,4,5,6}\n\n"
             + "@DATA\n")
    for line in line_res:
        fp.write(line)
    fp.write("\n")
    fp.close()


if __name__ == '__main__':
    start(8)
