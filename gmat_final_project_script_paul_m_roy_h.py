# Author: Paul Molina, Erik Roy Hamlin
# Date: April 2018
# Version: 1.0
#
# This script was created in order to find a correlation between the semantics of a general GMAT
# question and the outcome of its associated answer. This is done by implementing association analysis
# on preprocessed data.

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules



# Creates n-grams from a given input string. The number of each gram created from each string is determined
# by the numOfGrams paramater.
def createGrams(inputString, numOfGrams):
    inputString = inputString.split(' ')
    grams = []
    for i in range(len(inputString)- numOfGrams + 1 ):
        grams.append(inputString[i : i + numOfGrams])
    return grams


# Checks for the association of certain answers or tokens in a set of questions split into n-grams
def checkForQuestionAnswerAssociation(numOfGrams):
    # Holds questions in token form
    question_bank = []
    question_line = []
    question_line_parsed = []

    # Opens file and reads in data
    with open("Questions.txt") as f:
        for line in f:

            # Each line is parsed individually
            question_line_parsed = []

            question_line = (createGrams( (line.replace('\n', '')).replace('\r', '') , numOfGrams))
            new_string = ""
            for gram in question_line:
                # Makes sure that only valid tokens are used
                if gram.__len__() == numOfGrams and not '' in gram:
                    for a_string_value in gram:
                        new_string += a_string_value + " "
                    question_line_parsed.append(new_string.strip())
                    new_string = ""

            question_bank.append(question_line_parsed)

    # Will be used to append the correct answer to each tuple
    bank_counter = 0

    # This loop appends each answer to the end of each processed question. As every answer is represented
    # by a single character, they are unique to each tuple in the dataset
    with open("Answers") as f:
        for line in f:
            question_bank[bank_counter].append( (line.replace('\n', '')).replace('\r', '') )
            bank_counter += 1

    # TransactionEncoder changes all values to 0/1's for final processing using the apriori algorithm
    oht = TransactionEncoder()
    oht_array = oht.fit(question_bank).transform(question_bank)
    df = pd.DataFrame(oht_array, columns=oht.columns_)


    # Apriopri algorithm usd with a minimum support of 0.02
    frequent_itemsets = apriori(df, min_support=0.02, use_colnames=True)
    # Length is added for convenience though not required
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    # Confidence will be the metric used to check for association. It will be set at 2
    # to check for any associations with at least 20% association
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)

    rules["antecedant_len"] = rules["antecedants"].apply(lambda x: len(x))

    print(rules[(rules['antecedant_len'] == 1)  & (rules['confidence'] > 0.2)])

checkForQuestionAnswerAssociation(numOfGrams=4)
