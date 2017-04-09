from project_utils import *

def output(matrix):
    """Matrix must be be of shape (10000,6) first column is for user_id, the remaining five are the id's of the recommended items"""
    submission = open('items.csv', 'w')

    for row in matrix:
        submission.write(str(row) + '\n')

    submission.close()
    return