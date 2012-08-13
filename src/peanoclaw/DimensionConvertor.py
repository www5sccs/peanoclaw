
def get_dimension( q ):
    """ This function is used to find out the dimension of problem with the help of the q array
    since the solution field have one element more which contains the number of unknowns per subcell the dimension 
    computed by substracting one from the size of the solution list.
    """
    size = len(q)
    return size - 1

if __name__ == '__main__':
    pass

