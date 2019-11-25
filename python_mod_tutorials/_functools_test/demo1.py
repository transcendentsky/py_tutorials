"""
functools.partial(func, *args, **keywords)

Return a new partial object which when called will behave like func called
with the positional arguments args and keyword arguments keywords.
If more arguments are supplied to the call, they are appended to args.
If additional keyword arguments are supplied, they extend and override keywords.
Roughly equivalent to:

'''
def partial(func, *args, **keywords):
    def newfunc(*fargs, **fkeywords):
        newkeywords = keywords.copy()
        newkeywords.update(fkeywords)
        return func(*args, *fargs, **newkeywords)
    newfunc.func = func
    newfunc.args = args
    newfunc.keywords = keywords
    return newfunc
'''

"""

def main():
    from functools import partial

    basetwo = partial(int, base=2)
    basetwo.__doc__ = "Convert base 2 string to an int"
    print basetwo('10010')
    print basetwo('10010', base=10)

if __name__ == '__main__':
    main()