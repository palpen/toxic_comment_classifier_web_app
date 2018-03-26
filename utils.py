import re
import string


def tokenize(s):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    out = regex.sub(' ', s).split()
    return out
