import re


def number_replacement(string, replacer):
    line = re.sub(r"\d+", replacer, string)
    return line