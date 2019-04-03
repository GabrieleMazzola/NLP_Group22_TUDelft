import re


def number_replacement(string, replacer):
    line = re.sub(r"\d+", replacer, string)
    return line


def quote_replacement(string, replacer):
    line = re.sub(r'\".+\"', replacer, string)
    line = re.sub(r'“(.+?)”', replacer, line)
    line = re.sub(r"'(.+?)'", replacer, line)
    return line


def hashtag_replacement(string, replacer):
    line = re.sub(r'#[\w]+', replacer, string)
    return line