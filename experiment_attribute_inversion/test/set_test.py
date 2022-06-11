import re

def parse_string_for_set(line: str) -> set:
    return line.split(',')

print(parse_string_for_set('age, race, gender'))