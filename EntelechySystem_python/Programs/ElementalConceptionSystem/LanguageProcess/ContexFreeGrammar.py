"""
@File   : ContexFreeGrammar.py
@Author : Yee Cube
@Date   : 2022/08/17
@Desc   : 上下文无关语法。来源：aima-python-master。
"""


# ______________________________________________________________________________
# Grammars and Lexicons
from collections import defaultdict


def Rules(**rules):
    """创建一个字典用以映射符号到序列。
    >>> Rules(A = "B C | D E")
    {'A': [['B', 'C'], ['D', 'E']]}
    """
    for (lhs, rhs) in rules.items():
        rules[lhs] = [alt.strip().split() for alt in rhs.split('|')]
    return rules


def Lexicon(**rules):
    """Create a dictionary mapping symbols to alternative words.
    >>> Lexicon(Art = "the | a | an")
    {'Art': ['the', 'a', 'an']}
    """
    for (lhs, rhs) in rules.items():
        rules[lhs] = [word.strip() for word in rhs.split('|')]
    return rules


class Grammar:

    def __init__(self, name, rules, lexicon):
        """A grammar has a set of rules and a lexicon."""
        self.name = name
        self.rules = rules
        self.lexicon = lexicon
        self.categories = defaultdict(list)
        for lhs in lexicon:
            for word in lexicon[lhs]:
                self.categories[word].append(lhs)

    def rewrites_for(self, cat):
        """Return a sequence of possible rhs's that cat can be rewritten as."""
        return self.rules.get(cat, ())

    def isa(self, word, cat):
        """Return True iff word is of category cat"""
        return cat in self.categories[word]

    def __repr__(self):
        return '<Grammar {}>'.format(self.name)

E0 = Grammar('E0',
             Rules(  # Grammar for E_0 [Figure 22.4]
                 S='NP VP | S Conjunction S',
                 NP='Pronoun | Name | Noun | Article Noun | Digit Digit | NP PP | NP RelClause',  # noqa
                 VP='Verb | VP NP | VP Adjective | VP PP | VP Adverb',
                 PP='Preposition NP',
                 RelClause='That VP'),

             Lexicon(  # Lexicon for E_0 [Figure 22.3]
                 动作名词="",
                 名词="stench | breeze | glitter | nothing | wumpus | pit | pits | gold | east",  # noqa
                 动词="is | see | smell | shoot | fell | stinks | go | grab | carry | kill | turn | feel",  # noqa
                 形容词="right | left | east | south | back | smelly",
                 副词="here | there | nearby | ahead | right | left | east | south | back",  # noqa
                 代词="me | you | I | it",
                 专有名词="John | Mary | Boston | Aristotle",
                 冠词="the | a | an",
                 介词="to | in | on | near",
                 连词="and | or | but",
                 数词="0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9",
                 That="that"
             ))

E_ = Grammar('E_',  # Trivial Grammar and lexicon for testing
             Rules(
                 S='NP VP',
                 NP='Art N | Pronoun',
                 VP='V NP'),

             Lexicon(
                 Art='the | a',
                 N='man | woman | table | shoelace | saw',
                 Pronoun='I | you | it',
                 V='saw | liked | feel'
             ))

E_NP_ = Grammar('E_NP_',  # another trivial grammar for testing
                Rules(NP='Adj NP | N'),
                Lexicon(Adj='happy | handsome | hairy',
                        N='man'))
