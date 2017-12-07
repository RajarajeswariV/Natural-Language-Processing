# coding: utf-8
"""CS585: Assignment 1

This assignment has two parts:

(1) Finite-State Automata

- Implement a function run_fsa that takes in a *deterministic* FSA and
  a list of symbols and returns True if the FSA accepts the string.

- Design a deterministic FSA to represent a toy example of person names.

(2) Parsing
- Read a context-free grammar from a string.

- Implement a function that takes in a parse tree, a sentence, and a
  grammar, and returns True if the parse tree is a valid parse of the
  sentence.

See the TODO comments below to complete your assignment. I strongly
recommend working one function at a time, using a1_test to check that
your implementation is correct.

For example, after implementing the run_fsa function, run `python
a1_test.py TestA1.test_baa_fsa` to make sure your FSA implementation
works for the sheep example. Then, move on to the next function.

"""

#######
# FSA #
#######

def run_fsa(states, initial_state, accept_states, transition, input_symbols):
    """
    Implement a deterministic finite-state automata.
    See test_baa_fsa.

    Params:
      states..........list of ints, one per state
      initial_state...int for the starting state
      accept_states...List of ints for the accept states
      transition......dict of dicts representing the transition function
      input_symbols...list of strings representing the input string
    Returns:
      True if this FSA accepts the input string; False otherwise.
    """
    if len(input_symbols) == 0:
      return False

    present_state = initial_state
    result = False

    for symbol in input_symbols:
      present_state_tx_dict = transition.get(present_state, -1)
      if present_state_tx_dict == -1:
        result = False
        break

      present_state = present_state_tx_dict.get(symbol, -1)
      if(present_state == -1):
        result = False
        break

      if(present_state in accept_states):
        result = True

    return result

def get_name_fsa():
    """
    Define a deterministic finite-state machine to recognize a small set of person names, such as:
      Mr. Frank Michael Lewis
      Ms. Flo Lutz
      Frank Micael Lewis
      Flo Lutz

    See test_name_fsa for examples.

    Names have the following:
    - an optional prefix in the set {'Mr.', 'Ms.'}
    - a required first name in the set {'Frank', 'Flo'}
    - an optional middle name in the set {'Michael', 'Maggie'}
    - a required last name in the set {'Lewis', 'Lutz'}

    Returns:
      A 4-tuple of variables, in this order:
      states..........list of ints, one per state
      initial_state...int for the starting state
      accept_states...List of ints for the accept states
      transition......dict of dicts representing the transition function
    """
    states = [0, 1, 2]
    initial_state = 0
    accept_states = [2]
    transition = {
        0: {'Mr.': 0, 'Ms.': 0, 'Frank' : 1, 'Flo' : 1},
        1: {'Michael': 1, 'Maggie' : 1, 'Lewis' : 2, 'Lutz' : 2}
    }

    return states, initial_state, accept_states, transition
    

###########
# PARSING #
###########

def read_grammar(lines):
    """Read a list of strings representing CFG rules. E.g., the string
    'S :- NP VP'
    should be parsed into a tuple
    ('S', ['NP', 'VP'])

    Note that the first element of the tuple is a string ('S') for the
    left-hand-side of the rule, and the second element is a list of
    strings (['NP', 'VP']) for the right-hand-side of the rule.
    
    See test_read_grammar.

    Params:
      lines...A list of strings, one per rule
    Returns:
      A list of (LHS, RHS) tuples, one per rule.
    """

    rules = []

    for line in lines:
      main_split = line.split(":-")
      line_tuple = []
      line_tuple.append(main_split[0].strip())
      line_tuple.append(main_split[1].split())
      rules.append(tuple(line_tuple))

    return rules

class Tree:
    """A partial implementation of a Tree class to represent a parse tree.
    Each node in the Tree is also a Tree.
    Each Tree has two attributes:
      - label......a string representing the node (e.g., 'S', 'NP', 'dog')
      - children...a (possibly empty) list of children of this
                   node. Each element of this list is another Tree.

    A leaf node is a Tree with an empty list of children.
    """
    def __init__(self, label, children=[]):
        """The constructor.
        Params:
          label......A string representing this node
          children...An optional list of Tree nodes, representing the
                     children of this node.
        This is done for you and should not be modified.
        """
        self.label = label
        self.children = children

    def __str__(self):
        """
        Print a string representation of the tree, for debugging.
        This is done for you and should not be modified.
        """
        s = self.label
        for c in self.children:
            s += ' ( ' + str(c) + ' ) '
        return s

    def __is_leaf(self):
      if len(self.children) == 0:
        return True
      else:
        return False

    def get_leaves(self):
        """
        Returns:
          A list of strings representing the leaves of this tree.
        See test_get_leaves.
        """

        leaves = []
        '''
        Base condition: if Tree is a leaf, then return the Tree label as the list
        of leaves
        '''
        if(self.__is_leaf()):
          leaves.append(self.label)
          return leaves

        for child in self.children:
          [leaves.append(leaf) for leaf in child.get_leaves()]

        return leaves

    def get_productions(self):
        """Returns:
          A list of tuples representing a depth-first traversal of
          this tree.  Each tuple is of the form (LHS, RHS), where LHS
          is a string representing the left-hand-side of the
          production, and RHS is a list of strings representing the
          right-hand-side of the production.

        See test_get_productions.
        """
        productions = []

        # return productions

        if(self.__is_leaf()):
          return productions

        children_array = []
        [children_array.append(child.label) for child in self.children]
        productions.append(tuple((self.label, children_array)))

        for child in self.children:
          [productions.append(child_production) for child_production in child.get_productions()]

        return productions


def is_pos(rule, rules):
    """
    Returns:
      True if this rule is a part-of-speech rule, which is true if none of the
      RHS symbols appear as LHS symbols in other rules.
    E.g., if the grammar is:
    S :- NP VP
    NP :- N
    VP :- V
    N :- dog cat
    V :- run likes

    Then the final two rules are POS rules.
    See test_is_pos.

    This function should be used by the is_valid_production function
    below.
    """
    LHS_list = []
    [LHS_list.append(rule_tuple[0]) for rule_tuple in rules]

    for symbol in rule[1]:
      if symbol in LHS_list:
        return False

    return True


def is_valid_production(production, rules):
    """
    Params:
      production...A (LHS, RHS) tuple representing one production,
                   where LHS is a string and RHS is a list of strings.
      rules........A list of tuples representing the rules of the grammar.

    Returns:
      True if this production is valid according to the rules of the
      grammar; False otherwise.

    See test_is_valid_production.

    This function should be used in the is_valid_tree method below.
    """

    if not is_pos(production, rules) and production in rules:
      return True

    if is_pos(production, rules):
      matched_rules = [rule for rule in rules if production[0] == rule[0]]
      for rule in matched_rules:
        if production[1][0] in rule[1]:
          return True

    return False


    
def is_valid_tree(tree, rules, words):
    """
    Params:
      tree....A Tree object representing a parse tree.
      rules...The list of rules in the grammar.
      words...A list of strings representing the sentence to be parsed.

    Returns:
      True if the tree is a valid parse of this sentence. This requires:
        - every production in the tree is valid (present in the list of rules).
        - the leaf nodes in the tree match the words in the sentence, in order.

    See test_is_valid_tree.
    """
    input_tree_productions = tree.get_productions()

    for production in input_tree_productions:
      if not is_valid_production(production, rules):
        return False

    input_tree_leaves = tree.get_leaves()
    
    '''
    Checks if 'input_tree_leaves' list is same as 'words' list based on the 
    order of elements.
    '''
    if len([False for i,j in zip(input_tree_leaves, words) if i!=j]) == 0:
      return True

    return False

