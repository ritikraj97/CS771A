import numpy as np

# You are not allowed to import any libraries other than numpy

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SKLEARN, SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF PROHIBITED LIBRARIES WILL RESULT IN PENALTIES

# DO NOT CHANGE THE NAME OF THE METHOD my_fit BELOW
# IT WILL BE INVOKED BY THE EVALUATION SCRIPT
# CHANGING THE NAME WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, classes to create the Tree, Nodes etc

################################
# Non Editable Region Starting #
################################
def my_fit(words, verbose=False):
################################
#  Non Editable Region Ending  #
################################

    global all_words
    all_words = words
    model = DecisionTree(min_leaf_size=1, max_depth=15)
    model.fit(verbose)

    # Use this method to train your decision tree model using the word list provided
    # Return the trained model as is -- do not compress it using pickle etc
    # Model packing or compression will cause evaluation failure
    
    return model                    # Return the trained model


all_words = []


class DecisionTree:
    def __init__(self, min_leaf_size, max_depth):
        self.root = None
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        
    def fit(self, verbose):
        global all_words
        self.root = Node(depth=0, parent=None)
        if verbose:
            print("root")
            print("└───", end='')
        # The root is trained with all the words
        self.root.fit(my_words_idx=np.arange(len(all_words)), min_leaf_size=self.min_leaf_size, max_depth=self.max_depth, verbose=verbose)


class Node:
    # A node stores its own depth (root = depth 0), a link to its parent
    # A dictionary is used to store the children of a non-leaf node.
    # Each child is paired with the response that selects that child.
    # A node also stores the query-response history that led to that node
    # Note: my_words_idx only stores indices and not the words themselves
    def __init__(self, depth, parent):
        self.depth = depth
        self.parent = parent
        self.my_words_idx = None
        self.children = {}
        self.is_leaf = True
        self.query_idx = None
    
    # Each node must implement a get_query method that generates the
    # query that gets asked when we reach that node. Note that leaf nodes
    # also generate a query which is usually the final answer
    def get_query(self):
        return self.query_idx
    
    # Each non-leaf node must implement a get_child method that takes a
    # response and selects one of the children based on that response
    def get_child(self, response):
        # This case should not arise if things are working properly
        # Cannot return a child if I am a leaf so return myself as a default action
        if self.is_leaf:
            print("Why is a leaf node being asked to produce a child? Melbot should look into this!")
            child = self
        else:
            # This should ideally not happen. The node should ensure that all possibilities
            # are covered, e.g. by having a catch-all response. Fix the model if this happens
            # For now, hack things by modifying the response to one that exists in the dictionary
            if response not in self.children:
                print(f"Unknown response {response} -- need to fix the model")
                response = list(self.children.keys())[0]
            child = self.children[response]
        return child

    # Process to be executed if a leaf is encountered
    def process_leaf(self, my_words_idx):
        return my_words_idx[0]

    def reveal(self, word, query):
        # Find out the intersection between the query and the word
        mask = [*('_' * len(word))]
        for i in range(min(len(word), len(query))):
            if word[i] == query[i]:
                mask[i] = word[i]
        return ' '.join(mask)
        # Returns "_ _ _ _ _ _" or any alphabet that fits
    
    def get_entropy(self, counts):
        # Don't include elements that don't appear at all
        assert np.min(counts) > 0, "Elements with no positive coint detected"
        num = counts.sum()
        # Single element = 0 entropy
        if num <= 1:
            print(f"Warning: {num} elements in total")
            return 0
        proportions = counts / num
        return np.sum(proportions * np.log2(counts))
    
    def get_mask(self, str, word):
        mask = []
        for letter in word:
            if letter in str:
                mask.append(letter)
            else:
                mask.append('_')
        return '_'.join(mask)
    
    def try_attr(self, str, words):
        # Calculate the entropy and the splitted dictionary
        split_dict = {}
        count_dict = {}
        for word in words:
            mask = self.get_mask(str, word)
            if mask not in split_dict:
                split_dict[mask] = []
                count_dict[mask] = 0
            split_dict[mask].append(word)
            count_dict[mask] += 1
        entropy = self.get_entropy(np.array(list(count_dict.values())))
        return (entropy, split_dict)

    def get_split_dict(self, my_words_idx, query):
        """
        Splits a list of word indices based on a query mask and returns a dictionary of splits and their counts.

        Args:
            my_words_idx (list): A list of word indices to be split.
            query (str): A query mask used to split the indices.

        Returns:
            tuple: A tuple containing two dictionaries - split_dict and count_dict.
                   split_dict contains the splits as keys and the corresponding indices as values.
                   count_dict contains the splits as keys and their counts as values.
        """
        split_dict = {}
        count_dict = {}
        for idx in my_words_idx:
            mask = self.reveal(all_words[idx], query)
            if mask not in split_dict:
                split_dict[mask] = []
                count_dict[mask] = 0
            split_dict[mask].append(idx)
            count_dict[mask] += 1
        return (split_dict, count_dict)

    # Process to be executed if a node is encountered
    def process_node(self, my_words_idx, verbose):
        # For the root, we do not ask any query -- Melbot simply gives us the length of the secret word
        global all_words
        best_split_dict = {}
        if self.depth == 0:
            query_idx = -1
            query = ""
            (best_split_dict, count_dict) = self.get_split_dict(my_words_idx, query)
        else:
            size = len(my_words_idx)
            new_words_idx = my_words_idx
            if size >= 100:
                size = 100
                # randomly selecting elements from the new_words_idx list
                new_words_idx = np.random.choice(my_words_idx, size)
            (best_split_dict, count_dict) = self.get_split_dict(my_words_idx, all_words[new_words_idx[0]])

            min_entropy = self.get_entropy(np.array(list(count_dict.values())))
            query_idx = new_words_idx[0]
            for i in new_words_idx:
                split_dict, count_dict = self.get_split_dict(my_words_idx, all_words[i])
                entropy = self.get_entropy(np.array(list(count_dict.values())))
                if (entropy < min_entropy):
                    min_entropy = entropy
                    query_idx = i
                    best_split_dict = split_dict
        return query_idx, best_split_dict

    def fit(self, my_words_idx, min_leaf_size, max_depth, fmt_str="    ", verbose=False):
        global all_words
        self.my_words_idx = my_words_idx
        # If the node is too smallor too deep, make it a leaf
        # In general, can also include purity considerations into account
        if len(my_words_idx)<=min_leaf_size or self.depth>=max_depth:
            self.is_leaf = True
            self.query_idx = self.process_leaf(self.my_words_idx)
            if verbose:
                print(' ')
        else:
            self.is_leaf = False
            (self.query_idx, split_dict) = self.process_node(self.my_words_idx, verbose)
            if verbose:
                print(all_words[self.query_idx])
            for (i, (response, split)) in enumerate(split_dict.items()):
                if verbose:
                    if i == len(split_dict) - 1:
                        print(fmt_str+"└───", end='')
                        fmt_str += "    "
                    else:
                        print(fmt_str+"├───", end='')
                        fmt_str += "|   "
                # For every split, create a new child
                self.children[response] = Node(depth=self.depth+1, parent=self)
                # Recursively train this child node
                self.children[response].fit(split, min_leaf_size, max_depth, fmt_str, verbose)
