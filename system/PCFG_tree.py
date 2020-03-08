import re

#Change digits with #-s
DIGITS = re.compile("[0-9]", re.UNICODE)



class Node:
    def __init__(self, parent, value):
        self.parent = parent
        if self.parent != None:
            self.parent.children.append(self)
        self.value = value
        self.children = []

    def GetChildrenValues(self):
        return list(map(lambda x: x.value, self.children))
    
    def IsWord(self):
        return self.children == []
    
    def Copy(self):
        new_node = Node(self.parent, self.value)
        new_node.children = self.children
        return new_node

class PCFG_Tree:
    def __init__(self, sentence=None, root=None):
        self.grammar = []
        self.lexicon = []
        if sentence != None:
            self.root = Node(None, 'SENT')
            self.nodes = []
            self.leafs = []
            sentence = DIGITS.sub("#", sentence)
            words = sentence.split()
            n = len(words)

            current_node = self.root
            for i in range(2, n):
                new_node = Node(current_node, words[i].replace('(', '').replace(')', ''))
                if words[i][-1] == ')' and i != n-1:
                    self.leafs.append(words[i].replace(')', ''))
                    while words[i][-1] == ')':
                        current_node = current_node.parent
                        words[i] = words[i].replace(')', '', 1)
                else:
                    current_node = new_node
            self.leafs.append(words[-1].replace(')', ''))
        else:
            self.root = root.Copy()
            self.Reconstruct()
    
    def ExtractGrammar(self, node=None):
        if node is None:
            node = self.root
            self.grammar = []
        if len(node.children) > 0:
            self.grammar.append((node.value, ' '.join(node.GetChildrenValues())))
            for child in node.children:
                self.ExtractGrammar(child)
        else:
            self.lexicon.append((node.value, node.parent.value))
                
    def GetLeafs(self):
        self.GetNodesList()
        self.leafs = list(filter(lambda x: x.IsWord(), self.nodes))
        self.nodes = []
        
    def GetNodesList(self, node=None):
        if node is None:
            node = self.root
            self.nodes = [self.root]
        if not node.IsWord():
            self.nodes.extend(node.children)
            for child in node.children:
                self.GetNodesList(child)
            
    #transform the tree into Chomsky normal form
    #no need of START, TERM and DEL steps
    def TransformToCNF(self, node=None):
        if node == None:
            node = self.root
        #UNIT
        if len(node.children) == 1:
            if not node.children[0].IsWord():
                child = node.children[0]
                node.value = node.value + "&" + child.value
                node.children = child.children
                for gchild in child.children:
                    gchild.parent = node
                self.TransformToCNF(node)
        #BIN
        if len(node.children) > 2:
            children = node.children[1:]
            node.children = node.children[:1]
            new_node = Node(node, node.value + '|' + children[0].value)
            new_node.children = children
            for child in children:
                child.parent = new_node
            self.TransformToCNF(new_node)
            
        if len(node.children) == 2:
            for child in node.children:
                self.TransformToCNF(child)
    
    def GetProba(self, lexicon, grammar, node=None):
        if node == None:
            node = self.root
            self.p = 1
            
        if not node.IsWord():
            if node.children[0].IsWord():
                self.p *= lexicon[node.children[0].value][node.value]
            else:
                if node.parent != None:
                    ch = ' '.join(node.GetChildrenValues())
                    self.p *= grammar[node.value][ch]
                for child in node.children:
                    self.GetProba(lexicon, grammar, child)
        return self.p
            
    def Visualize(self, node=None):
        if node == None:
            node = self.root
        if not node.IsWord():
            print(node.value)
            print(node.GetChildrenValues())
            for child in node.children:
                self.Visualize(child)
                
    def inv_trans_unit(self, node=None):
        if node == None:
            node = self.root
            
        if '&' in node.value:
            names = node.value.split('&')
            node.value = names[0]
            for i in reversed(range(1, len(names))):
                new_node = Node(None, names[i])
                new_node.children = node.children
                for child in node.children:
                    child.parent = new_node
                new_node.parent = node
                node.children = [new_node]
            self.inv_trans_unit(node)
        elif not node.IsWord():
            for child in node.children:
                self.inv_trans_unit(child)
                
    def inv_trans_bin(self, node=None):
        if node == None:
            node = self.root
        if '|' in node.value:
            node.parent.children.remove(node)
            node.parent.children.extend(node.children)
            for child in node.children:
                child.parent = node.parent
            if not '|' in node.parent.children[-1].value:
                self.inv_trans_bin(node.parent)
            else:
                self.inv_trans_bin(node.parent.children[-1])
        elif not node.IsWord():
            for child in node.children:
                self.inv_trans_bin(child)
            
    def InvTransformFromCNF(self):
        self.inv_trans_bin()
        self.inv_trans_unit()
        
    def GetSentence(self):
        sentence = '( (' + self.root.value + ' '
        ite = self.root.children[0]
        while True:
            if not ite.IsWord():
                sentence += '(' + ite.value + ' '
                ite = ite.children[0]
            else:
                sentence += ite.value
                while ite == ite.parent.children[-1]:
                    sentence += ')'
                    ite = ite.parent
                    if ite == self.root:
                        break
                if ite == self.root:
                    sentence += ')'
                    break
                else:
                    sentence += ' '
                    idx = ite.parent.children.index(ite)
                    ite = ite.parent.children[idx+1]
        return sentence
            
    def Reconstruct(self, node=None):
        if node == None:
            node = self.root
        
        if len(node.children) > 0:
            if not node.children[0].IsWord():
                children = node.children
                node.children = []
                for child in children:
                    new_child = child.Copy()
                    new_child.parent = node
                    node.children.append(new_child)
                    
                for child in node.children:
                    self.Reconstruct(child)
            else:
                child = node.children[0]
                child.parent = None
                new_child = child.Copy()
                new_child.parent = node
                node.children = [new_child]
                