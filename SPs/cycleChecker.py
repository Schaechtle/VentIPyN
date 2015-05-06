# ToDo: Taken from the internet at certain stage, changed, added some methods
def diagIndeces(n):
    indec=[]
    for i in range(0,n):
        indec.append(i*n +(1+i))
    return indec

def cyclic(dag):
    graph_unsorted = dag2graph_unsorted(dag)
    return topolgical_sort_bool(graph_unsorted)
def dagCheck(bindag,n):
    graph_unsorted = bin2graph_unsorted(bindag,n)
    return topolgical_sort_bool(graph_unsorted)

def computeNegIndex(i,j,n):
    return n*n - ((i-1)*n + j) + 1


def dag2graph_unsorted(dag):
    graph_unsorted = []
    for i in range(len(dag)):
        children =[]
        for j in range(len(dag[0])):
            if dag[i][j]==1:
                children.append(j)
        graph_unsorted.append((i,children))
    return graph_unsorted


def dagBool2graph_unsorted(dag):
    graph_unsorted = []
    for i in range(len(dag)):
        children =[]
        for j in range(len(dag[0])):
            if dag[i][j]:
                children.append(j)
        graph_unsorted.append((i,children))
    return graph_unsorted

        
def bin2graph_unsorted(bindag,n):
    lenbinRep =len(bindag)-2
    graph_unsorted = []
    for i in range(n):
        children =[]
        for j in range(n):
                negindex = computeNegIndex(i+1,j+1,n)
                if negindex<=lenbinRep:
                    if  bindag[-negindex]=="1":
                        children.append(j)
        graph_unsorted.append((i,children))
    return graph_unsorted        
            
def topolgical_sort_bool(graph_unsorted):
    """
    Repeatedly go through all of the nodes in the graph, moving each of
    the nodes that has all its edges resolved, onto a sequence that
    forms our sorted graph. A node has all of its edges resolved and
    can be moved once all the nodes its edges point to, have been moved
    from the unsorted graph onto the sorted one.
    """

    # This is the list we'll return, that stores each node/edges pair
    # in topological order.
    graph_sorted = []

    # Convert the unsorted graph into a hash table. This gives us
    # constant-time lookup for checking if edges are unresolved, and
    # for removing nodes from the unsorted graph.
    graph_unsorted = dict(graph_unsorted)

    # Run until the unsorted graph is empty.
    while graph_unsorted:

        # Go through each of the node/edges pairs in the unsorted
        # graph. If a set of edges doesn't contain any nodes that
        # haven't been resolved, that is, that are still in the
        # unsorted graph, remove the pair from the unsorted graph,
        # and append it to the sorted graph. Note here that by using
        # using the items() method for iterating, a copy of the
        # unsorted graph is used, allowing us to modify the unsorted
        # graph as we move through it. We also keep a flag for
        # checking that that graph is acyclic, which is true if any
        # nodes are resolved during each pass through the graph. If
        # not, we need to bail out as the graph therefore can't be
        # sorted.
        acyclic = False
        for node, edges in graph_unsorted.items():
            for edge in edges:
                if edge in graph_unsorted:
                    break
            else:
                acyclic = True
                del graph_unsorted[node]
                graph_sorted.append((node, edges))

        if not acyclic:
            # Uh oh, we've passed through all the unsorted nodes and
            # weren't able to resolve any of them, which means there
            # are nodes with cyclic edges that will never be resolved,
            # so we bail out with an error.
            return True

    return False
def topolgical_sort(graph_unsorted):
    """
    Repeatedly go through all of the nodes in the graph, moving each of
    the nodes that has all its edges resolved, onto a sequence that
    forms our sorted graph. A node has all of its edges resolved and
    can be moved once all the nodes its edges point to, have been moved
    from the unsorted graph onto the sorted one.
    """

    # This is the list we'll return, that stores each node/edges pair
    # in topological order.
    graph_sorted = []

    # Convert the unsorted graph into a hash table. This gives us
    # constant-time lookup for checking if edges are unresolved, and
    # for removing nodes from the unsorted graph.
    graph_unsorted = dict(graph_unsorted)

    # Run until the unsorted graph is empty.
    while graph_unsorted:

        # Go through each of the node/edges pairs in the unsorted
        # graph. If a set of edges doesn't contain any nodes that
        # haven't been resolved, that is, that are still in the
        # unsorted graph, remove the pair from the unsorted graph,
        # and append it to the sorted graph. Note here that by using
        # using the items() method for iterating, a copy of the
        # unsorted graph is used, allowing us to modify the unsorted
        # graph as we move through it. We also keep a flag for
        # checking that that graph is acyclic, which is true if any
        # nodes are resolved during each pass through the graph. If
        # not, we need to bail out as the graph therefore can't be
        # sorted.
        acyclic = False
        for node, edges in graph_unsorted.items():
            for edge in edges:
                if edge in graph_unsorted:
                    break
            else:
                acyclic = True
                del graph_unsorted[node]
                graph_sorted.append((node, edges))

        if not acyclic:
            # Uh oh, we've passed through all the unsorted nodes and
            # weren't able to resolve any of them, which means there
            # are nodes with cyclic edges that will never be resolved,
            # so we bail out with an error.
            raise RuntimeError("A cyclic dependency occurred")

    return graph_sorted

