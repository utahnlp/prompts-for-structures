from copy import deepcopy
from collections import defaultdict
import networkx as nx

#class Graph():
#    def __init__(self):
#        pass


def subseq_match(subseq, sent):
    """ Subsequence Match
    """
    l = len(subseq)
    
    for i in range(len(sent)-l+1):
        if sent[i:i+l] == subseq:
            return True, i, i+l
    return False, None, None



def find_optimal_path(paths, gr, num_ques):
    traversed = []
    valid = [1]*num_ques

    for path in paths:
        counter = [0]*num_ques
        for ix in range(1,len(path)):
            lab =  gr.edges[path[ix-1],path[ix]]["label"]
            if lab != "O":
                counter[int(lab)-1] += 1
        if valid == counter:
            return path
    
    return paths[0]




class Graph:
    """ Inspired from https://www.geeksforgeeks.org/shortest-path-for-directed-acyclic-graphs/
    """
    def __init__(self, num_nodes):
        self.V = num_nodes
        self.graph = defaultdict(list)


    def add_edges(self, source, targ, wt, meta):
        self.graph[source].append((targ,wt,meta))
    
    def remove_edge(self, source, targ):
        for ed in self.graph[source[0]]:
            if (ed[0] == targ[0]) and (ed[2]==source[1]):
                self.graph[source[0]].remove(ed)
                break

    def remove_node(self, node):
        self.graph.pop(node[0])
        
        for key in self.graph.keys():
            for ed in self.graph[key]:
                if ed[0] == node[0]:
                    self.graph[key].remove(ed)

    def calc_dist(self, path):
        dist = 0
        for el_ix, el in enumerate(path):
            if el_ix == len(path)-1:
                break
            for ed in  self.graph[el[0]]:
                if (ed[0] == path[el_ix+1][0]) and (ed[2] == path[el_ix][1]):
                    dist += ed[1]
                    break

        return dist


    def top_sort(self, v, visited, stack):
        visited[v] = True
        if v in self.graph.keys():
            for node, weight, meta in self.graph[v]:
                #print(node)
                if visited[node] == False:
                    self.top_sort(node, visited, stack)
        stack.append(v)


    def shortestpath(self,s):
        """ Returns a list of paths. The label info for a labeled
        edge is in the second element of the tuple corresponding to the
        source node
        """
        visited = [False]*self.V
        stack = []

        for i in range(self.V):
            if visited[i] == False:
                self.top_sort(s, visited, stack)
        
        dist = [float("Inf")]*self.V
        #paths = [[]]*self.V
        #paths[0].append(([],0))
        back_mark = [0]*self.V
        dist[s] = 0

        while stack:
            i = stack.pop()

            for node, weight, meta in self.graph[i]:
                if dist[node] > dist[i] + weight:
                    dist[node] = dist[i] + weight
                    back_mark[node] = (i,meta)

        ## Retracing the shortest path 
        s_path = [(self.V-1,{"label":'O'})]
        while True:
            s_path.append(back_mark[s_path[-1][0]]) 
            # This means that a path doesn't exist
            if s_path[-1] == 0:
                return None, 0
            if s_path[-1][0] == s:
                break
        # Will be in reverse order
        s_path.reverse()
        
        return s_path, dist[self.V-1]
    



def yens_ksp(g, source, dest, K=20):
    sh_path_0, sh_dist= g.shortestpath(source)
    sh_paths = [sh_path_0]
    
    b = []

    for k in range(1,K+1):
        for i in range(len(sh_paths[k-1])-2):
            spur_node = sh_paths[k-1][i]
            root_path = sh_paths[k-1][:i]
            
            root_dist = g.calc_dist(root_path)
            
            new_graph = deepcopy(g)

            for path in sh_paths:
                if root_path == path[:i]:
                    new_graph.remove_edge(path[i],path[i+1]) 

            for node in root_path:
                if node[0] != spur_node[0]:
                    new_graph.remove_node(node)
            
            spur_path, spur_dist = new_graph.shortestpath(spur_node[0])
            totalpath = None
            if spur_path != None:
                totalpath = root_path + spur_path
        
            flag = False
            for p in b:
                if totalpath == p[0]:
                    flag = True
                    break

            if (not flag) and (totalpath!=None):
                b.append((totalpath, root_dist+spur_dist))
        
        b.sort(key=lambda y: y[1])
        if len(b) == 0:
            break
        sh_paths.append(b[0][0])
        b = b[1:]
        
    return sh_paths



def construct_graph(sent, generations, inst_ix):
    tokens = sent.split()
    num_nodes = len(tokens) + 1 

    g = Graph(num_nodes)

    null_wt = 0

    ## Adding default edges
    for ix, tok in enumerate(tokens):
        g.add_edges(ix, ix+1, null_wt, {"label":'O'})
    #g.add_edges(len(tokens), len(tokens)+1, null_wt, {"label":'O'})


    # Adding all the argument edges for the beams generated
    for g_ix, gen_q in enumerate(generations):  #Loops over all the questions
        for beam_ix in range(len(gen_q)):       #Loops over all the beams in the question
            # We consider only continuous spans for extraction
            match, start_ix, end_ix = subseq_match(gen_q[beam_ix]["sentence"].split(), tokens)
            # Considering matched subsequences
            #if match:
            #    print(tokens)
            #    print(gen_q[beam_ix]["sentence"].split())
            #    print(start_ix)
            #    print(end_ix)
            #    print()
                
            if match:
                # Adding the argument edges. Note here that the starting node is defined
                # by node with index one less than the token location. This helps in 
                # capturing span overlaps in cases like, e.g., when say a token x is the 
                # last token in a span and the first one in another
                g.add_edges(start_ix, end_ix, gen_q[0]["score"]-gen_q[beam_ix]["score"],{"label":f"{g_ix+1}", "beam_pos":beam_ix})
    #g2 = Graph(g.V)
    #print(g.shortestpath(1))
    #print(g2.shortestpath(1))
    #exit()
    #try:
    sh_paths = yens_ksp(g,0,len(tokens))
    #except IndexError:
        #print("Error")
        #print(inst_ix)
        #print(len(generations))
        #return ["Has a none response. "]*len(generations)

    def find_optimal_path(paths, num_ques):
        traversed = []
        valid = [1]*num_ques

        for path in paths:
            counter = [0]*num_ques
            for ix in range(len(path)):
                lab =  path[ix][1]["label"]
                if lab != "O":
                    counter[int(lab)-1] += 1
            if valid == counter:
                return path
    
        return paths[0]
    
    optimal_path = find_optimal_path(sh_paths,len(generations))
    
    def get_answers(path, num_ques, tokens):
        ans = ["Has a none response ."]*num_ques
        for ix in range(1,len(path)+1):
            lab =  path[ix-1][1]["label"]
            if lab != "O":
                resp = " ".join( tokens[path[ix-1][0]:path[ix][0]])
                if resp != '':
                    ans[int(lab)-1] = resp
        return ans
    
    ans = get_answers(optimal_path,len(generations), tokens)

    return ans


def construct_graph_old(sent, generations):
    gr = nx.DiGraph()
    tokens = sent.split()
    nodes = []
    default_edges = []

    null_wt = 0 # Weight of Null Relations
    #for g_ix, gen_q in enumerate(generations):
    #    max_wt += gen_q[0]["score"]
    #print(max_wt)

    # Adding an additional token. 
    # This helps us account for span edges
    nodes.append(tuple([0,{"text":"<start>"}])) 
    for ix, tok in enumerate(tokens):
        nodes.append(tuple([ix+1,{"text":tok}]))
        default_edges.append((ix,ix+1,{"weight":null_wt,"label":'O'}))  #Adding null edge

    # Adding all the nodes and null relation edges
    gr.add_nodes_from(nodes)
    gr.add_edges_from(default_edges) 
    
    # Adding all the argument edges for the beams generated
    for g_ix, gen_q in enumerate(generations):  #Loops over all the questions
        for beam_ix in range(len(gen_q)):       #Loops over all the beams in the question
            # We consider only continuous spans for extraction
            match, start_ix, end_ix = subseq_match(gen_q[beam_ix]["sentence"].split(), tokens)
            # Considering matched subsequences
            if match:
                # Adding the argument edges. Note here that the starting node is defined
                # by node with index one less than the token location. This helps in 
                # capturing span overlaps in cases like, e.g., when say a token x is the 
                # last token in a span and the first one in another
                try:
                    if gr.edges[start_ix-1,end_ix]:
                        if gr.edges[start_ix-1,end_ix]["beam_pos"] < beam_ix:
                            continue
                except KeyError:
                    pass
                gr.add_edges_from([(start_ix-1, end_ix, {"weight":gen_q[0]["score"]-gen_q[beam_ix]["score"],"label":f"{g_ix+1}", "beam_pos":beam_ix})])
            
            #else:
            #    print(f"Mismatch: {g_ix+1} {beam_ix+1}")

    #print(nx.dag_longest_path(gr))
    #print(list(nx.all_simple_paths(gr,1,32)))
    #sh_path = nx.shortest_path(gr, source=0, target=len(tokens),weight="weight")
    #print(sh_path)
    #exit()
    #for ix in range(1,len(sh_path)):
        #print(gr.edges([(sh_path[ix-1],sh_path[ix])],data="label"))
    #    print(gr.edges[sh_path[ix-1],sh_path[ix]])

    #all_sh_path = list(nx.all_shortest_paths(gr, source=0, target=len(tokens), weight="weight") )
    all_sh_paths = list(nx.shortest_simple_paths(gr, source=0, target=len(tokens), weight="weight"))
    
    optimal_path = find_optimal_path(all_sh_paths, gr, len(generations))
    
    def get_answers(path, gr, num_ques, tokens):
        ans = ["Has a none response ."]*num_ques
        for ix in range(1,len(path)):
            lab =  gr.edges[path[ix-1],path[ix]]["label"]
            if lab != "O":
                resp = " ".join( tokens[path[ix-1]:path[ix]-1])
                if resp != '':
                    ans[int(lab)-1] = resp
        return ans
    
    ans = get_answers(optimal_path,gr,len(generations), tokens)
    #for ix in range(1,len(all_sh_path[path_ix])):
    #    print(gr.edges[all_sh_path[path_ix][ix-1],all_sh_path[path_ix][ix]])
    return ans
    #print()
    #print(sent)
    #print(generations[0][0]["sentence"])
    #print(generations[1][0]["sentence"])
    
   
