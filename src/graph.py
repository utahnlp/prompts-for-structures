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
    start_ix = []
    end_ix = []
    m_flag = False
    for i in range(len(sent)-l+1):
        if sent[i:i+l] == subseq:
            m_flag = True 
            start_ix.append(i)
            end_ix.append(i+l)
    return m_flag, start_ix, end_ix



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
        #for p in b:
        #    print(p)
        #print("###############")
        if len(b) == 0:
            break
        sh_paths.append(b[0][0])
        b = b[1:]
        
    return sh_paths



def construct_graph(sent, generations, inst_ix, gold_ans, sanity=False, ans_span=[]):
    if type(sent) != list:
        tokens = sent.split()
    else:
        tokens = sent
    num_nodes = len(tokens) + 2 #+1
    
    g = Graph(num_nodes)

    null_wt = 0

    ## Adding default edges
    for ix, tok in enumerate(tokens):
        g.add_edges(ix, ix+1, null_wt, {"label":'O'})
    g.add_edges(len(tokens), len(tokens)+1, null_wt, {"label":'O'})

    gold_invalid = 0
    gold_edge_meta = {}
    for a_ix, ans in enumerate(gold_ans):
        #print(ans)
        #print(tokens)
        #print(ans_span)
        possb_ans = ans.split(" ### ")
        match_flag = False
        for opt_ix, opt in enumerate(possb_ans):
            if ans_span != []:
                match = True
                print('ok')
                print(ans_span)
                print(a_ix)
                print(opt_ix)
                print(possb_ans)
                s_ix = [ans_span[a_ix][opt_ix][0]]
                e_ix = [ans_span[a_ix][opt_ix][1]]
            else:
                match, s_ix , e_ix = subseq_match(opt.split(),tokens)
            #print(opt)
            #print(match)
            #print(s_ix)
            #print(e_ix)
            #print()
            if match:
                match_flag = True
                if sanity:
                    for match_ix in range(len(s_ix)): 
                        g.add_edges(s_ix[match_ix], e_ix[match_ix], 0, {"label":f"{a_ix+1}", "beam_pos":"gold"})
                        gold_edge_meta[a_ix+1] = [s_ix[match_ix], e_ix[match_ix]]
        
        if not match_flag:
            gold_invalid += 1
    
    # Adding all the argument edges for the beams generated
    for g_ix, gen_q in enumerate(generations):  #Loops over all the questions
        for beam_ix in range(len(gen_q)):       #Loops over all the beams in the question
            # We consider only continuous spans for extraction
            match, s_ix, e_ix = subseq_match(gen_q[beam_ix]["sentence"].split(), tokens)
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
                for match_ix in range(len(s_ix)):
                    start_ix = s_ix[match_ix]
                    end_ix = e_ix[match_ix]
                    s_add = 0
                    if sanity:  # We increase the overall score by 1 if we want to sanity check with gold answers as highest
                        s_add = 1
                        if g_ix+1 in gold_edge_meta.keys():
                            if gold_edge_meta[g_ix+1] == [start_ix,end_ix]:
                                continue
                    g.add_edges(start_ix, end_ix, gen_q[0]["score"]-gen_q[beam_ix]["score"]+s_add,{"label":f"{g_ix+1}", "beam_pos":beam_ix})
    try:
        sh_paths = yens_ksp(g,0,len(tokens), K=20)
    except IndexError:
        print(tokens)
        print(ans_span)
        print(gold_ans)
        print(g.graph)
        exit()
    #except IndexError:
        #print("Error")
        #print(inst_ix)
        #print(len(generations))
        #return ["Has a none response. "]*len(generations)

    def find_optimal_path(paths, num_ques):
        traversed = []
        valid = [1]*num_ques
        best_partial_path = paths[0]
        best_partial_cnt = 0

        for path in paths:
            counter = [0]*num_ques
            for ix in range(len(path)):
                lab =  path[ix][1]["label"]
                if lab != "O":
                    counter[int(lab)-1] += 1
            if valid == counter:
                return path
            else:
                single_check=True
                for el in counter:
                    if el not in [0,1]:
                        single_check= False
                        break
                if single_check:
                    if sum(counter) > best_partial_cnt:
                        best_partial_cnt = sum(counter)
                        best_partial_path = path 
        #print("Here") 
        return best_partial_path
    
    #for path in sh_paths:
    #    n = []
    #    for p in path:
    #        if p[1]['label']!= 'O':
    #            n.append(p)
    #    print(path)
    #    print(n)
    #    print()
    #exit()

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

    # Checks for valid answers not matching the sanity checks
    if ans != gold_ans and (gold_invalid==0):
        for a_ix, a in enumerate(ans):
            g_ans = gold_ans[a_ix].split(" ### ")
            if (a not in g_ans) and (a == ''):
                print(inst_ix)
                print(sent)
                print(ans)
                print(gold_ans)
                print()
                break
        

    
    return ans, gold_invalid


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
    





def get_all_cliques(relations, max_vertices):
    """ Get all cliques from pairwise connections. This function can be used to 
    obtain cliques based on pairs of connections. Useful for applications like
    coref resolution.

    Inputs
    --------------------
    relations - List[List[int]]. List of connections in the form of tuple
    max_vertices - int. Number of verices in the graph.
    """
    gr = nx.Graph()
    nodes = list(range(max_vertices))
    gr.add_nodes_from(nodes)
    gr.add_edges_from(relations)

    clusters = []

    violations = 0

    max_clique = list(nx.algorithms.approximation.max_clique(gr)) 
    clusters.append(max_clique)

    prev_rels = relations
    
    # Compute cliques for rest of the graph
    while True:
        # Remove already clustered nodes
        for node in clusters[-1]:
            nodes.remove(node)
        
        if len(nodes)==0:
            break

        new_rels = []
        for rel in prev_rels:
            if ((rel[0] in clusters[-1]) and (rel[1] in nodes)) or ((rel[1] in clusters[-1]) and (rel[0] in nodes)):
                violations += 1
            elif (rel[0] in nodes) and (rel[1] in nodes):
                new_rels.append(rel)
        prev_rels = new_rels
        

        sub_gr = nx.Graph()
        sub_gr.add_nodes_from(nodes)
        sub_gr.add_edges_from(new_rels)

        m_clique = list(nx.algorithms.approximation.max_clique(sub_gr)) 
        clusters.append(m_clique)
        
    return clusters, violations


    
    

