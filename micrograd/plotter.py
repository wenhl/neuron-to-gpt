import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from graphviz import Digraph
from io import BytesIO
#import pickle

def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

#def draw_dot(root, format='svg', rankdir='LR', filename='output'):
def draw_dot(root, format='svg', rankdir='LR'):

    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) 
    
    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
        
    #dot.render(filename, view=True)
    # Convert dot object to a PNG image
    png_data = dot.pipe(format='png')
    sio = BytesIO(png_data)
    img = mpimg.imread(sio)
    plt.imshow(img)
    plt.axis('off')
    plt.show()



    