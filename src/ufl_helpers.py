from firedrake import as_vector

#CHECK THESE SIGNS!
#THESE ARE REALLY PLANAR SPECIFIC- IDEALLY THEY DEPEND ON KHAT VECTOR!!!
def skewgrad(f):
    return as_vector([-f.dx(1), f.dx(0)])

def curl2D(v):
    return v[1].dx(0) - v[0].dx(1)

def rot2D(v):
    return as_vector([-v[1], v[0]])
