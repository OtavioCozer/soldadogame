class Mesh(object):
    def __init__(self):
        self.faces = []
        self.vertices = []
        self.texcoords = []
        self.normals = []
        self.material = None



def load(filename, scale=1, swapyz=False) -> Mesh:
    vertices: list[float] = []
    normals: list[float] = []
    texcoords: list[float] = []
    current_mesh: Mesh = Mesh()
    print(f"filename {filename}")
    for line in open(filename, "r"):
        values = line.split()
        if not values:
            continue
        elif values[0] in ('usemtl', 'usemat'):
            pass
        if values[0] == 'v':
            v = [float(x) * scale for x in values[1:4]]
            if swapyz:
                v = v[0], v[2], v[1]
            vertices.append(v)
        elif values[0] == 'vn':
            v = [float(x) for x in values[1:4]]
            if swapyz:
                v = v[0], v[2], v[1]
            normals.append(v)
        elif values[0] == 'vt':
            texcoords.append([float(x) for x in values[1:3]])
        elif values[0] == 'f': # vertex_index/texture_index/normal_index
            vs = []
            ts = []
            ns = []
            for v in values[1:]:
                w = v.split('/')
                vs.append(vertices[int(w[0]) - 1])
                ts.append(texcoords[int(w[1]) - 1])
                ns.append(normals[int(w[2]) - 1])
                current_mesh.vertices += vertices[int(w[0]) - 1]
                current_mesh.texcoords += texcoords[int(w[1]) - 1]
                current_mesh.normals += normals[int(w[2]) - 1]

            current_mesh.faces.append((vs, ts, ns))
    
    return current_mesh
    