# RCAIDE/Library/Methods/Geometry/Mesh.py
# 
# Created:  April 2026, Antigravity 

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
import RNUMPY as rp
from copy import deepcopy

# ----------------------------------------------------------------------------------------------------------------------
#  Mesh Class
# ----------------------------------------------------------------------------------------------------------------------
class Mesh:
    """A lightweight mesh class to replace trimesh dependency in RCAIDE.
    
    Provides basic mesh operations and mass property calculations.
    """
    
    def __init__(self, vertices=None, faces=None):
        self._vertices = rp.array(vertices) if vertices is not None else rp.zeros((0, 3))
        self._faces    = rp.array(faces) if faces is not None else rp.zeros((0, 3), dtype=int)
        self.density  = 1.0
        self._cached_props = {}

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, value):
        self._vertices = rp.array(value)
        self._clear_cache()

    @property
    def faces(self):
        return self._faces

    @faces.setter
    def faces(self, value):
        self._faces = rp.array(value)
        self._clear_cache()

    def _clear_cache(self):
        self._cached_props = {}

    def _compute_mass_properties(self):
        if 'volume' in self._cached_props:
            return
        
        if len(self.faces) == 0:
            self._cached_props['volume'] = 0.0
            self._cached_props['centroid'] = rp.zeros(3)
            self._cached_props['moment_inertia'] = rp.zeros((3, 3))
            return

        # Get vertices for each face
        P1 = self._vertices[self._faces[:, 0]]
        P2 = self._vertices[self._faces[:, 1]]
        P3 = self._vertices[self._faces[:, 2]]

        # Compute face volumes (signed tetrahedra volumes from origin)
        V_f    = rp.sum(P1 * rp.cross(P2, P3), axis=1) / 6.0
        volume = rp.sum(V_f)
        
        if abs(volume) < 1e-18:
            self._cached_props['volume'] = 0.0
            self._cached_props['centroid'] = rp.zeros(3)
            self._cached_props['moment_inertia'] = rp.zeros((3, 3))
            return

        # Compute center of mass (centroid)
        S        = P1 + P2 + P3
        centroid = rp.sum(V_f[:, None] * (S / 4.0), axis=0) / volume

        # Compute inertia integrals at origin
        # Note: These are based on the standard formulas for polyhedral mass properties
        Jxx_int = rp.sum(V_f / 20.0 * (P1[:, 0]**2 + P2[:, 0]**2 + P3[:, 0]**2 + S[:, 0]**2))
        Jyy_int = rp.sum(V_f / 20.0 * (P1[:, 1]**2 + P2[:, 1]**2 + P3[:, 1]**2 + S[:, 1]**2))
        Jzz_int = rp.sum(V_f / 20.0 * (P1[:, 2]**2 + P2[:, 2]**2 + P3[:, 2]**2 + S[:, 2]**2))
        Jxy_int = rp.sum(V_f / 20.0 * (P1[:, 0]*P1[:, 1] + P2[:, 0]*P2[:, 1] + P3[:, 0]*P3[:, 1] + S[:, 0]*S[:, 1]))
        Jxz_int = rp.sum(V_f / 20.0 * (P1[:, 0]*P1[:, 2] + P2[:, 0]*P2[:, 2] + P3[:, 0]*P3[:, 2] + S[:, 0]*S[:, 2]))
        Jyz_int = rp.sum(V_f / 20.0 * (P1[:, 1]*P1[:, 2] + P2[:, 1]*P2[:, 2] + P3[:, 1]*P3[:, 2] + S[:, 1]*S[:, 2]))

        # Moments of inertia tensor at origin
        rho      = self.density
        mass     = volume * rho
        I_origin = rp.array([
            [Jyy_int + Jzz_int, -Jxy_int,          -Jxz_int],
            [-Jxy_int,          Jxx_int + Jzz_int, -Jyz_int],
            [-Jxz_int,          -Jyz_int,          Jxx_int + Jyy_int]
        ]) * rho

        # Convert to centroid (reverse parallel axis theorem)
        c          = centroid
        I_centroid = I_origin - mass * (rp.dot(c, c) * rp.identity(3) - rp.outer(c, c))

        self._cached_props['volume'] = volume
        self._cached_props['centroid'] = centroid
        self._cached_props['moment_inertia'] = I_centroid

    @property
    def volume(self):
        self._compute_mass_properties()
        return self._cached_props['volume']

    @property
    def centroid(self):
        self._compute_mass_properties()
        return self._cached_props['centroid']

    @property
    def center_mass(self):
        return self.centroid

    @property
    def moment_inertia(self):
        self._compute_mass_properties()
        return self._cached_props['moment_inertia']

    def apply_transform(self, matrix):
        """Applies a 4x4 homogenous transformation matrix to the mesh."""
        # Ensure type compatibility for matrix multiplication (float vs double)
        pts_h = rp.column_stack([self.vertices, rp.ones(len(self.vertices))])
        matrix = rp.array(matrix, dtype=pts_h.dtype)
        # Transform
        new_pts_h = (matrix @ pts_h.T).T
        # Update vertices
        self._vertices = new_pts_h[:, :3]
        self._clear_cache()
        return self

    @staticmethod
    def concatenate(meshes):
        """Concatenates multiple Mesh objects into a single Mesh."""
        if not meshes:
            return Mesh()
        
        all_vertices = []
        all_faces    = []
        vertex_offset = 0
        
        for m in meshes:
            all_vertices.append(m.vertices)
            all_faces.append(m.faces + vertex_offset)
            vertex_offset += len(m.vertices)
            
        new_v = rp.vstack(all_vertices)
        new_f = rp.vstack(all_faces)
        
        return Mesh(new_v, new_f)

    @staticmethod
    def translation_matrix(vector):
        T = rp.eye(4)
        T[0:3, 3] = vector
        return T

    @staticmethod
    def rotation_matrix(angle, axis, point=None):
        axis = rp.array(axis) / rp.linalg.norm(axis)
        v = rp.scipy.spatial.transform.Rotation.from_rotvec(angle * axis)
        R = rp.eye(4)
        R[0:3, 0:3] = v.as_matrix()
        if point is not None:
            point = rp.array(point)
            # T(p) @ R @ T(-p)
            T1 = Mesh.translation_matrix(-point)
            T2 = Mesh.translation_matrix(point)
            R = T2 @ R @ T1
        return R

def get_convex_hull(points):
    """Creates a Mesh object representing the convex hull of the given points."""
    hull     = rp.scipy.spatial.ConvexHull(points)
    vertices = hull.points
    faces    = hull.simplices
    return Mesh(vertices, faces)

def extrude_polygon(x, y, height):
    """Extrudes a 2D polygon into a 3D Mesh by computing the convex hull of the bottom and top points."""
    # This assumes the polygon is convex, which is true for most aircraft tank sections.
    pts_bottom = rp.column_stack([x, y, rp.zeros(len(x))])
    pts_top    = rp.column_stack([x, y, rp.ones(len(x)) * height])
    all_pts    = rp.vstack([pts_bottom, pts_top])
    return get_convex_hull(all_pts)

def get_polygon_area(x, y):
    """Computes the signed area of a 2D polygon using the Shoelace formula."""
    x = rp.array(x)
    y = rp.array(y)
    # Ensure closed
    if not rp.allclose(x[0], x[-1]) or not rp.allclose(y[0], y[-1]):
        x = rp.concatenate([x, [x[0]]])
        y = rp.concatenate([y, [y[0]]])
    return rp.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) / 2.0

def get_polygon_centroid(x, y):
    """Computes the centroid of a 2D polygon."""
    x = rp.array(x)
    y = rp.array(y)
    # Ensure closed
    if not rp.allclose(x[0], x[-1]) or not rp.allclose(y[0], y[-1]):
        x = rp.concatenate([x, [x[0]]])
        y = rp.concatenate([y, [y[0]]])
    area = get_polygon_area(x, y)
    if abs(area) < 1e-18:
        return rp.array([rp.mean(x[:-1]), rp.mean(y[:-1])])
    
    # Standard formula for centroid of a non-self-intersecting polygon
    cross = (x[:-1] * y[1:] - x[1:] * y[:-1])
    cx = rp.sum((x[:-1] + x[1:]) * cross) / (6.0 * area)
    cy = rp.sum((y[:-1] + y[1:]) * cross) / (6.0 * area)
    return rp.array([cx, cy])

def clip_polygon_x(x, y, x_min=-1e9, x_max=1e9):
    """Clips a 2D polygon to an x-range [x_min, x_max] using Sutherland-Hodgman."""
    pts = rp.column_stack([x, y])
    # Close if not
    if not rp.allclose(pts[0], pts[-1]):
        pts = rp.vstack([pts, pts[0]])

    def clip(current_pts, x_val, left=True):
        res = []
        for i in range(len(current_pts)-1):
            p1 = current_pts[i]
            p2 = current_pts[i+1]
            in1 = (p1[0] <= x_val) if left else (p1[0] >= x_val)
            in2 = (p2[0] <= x_val) if left else (p2[0] >= x_val)
            if in1:
                if in2:
                    res.append(p2)
                else:
                    # Intersection
                    denom = (p2[0] - p1[0])
                    if abs(denom) < 1e-18:
                         res.append(rp.array([x_val, p1[1]]))
                    else:
                        t = (x_val - p1[0]) / denom
                        res.append(rp.array([x_val, p1[1] + t * (p2[1] - p1[1])]))
            elif in2:
                # Intersection
                denom = (p2[0] - p1[0])
                if abs(denom) < 1e-18:
                     res.append(rp.array([x_val, p1[1]]))
                else:
                    t = (x_val - p1[0]) / denom
                    res.append(rp.array([x_val, p1[1] + t * (p2[1] - p1[1])]))
                res.append(p2)
        if len(res) > 0 and not rp.allclose(res[0], res[-1]):
            res.append(res[0])
        return rp.array(res)

    if x_min > -1e8:
        pts = clip(pts, x_min, left=False)
    if len(pts) > 0 and x_max < 1e8:
        pts = clip(pts, x_max, left=True)
    
    if len(pts) == 0:
        return rp.array([]), rp.array([])
    return pts[:, 0], pts[:, 1]

def point_in_polygon(px, py, x, y):
    """Ray-casting algorithm to determine if a point is inside a polygon."""
    x = rp.array(x)
    y = rp.array(y)
    if not rp.allclose(x[0], x[-1]):
        x = rp.concatenate([x, [x[0]]])
        y = rp.concatenate([y, [y[0]]])
    
    inside = False
    for i in range(len(x)-1):
        if ((y[i] > py) != (y[i+1] > py)) and (px < (x[i+1] - x[i]) * (py - y[i]) / (y[i+1] - y[i]) + x[i]):
            inside = not inside
    return inside

def point_to_polygon_distance(px, py, x, y):
    """Computes minimum distance from point to polygon boundary."""
    x = rp.array(x)
    y = rp.array(y)
    # Ensure closed
    if not rp.allclose(x[0], x[-1]):
        x = rp.concatenate([x, [x[0]]])
        y = rp.concatenate([y, [y[0]]])
        
    dist_sq = []
    for i in range(len(x)-1):
        x1, y1 = x[i], y[i]
        x2, y2 = x[i+1], y[i+1]
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            dist_sq.append((px - x1)**2 + (py - y1)**2)
            continue
        t = ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)
        t = rp.clip(t, 0, 1)
        dist_sq.append((px - (x1 + t * dx))**2 + (py - (y1 + t * dy))**2)
    return rp.sqrt(rp.min(rp.array(dist_sq)))

def intersect_convex_polygons(x1, y1, x2, y2):
    """Intersects two convex polygons using Sutherland-Hodgman."""
    pts1 = rp.column_stack([x1, y1])
    if not rp.allclose(pts1[0], pts1[-1]):
        pts1 = rp.vstack([pts1, pts1[0]])
    
    pts2 = rp.column_stack([x2, y2])
    if not rp.allclose(pts2[0], pts2[-1]):
        pts2 = rp.vstack([pts2, pts2[0]])

    # Ensure CCW
    if get_polygon_area(pts2[:, 0], pts2[:, 1]) < 0:
        pts2 = pts2[::-1]
    
    def clip_edge(current_pts, a, b):
        res = []
        for i in range(len(current_pts)-1):
            p1 = current_pts[i]
            p2 = current_pts[i+1]
            
            # Cross product to check if p is to the left of ab
            def is_left(p):
                return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-12

            in1 = is_left(p1)
            in2 = is_left(p2)
            
            if in1:
                if in2:
                    res.append(p2)
                else:
                    # Intersection
                    # (p1.x + t*v.x - a.x)*w.y - (p1.y + t*v.y - a.y)*w.x = 0
                    # where v = p2-p1, w = b-a
                    v = p2 - p1
                    w = b - a
                    denom = (v[0]*w[1] - v[1]*w[0])
                    if abs(denom) > 1e-18:
                        t = ((a[0]-p1[0])*w[1] - (a[1]-p1[1])*w[0]) / denom
                        res.append(p1 + t * v)
            elif in2:
                # Intersection
                v = p2 - p1
                w = b - a
                denom = (v[0]*w[1] - v[1]*w[0])
                if abs(denom) > 1e-18:
                    t = ((a[0]-p1[0])*w[1] - (a[1]-p1[1])*w[0]) / denom
                    res.append(p1 + t * v)
                res.append(p2)
        if len(res) > 0 and not rp.allclose(res[0], res[-1]):
            res.append(res[0])
        return rp.array(res)

    result = pts1
    for i in range(len(pts2)-1):
        if len(result) < 3: break
        result = clip_edge(result, pts2[i], pts2[i+1])
        
    if len(result) == 0:
        return rp.array([]), rp.array([])
    return result[:, 0], result[:, 1]
