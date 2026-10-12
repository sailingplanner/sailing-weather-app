"""
Sailplan - TSS-module (verkeersscheidingsstelsels, COLREG regel 10).

Doel: een routelijn zo aanpassen dat een recreatievaartuig een TSS niet
"in de lengte" bevaart maar loodrecht kruist (of er omheen blijft).

Opbouw
------
1. Data      : echte geometrie uit OpenStreetMap/OpenSeaMap (Overpass API).
               Valt dat weg, dan blijft de route ONGEWIJZIGD en wordt alleen
               een indicatieve waarschuwing getoond (zie indicative_hits).
2. Schemes   : build_schemes() maakt van de ruwe OSM-objecten voetafdrukken
               (polygonen) met per punt de lokale baanrichting.
3. Router    : reroute_around_schemes() zoekt lokaal (raster + A*) een pad
               dat binnen een voetafdruk alleen loodrecht op de baan loopt.

Geen Streamlit-afhankelijkheid, zodat alles los te testen is.
Vereist: numpy, shapely (>= 1.8), requests (alleen voor fetch_overpass).
"""
from __future__ import annotations

import heapq
import math
from collections import Counter

import numpy as np

try:
    from shapely.geometry import LineString, Point, Polygon
    from shapely.ops import polygonize, unary_union
    try:                                   # shapely >= 2.0
        from shapely import contains_xy as _contains_xy
    except ImportError:                    # shapely 1.8
        from shapely.vectorized import contains as _contains_xy
    HAVE_SHAPELY = True
except Exception:                          # pragma: no cover
    HAVE_SHAPELY = False

R_NM = 3440.065

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
USER_AGENT = "Sailplan/1.0 (Streamlit passage planner)"


# ----------------------------------------------------------------------------
# Basis-geometrie
# ----------------------------------------------------------------------------
def haversine_nm(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R_NM * math.asin(math.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    dlon = math.radians(lon2 - lon1)
    la1, la2 = math.radians(lat1), math.radians(lat2)
    x = math.sin(dlon) * math.cos(la2)
    y = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


class _Proj:
    """Lokale equirectangulaire projectie in zeemijlen rond (lat0, lon0)."""

    def __init__(self, lat0, lon0):
        self.lat0, self.lon0 = lat0, lon0
        self.k = max(0.05, math.cos(math.radians(lat0)))

    def fwd(self, lat, lon):
        return ((lon - self.lon0) * self.k * 60.0, (lat - self.lat0) * 60.0)

    def fwd_arr(self, pts):
        a = np.asarray(pts, dtype=float).reshape(-1, 2)
        return np.column_stack(((a[:, 1] - self.lon0) * self.k * 60.0,
                                (a[:, 0] - self.lat0) * 60.0))

    def inv(self, x, y):
        return (self.lat0 + y / 60.0, self.lon0 + x / (60.0 * self.k))


def _contains(poly, X, Y):
    return np.asarray(_contains_xy(poly, np.ravel(X), np.ravel(Y))).reshape(np.shape(X))


def _densify(route, max_step_nm=1.0):
    out = [list(route[0])]
    for p, q in zip(route[:-1], route[1:]):
        d = haversine_nm(p[0], p[1], q[0], q[1])
        n = int(d // max_step_nm)
        for k in range(1, n + 1):
            f = k / (n + 1)
            out.append([p[0] + f * (q[0] - p[0]), p[1] + f * (q[1] - p[1])])
        out.append([q[0], q[1]])
    return out


def _cum_nm(route):
    c = [0.0]
    for p, q in zip(route[:-1], route[1:]):
        c.append(c[-1] + haversine_nm(p[0], p[1], q[0], q[1]))
    return c


def _length_nm(route):
    return _cum_nm(route)[-1] if len(route) > 1 else 0.0


# ----------------------------------------------------------------------------
# Datamodel
# ----------------------------------------------------------------------------
class Scheme:
    """Een verkeersscheidingsstelsel: voetafdruk + lokale baanrichting."""

    def __init__(self, name, parts, samples, source="OSM"):
        self.name = name
        self.parts = parts            # lijst exterieur-ringen [[(lat, lon), ...], ...]
        self.samples = samples        # lijst (lat, lon, as_graden 0-180)
        self.source = source
        lats = [p[0] for r in parts for p in r]
        lons = [p[1] for r in parts for p in r]
        self.bbox = (min(lats), min(lons), max(lats), max(lons))
        self.center = ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)

    def polygon(self, pr):
        polys = []
        for ring in self.parts:
            if len(ring) >= 3:
                g = Polygon([pr.fwd(la, lo) for la, lo in ring]).buffer(0)
                if not g.is_empty:
                    polys.append(g)
        return unary_union(polys)

    def samples_xy(self, pr):
        if not self.samples:
            return np.zeros((0, 3))
        s = np.asarray(self.samples, dtype=float)
        xy = pr.fwd_arr(s[:, :2])
        return np.column_stack((xy, s[:, 2]))


# ----------------------------------------------------------------------------
# 1. Data ophalen (OpenStreetMap / OpenSeaMap via Overpass)
# ----------------------------------------------------------------------------
def overpass_query(s, w, n, e):
    bb = f"{s:.4f},{w:.4f},{n:.4f},{e:.4f}"
    return ('[out:json][timeout:20];('
            f'way["seamark:type"~"^separation_"]({bb});'
            f'relation["seamark:type"~"^separation_"]({bb});'
            ');out geom;')


def compact_overpass(data):
    """Overpass-JSON (out geom) -> compacte lijst met alleen wat we nodig hebben."""
    out = []
    for el in (data or {}).get("elements", []):
        tags = el.get("tags") or {}
        kind = tags.get("seamark:type", "")
        if not kind.startswith("separation_"):
            continue
        lines = []
        if el.get("type") == "way":
            geo = el.get("geometry") or []
            pts = [(g["lat"], g["lon"]) for g in geo if "lat" in g and "lon" in g]
            if len(pts) >= 2:
                lines.append(pts)
        elif el.get("type") == "relation":
            for m in el.get("members", []):
                geo = m.get("geometry") or []
                pts = [(g["lat"], g["lon"]) for g in geo if "lat" in g and "lon" in g]
                if len(pts) >= 2:
                    lines.append(pts)
        if lines:
            out.append({
                "id": f"{el.get('type')}{el.get('id')}",
                "t": kind,
                "name": tags.get("seamark:name") or tags.get("name") or "",
                "lines": lines,
            })
    return out


def fetch_overpass(s, w, n, e, timeout=25):
    """Haalt de separation-objecten in een bbox op. Gooit RuntimeError bij falen
    (zodat een mislukte aanroep niet in een cache terechtkomt)."""
    import requests
    last = "onbekend"
    for url in OVERPASS_ENDPOINTS:
        try:
            r = requests.post(url, data={"data": overpass_query(s, w, n, e)},
                              headers={"User-Agent": USER_AGENT}, timeout=timeout)
            if r.status_code == 200:
                return compact_overpass(r.json())
            last = f"HTTP {r.status_code}"
        except Exception as ex:                       # netwerk, JSON, timeout
            last = repr(ex)[:100]
    raise RuntimeError(f"Overpass niet bereikbaar ({last})")


def tiles_for_route(route, size=2.0, margin=0.3):
    """Tegels (s, w, n, e) van `size` graden langs de route, voor cachebare queries."""
    tiles = set()
    step = max(1, len(route) // 80)
    for la, lo in route[::step] + [route[-1]]:
        for dla in (-margin, 0.0, margin):
            for dlo in (-margin, 0.0, margin):
                tiles.add((math.floor((la + dla) / size), math.floor((lo + dlo) / size)))
    return sorted((i * size, j * size, i * size + size, j * size + size) for i, j in tiles)


# ----------------------------------------------------------------------------
# 2. Voetafdrukken bouwen
# ----------------------------------------------------------------------------
def _line_samples(line, spacing_nm=1.0):
    """Punten langs een lat/lon-lijn met de lokale as (richting mod 180)."""
    out = []
    for p, q in zip(line[:-1], line[1:]):
        d = haversine_nm(p[0], p[1], q[0], q[1])
        if d < 0.05:
            continue
        axis = bearing_deg(p[0], p[1], q[0], q[1]) % 180.0
        k = max(1, int(d // spacing_nm))
        for i in range(k):
            f = (i + 0.5) / k
            out.append((p[0] + f * (q[0] - p[0]), p[1] + f * (q[1] - p[1]), axis))
    return out


def _polys_from_lines(lines_xy):
    """Maakt polygonen uit (gesloten) lijnen van een way of multipolygon-relatie."""
    try:
        g = unary_union([LineString(l) for l in lines_xy if len(l) >= 2])
        return [p for p in polygonize(g) if p.area > 0.05]
    except Exception:
        return []


def build_schemes(elements, lane_halfwidth_nm=1.0, group_nm=0.7, min_extent_nm=1.5):
    """Maakt voetafdrukken uit compacte OSM-objecten. Lege lijst als er niets bruikbaars is."""
    if not HAVE_SHAPELY or not elements:
        return []
    seen, els = set(), []
    for e in elements:
        if e["id"] not in seen:
            seen.add(e["id"])
            els.append(e)
    allp = [p for e in els for ln in e["lines"] for p in ln]
    if not allp:
        return []
    pr = _Proj(sum(p[0] for p in allp) / len(allp), sum(p[1] for p in allp) / len(allp))

    feats = []     # (geometrie-projected, footprint-bijdrage, element, samples)
    for e in els:
        kind = e["t"]
        lines_xy = [[pr.fwd(la, lo) for la, lo in ln] for ln in e["lines"] if len(ln) >= 2]
        samples = [s for ln in e["lines"] for s in _line_samples(ln)]
        if kind in ("separation_zone", "separation_roundabout"):
            polys = _polys_from_lines(lines_xy)
            if polys:
                for p in polys:
                    feats.append((p, p, e, []))
                continue
        if kind == "separation_lane":
            for ln in lines_xy:
                g = LineString(ln)
                feats.append((g, g.buffer(lane_halfwidth_nm), e, samples))
        else:                                   # boundary / line / roundabout-lijn
            for ln in lines_xy:
                g = LineString(ln)
                feats.append((g, g.buffer(0.3), e, samples))
    if not feats:
        return []

    grouped = unary_union([f[1].buffer(group_nm / 2) for f in feats])
    comps = list(grouped.geoms) if hasattr(grouped, "geoms") else [grouped]

    schemes = []
    for comp in comps:
        members = [f for f in feats if comp.intersects(f[0])]
        if not members:
            continue
        foot = unary_union([f[1] for f in members])
        foot = foot.buffer(0.6).buffer(-0.3).simplify(0.05)    # gaatjes dichten, +0.3 NM marge
        polys = list(foot.geoms) if hasattr(foot, "geoms") else [foot]
        parts, area = [], 0.0
        for pg in polys:
            if pg.is_empty or pg.area < 0.3:
                continue
            area += pg.area
            parts.append([pr.inv(x, y) for x, y in pg.exterior.coords])
        if not parts:
            continue
        minx, miny, maxx, maxy = foot.bounds
        if max(maxx - minx, maxy - miny) < min_extent_nm:
            continue
        samples, seen_s = [], set()
        for f in members:
            for s in f[3]:
                key = (round(s[0], 3), round(s[1], 3))
                if key not in seen_s:
                    seen_s.add(key)
                    samples.append(s)
        if not samples:                                         # alleen zones: as uit de vorm
            try:
                mrr = foot.minimum_rotated_rectangle
                c = list(mrr.exterior.coords)
                e1 = (c[1][0] - c[0][0], c[1][1] - c[0][1])
                e2 = (c[2][0] - c[1][0], c[2][1] - c[1][1])
                e = e1 if math.hypot(*e1) >= math.hypot(*e2) else e2
                axis = math.degrees(math.atan2(e[0], e[1])) % 180.0
                cx, cy = foot.centroid.x, foot.centroid.y
                la, lo = pr.inv(cx, cy)
                samples = [(la, lo, axis)]
            except Exception:
                continue
        names = [m[2]["name"] for m in members if m[2]["name"]]
        if names:
            name = Counter(names).most_common(1)[0][0]
        else:
            la, lo = pr.inv(foot.centroid.x, foot.centroid.y)
            name = f"TSS {abs(la):.1f}°{'N' if la >= 0 else 'S'} {abs(lo):.1f}°{'E' if lo >= 0 else 'W'}"
        schemes.append(Scheme(name, parts, samples))
    return schemes


# ----------------------------------------------------------------------------
# 3. Router: loodrecht kruisen of er omheen
# ----------------------------------------------------------------------------
_MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1),
          (2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]


def _axis_delta(h, ax):
    """Hoek tussen vaarrichting h en de LOODRECHTE op as ax (0 = precies loodrecht)."""
    return abs(((h - ax) % 180.0) - 90.0)


def _chaikin(pts, iters=2):
    for _ in range(iters):
        if len(pts) < 3:
            break
        new = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            new.append((0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]))
            new.append((0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]))
        new.append(pts[-1])
        pts = new
    return pts


class _Grid:
    def __init__(self, W, polys, samples, res, dev_cap, max_cells):
        pad = dev_cap + 2.0
        self.xmin, self.xmax = W[:, 0].min() - pad, W[:, 0].max() + pad
        self.ymin, self.ymax = W[:, 1].min() - pad, W[:, 1].max() + pad
        while True:
            self.nx = int((self.xmax - self.xmin) / res) + 1
            self.ny = int((self.ymax - self.ymin) / res) + 1
            if self.nx * self.ny <= max_cells:
                break
            res *= 1.25
        self.res = res
        xs = self.xmin + res * np.arange(self.nx)
        ys = self.ymin + res * np.arange(self.ny)
        X, Y = np.meshgrid(xs, ys)
        # voetafdruk
        F = np.zeros(X.shape, dtype=bool)
        for p in polys:
            minx, miny, maxx, maxy = p.bounds
            if maxx < self.xmin or minx > self.xmax or maxy < self.ymin or miny > self.ymax:
                continue
            F |= _contains(p, X, Y)
        # lokale baanrichting per cel (dichtstbijzijnde sample)
        A = np.full(X.shape, -1.0)
        if F.any() and len(samples):
            ci = np.nonzero(F.ravel())[0]
            cx, cy = X.ravel()[ci], Y.ravel()[ci]
            ax = np.empty(len(ci))
            for s in range(0, len(ci), 4000):
                dx = cx[s:s + 4000, None] - samples[None, :, 0]
                dy = cy[s:s + 4000, None] - samples[None, :, 1]
                ax[s:s + 4000] = samples[np.argmin(dx * dx + dy * dy, axis=1), 2]
            A.ravel()[ci] = ax
        # afstand tot de oorspronkelijke route
        dens = [W[0]]
        for a, b in zip(W[:-1], W[1:]):
            n = max(1, int(math.hypot(*(b - a)) / (res * 0.5)))
            for k in range(1, n + 1):
                dens.append(a + (b - a) * k / n)
        S = np.asarray(dens)
        D = np.full(X.shape, np.inf)
        for s in range(0, len(S), 8):
            blk = S[s:s + 8]
            d = np.sqrt((X[..., None] - blk[:, 0]) ** 2 + (Y[..., None] - blk[:, 1]) ** 2).min(axis=-1)
            D = np.minimum(D, d)
        self.F, self.A, self.D = F, A, D
        self.blocked = D > dev_cap
        self.X, self.Y = X, Y

    def idx(self, x, y):
        return (int(round((y - self.ymin) / self.res)), int(round((x - self.xmin) / self.res)))


def _astar(g, start_xy, goal_xy, perp_tol, lateral_weight=0.25):
    nx, ny, res = g.nx, g.ny, g.res
    Fl = g.F.ravel().tolist()
    Al = g.A.ravel().tolist()
    Bl = g.blocked.ravel().tolist()
    Ml = (1.0 + lateral_weight * g.D).ravel().tolist()
    mv = [(dx, dy, math.hypot(dx, dy) * res, math.degrees(math.atan2(dx, dy)) % 180.0)
          for dx, dy in _MOVES]
    sy, sx = g.idx(*start_xy)
    ty, tx = g.idx(*goal_xy)
    sy, sx = min(max(sy, 0), ny - 1), min(max(sx, 0), nx - 1)
    ty, tx = min(max(ty, 0), ny - 1), min(max(tx, 0), nx - 1)
    s_i, t_i = sy * nx + sx, ty * nx + tx
    gs = {s_i: 0.0}
    parent = {s_i: -1}
    heap = [(math.hypot(tx - sx, ty - sy) * res, s_i)]
    done = set()
    while heap:
        f, u = heapq.heappop(heap)
        if u in done:
            continue
        if u == t_i:
            break
        done.add(u)
        ux, uy = u % nx, u // nx
        gu = gs[u]
        for dx, dy, ln, h in mv:
            vx, vy = ux + dx, uy + dy
            if vx < 0 or vy < 0 or vx >= nx or vy >= ny:
                continue
            v = vy * nx + vx
            if Bl[v] and v != t_i:
                continue
            zm = 1.0
            if Fl[u] or Fl[v]:
                ax = Al[v] if Fl[v] else Al[u]
                if ax < 0:
                    ax = Al[u] if Al[u] >= 0 else 0.0
                zm = 4.0 if _axis_delta(h, ax) <= perp_tol else 60.0
            c = gu + ln * 0.5 * (Ml[u] + Ml[v]) * zm
            if c < gs.get(v, 1e18):
                gs[v] = c
                parent[v] = u
                heapq.heappush(heap, (c + math.hypot(tx - vx, ty - vy) * res, v))
    if t_i not in parent:
        return None
    path, u = [], t_i
    while u != -1:
        path.append((g.xmin + (u % nx) * res, g.ymin + (u // nx) * res))
        u = parent[u]
    return path[::-1]


def _compress(path):
    if len(path) < 3:
        return path
    out = [path[0]]
    for i in range(1, len(path) - 1):
        a, b, c = path[i - 1], path[i], path[i + 1]
        if abs((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])) > 1e-9:
            out.append(b)
    out.append(path[-1])
    return out


def _seg_ok(g, p, q, perp_tol, cap):
    L = math.hypot(q[0] - p[0], q[1] - p[1])
    k = max(2, int(L / (g.res * 0.5)) + 1)
    t = np.linspace(0.0, 1.0, k)
    xs, ys = p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])
    ix = np.rint((xs - g.xmin) / g.res).astype(int)
    iy = np.rint((ys - g.ymin) / g.res).astype(int)
    if (ix < 0).any() or (iy < 0).any() or (ix >= g.nx).any() or (iy >= g.ny).any():
        return False
    if g.blocked[iy, ix].any() or g.D[iy, ix].max() > cap:
        return False
    inF = g.F[iy, ix]
    if inF.any():
        h = math.degrees(math.atan2(q[0] - p[0], q[1] - p[1])) % 180.0
        ax = g.A[iy, ix][inF]
        ax = np.where(ax < 0, 0.0, ax)
        if (np.abs(((h - ax) % 180.0) - 90.0) > perp_tol).any():
            return False
    return True


def _string_pull(g, path, perp_tol, cap):
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _seg_ok(g, path[i], path[j], perp_tol, cap):
            j -= 1
        out.append(path[j])
        i = j
    return out


def _violation_nm(g, xy, perp_tol):
    """Lengte (NM) van een pad die binnen een voetafdruk NIET (bijna) loodrecht loopt."""
    bad = 0.0
    for p, q in zip(xy[:-1], xy[1:]):
        L = math.hypot(q[0] - p[0], q[1] - p[1])
        if L < 1e-6:
            continue
        k = max(2, int(L / (g.res * 0.5)) + 1)
        t = np.linspace(0, 1, k)
        xs, ys = p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])
        ix = np.rint((xs - g.xmin) / g.res).astype(int)
        iy = np.rint((ys - g.ymin) / g.res).astype(int)
        ok = (ix >= 0) & (iy >= 0) & (ix < g.nx) & (iy < g.ny)
        ixc, iyc = np.clip(ix, 0, g.nx - 1), np.clip(iy, 0, g.ny - 1)
        inF = ok & g.F[iyc, ixc]
        if not inF.any():
            continue
        h = math.degrees(math.atan2(q[0] - p[0], q[1] - p[1])) % 180.0
        ax = np.where(g.A[iyc, ixc] < 0, 0.0, g.A[iyc, ixc])
        wrong = inF & (np.abs(((h - ax) % 180.0) - 90.0) > perp_tol)
        bad += L * float(wrong.sum()) / k
    return bad


def _solve_window(seg, schemes, near_idx, res_nm, dev_cap, perp_tol, max_cells):
    la0 = sum(p[0] for p in seg) / len(seg)
    lo0 = sum(p[1] for p in seg) / len(seg)
    pr = _Proj(la0, lo0)
    W = pr.fwd_arr(seg)
    polys = [schemes[i].polygon(pr) for i in near_idx]
    samp = [schemes[i].samples_xy(pr) for i in near_idx]
    samples = np.vstack(samp) if samp else np.zeros((0, 3))
    g = _Grid(W, polys, samples, res_nm, dev_cap, max_cells)

    old_bad = _violation_nm(g, W, perp_tol)
    old_len = float(np.sum(np.hypot(*(W[1:] - W[:-1]).T)))
    if old_bad < 0.2:                                  # route is al in orde
        return None, {"changed": False, "bad_before": old_bad, "bad_after": old_bad,
                      "added_nm": 0.0, "res": g.res}

    path = _astar(g, tuple(W[0]), tuple(W[-1]), perp_tol)
    if not path:
        return None, {"changed": False, "failed": True, "bad_before": old_bad,
                      "bad_after": old_bad, "added_nm": 0.0, "res": g.res}
    cap = 0.0
    for p in path:
        iy, ix = g.idx(*p)
        if 0 <= iy < g.ny and 0 <= ix < g.nx:
            cap = max(cap, float(g.D[iy, ix]))
    cap += 0.75
    comp = _compress(path)
    comp[0], comp[-1] = tuple(W[0]), tuple(W[-1])
    pulled = _string_pull(g, comp, perp_tol, cap)
    smooth = _chaikin(pulled, 2)
    if all(_seg_ok(g, a, b, perp_tol + 5.0, cap + 0.5) for a, b in zip(smooth[:-1], smooth[1:])):
        final = smooth
    else:
        final = pulled
    final = [tuple(p) for p in final]
    final[0], final[-1] = tuple(W[0]), tuple(W[-1])
    new_len = float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(final[:-1], final[1:])))
    new_bad = _violation_nm(g, np.asarray(final), perp_tol + 5.0)
    if new_len > 1.6 * old_len + 6.0 or new_bad > old_bad:
        return None, {"changed": False, "rejected": True, "bad_before": old_bad,
                      "bad_after": new_bad, "added_nm": new_len - old_len, "res": g.res}
    latlon = [list(pr.inv(x, y)) for x, y in final]
    latlon[0], latlon[-1] = list(seg[0]), list(seg[-1])
    return latlon, {"changed": True, "bad_before": old_bad, "bad_after": new_bad,
                    "added_nm": new_len - old_len, "res": g.res}


def reroute_around_schemes(route, schemes, endpoint_margin_nm=5.0, pad_nm=8.0,
                           dev_cap_nm=10.0, res_nm=0.5, perp_tol=25.0, max_cells=160000):
    """
    Past de route lokaal aan zodat hij binnen een TSS-voetafdruk alleen (bijna)
    loodrecht op de baan loopt. Geeft (nieuwe_route, rapport) terug.
    Is er geen data of geen conflict, dan komt de route onveranderd terug.
    """
    report = {"windows": [], "touched": []}
    if not HAVE_SHAPELY or not schemes or len(route) < 3:
        return [list(p) for p in route], report
    rt = _densify([list(p) for p in route], 1.0)
    n = len(rt)
    cum = _cum_nm(rt)
    arr = np.asarray(rt)
    rb = (arr[:, 0].min() - 0.3, arr[:, 1].min() - 0.4, arr[:, 0].max() + 0.3, arr[:, 1].max() + 0.4)

    near = [i for i, s in enumerate(schemes)
            if not (s.bbox[2] < rb[0] or s.bbox[0] > rb[2] or s.bbox[3] < rb[1] or s.bbox[1] > rb[3])]

    wins = []
    for si in near:
        sc = schemes[si]
        pr = _Proj(*sc.center)
        poly = sc.polygon(pr)
        if poly.is_empty:
            continue
        xy = pr.fwd_arr(rt)
        line = LineString(xy)
        if not line.intersects(poly):
            continue
        if (poly.distance(Point(xy[0])) < endpoint_margin_nm
                or poly.distance(Point(xy[-1])) < endpoint_margin_nm):
            continue                                       # havennadering: niet corrigeren
        inside = _contains(poly.buffer(2.0), xy[:, 0], xy[:, 1])
        idx = np.nonzero(inside)[0]
        if idx.size == 0:
            continue
        lo, hi = int(idx.min()), int(idx.max())
        i0, i1 = lo, hi
        while i0 > 0 and cum[lo] - cum[i0 - 1] <= pad_nm:
            i0 -= 1
        while i1 < n - 1 and cum[i1 + 1] - cum[hi] <= pad_nm:
            i1 += 1
        wins.append([max(0, i0 - 1), min(n - 1, i1 + 1), {si}])

    if not wins:
        return [list(p) for p in route], report
    wins.sort(key=lambda w: w[0])
    merged = [wins[0]]
    for w in wins[1:]:
        if w[0] <= merged[-1][1] + 3:
            merged[-1][1] = max(merged[-1][1], w[1])
            merged[-1][2] |= w[2]
        else:
            merged.append(w)

    out, cursor, changed_any = [], 0, False
    for i0, i1, sset in merged:
        seg = rt[i0:i1 + 1]
        c_lat = sum(p[0] for p in seg) / len(seg)
        c_lon = sum(p[1] for p in seg) / len(seg)
        near_idx = [i for i in near
                    if abs(schemes[i].center[0] - c_lat) < 1.5 and abs(schemes[i].center[1] - c_lon) < 3.0]
        names = sorted({schemes[i].name for i in sset})
        try:
            new_pts, info = _solve_window(seg, schemes, near_idx or sorted(sset),
                                          res_nm, dev_cap_nm, perp_tol, max_cells)
        except Exception as ex:                            # nooit de route laten stukgaan
            new_pts, info = None, {"changed": False, "error": repr(ex)[:80]}
        info["names"] = names
        report["windows"].append(info)
        report["touched"].extend(names)
        if new_pts:
            out.extend(rt[cursor:i0])
            out.extend(_densify(new_pts, 1.0))
            cursor = i1 + 1
            changed_any = True
    out.extend(rt[cursor:])
    report["touched"] = sorted(set(report["touched"]))
    if not changed_any:
        return [list(p) for p in route], report
    return out, report


def thin_route(route, tol_nm=0.15, max_gap_nm=None):
    """Dunt een dichte route uit (Douglas-Peucker) en vult te grote gaten weer op."""
    if len(route) < 3:
        return [list(p) for p in route]
    mid = route[len(route) // 2]
    pr = _Proj(mid[0], mid[1])
    xy = pr.fwd_arr(route)
    keep = {0, len(route) - 1}
    stack = [(0, len(route) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = xy[i], xy[j]
        ab = b - a
        L = math.hypot(ab[0], ab[1])
        seg = xy[i + 1:j] - a
        if L < 1e-9:
            d = np.hypot(seg[:, 0], seg[:, 1])
        else:
            d = np.abs(ab[0] * seg[:, 1] - ab[1] * seg[:, 0]) / L
        k = int(np.argmax(d))
        if d[k] > tol_nm:
            m = i + 1 + k
            keep.add(m)
            stack += [(i, m), (m, j)]
    out = [list(route[k]) for k in sorted(keep)]
    return _densify(out, max_gap_nm) if max_gap_nm else out


# ----------------------------------------------------------------------------
# 4. Indicatieve terugval (geen echte geometrie): alleen waarschuwen
# ----------------------------------------------------------------------------
# Grove tabel (middelpunt + straal) uit de eerste versie van de app. De bronnen zijn niet
# geverifieerd: de ligging en vooral de richtingen zijn globaal. Daarom wordt de route
# hiermee NIET aangepast; de tabel dient alleen voor een indicatieve waarschuwing.
TABLE_ZONES = {
    # ── Dover Strait / Pas de Calais ────────────────────────────────────────
    # Drukste scheepvaartroute ter wereld. NE-SW lanes, loodrecht = NW-SE (135°/315°)
    "dover_strait": {
        "lane_bearing":  225,    # schepen varen NE of SW
        "cross_bearing":  135,   # recreatie kruist op 135° (ZO) of 315° (NW)
        "zone_center":   (51.05, 1.60),   # midden Pas de Calais
        "zone_radius_nm": 12,
        "cross_line": [          # loodrechte kruisingsas (punten N→Z langs 1.60°E)
            (51.30, 1.60),
            (50.80, 1.60),
        ],
    },
    # ── Noord-Hinder Junction ───────────────────────────────────────────────
    # Knooppunt Noordzee, ca. 51.5N 2.5E. NE/SW en N/Z lanes
    "noord_hinder": {
        "lane_bearing":  45,
        "cross_bearing": 135,
        "zone_center":   (51.50, 2.50),
        "zone_radius_nm": 10,
        "cross_line": [
            (51.80, 2.50),
            (51.20, 2.50),
        ],
    },
    # ── IJmuiden TSS (Zee/Kust bij IJmuiden) ────────────────────────────────
    "ijmuiden_tss": {
        "lane_bearing":  270,    # oost-west langs kust
        "cross_bearing":   0,    # loodrecht = N-Z
        "zone_center":   (52.45, 4.25),
        "zone_radius_nm":  8,
        "cross_line": [
            (52.70, 4.25),
            (52.20, 4.25),
        ],
    },
    # ── Texel TSS ───────────────────────────────────────────────────────────
    "texel_tss": {
        "lane_bearing":  350,
        "cross_bearing":  80,
        "zone_center":   (53.10, 4.50),
        "zone_radius_nm":  8,
        "cross_line": [
            (53.30, 4.10),
            (52.90, 4.90),
        ],
    },
    # ── Terschelling / German Bight Westbound ───────────────────────────────
    "terschelling_tss": {
        "lane_bearing":  270,
        "cross_bearing":   0,
        "zone_center":   (53.50, 4.80),
        "zone_radius_nm": 10,
        "cross_line": [
            (53.75, 4.80),
            (53.25, 4.80),
        ],
    },
    # ── Off Frisian Islands (Borkum/Elbe) ───────────────────────────────────
    "frisian_tss": {
        "lane_bearing":  90,
        "cross_bearing":   0,
        "zone_center":   (54.00, 6.50),
        "zone_radius_nm": 10,
        "cross_line": [
            (54.30, 6.50),
            (53.70, 6.50),
        ],
    },
    # ── German Bight Western Approach ────────────────────────────────────────
    # Kruispunt van scheepvaartroutes Elbe/Weser/Jade/Eems — Waddenzee → DK
    "german_bight_west_tss": {
        "lane_bearing":  315, "cross_bearing":  45,
        "zone_center":   (54.60, 7.80), "zone_radius_nm": 12,
        "cross_line": [(54.90, 7.50), (54.30, 8.10)],
    },
    # ── German Bight Eastern Approach (Elbe/Weser aanloop) ───────────────────
    "german_bight_east_tss": {
        "lane_bearing":  270, "cross_bearing":   0,
        "zone_center":   (54.00, 8.20), "zone_radius_nm": 10,
        "cross_line": [(54.30, 8.20), (53.70, 8.20)],
    },
    # ── Sylt / Lister Tief approach (Lauwersoog → Esbjerg / Denemarken) ──────
    "sylt_lister_tss": {
        "lane_bearing":  350, "cross_bearing":  80,
        "zone_center":   (55.05, 8.15), "zone_radius_nm":  9,
        "cross_line": [(55.10, 7.85), (55.00, 8.45)],
    },
    # Esbjerg approach en Horns Rev zijn te dicht bij de haven en worden
    # al gedekt door german_bight_west + sylt_lister — niet apart opnemen.
    # ── Elbe Aanloop TSS ──────────────────────────────────────────────────────
    "elbe_tss": {
        "lane_bearing":  270, "cross_bearing":   0,
        "zone_center":   (54.00, 8.70), "zone_radius_nm": 10,
        "cross_line": [(54.30, 8.70), (53.70, 8.70)],
    },
    # ── Weser Aanloop TSS ─────────────────────────────────────────────────────
    "weser_tss": {
        "lane_bearing":  315, "cross_bearing":  45,
        "zone_center":   (53.85, 8.10), "zone_radius_nm":  9,
        "cross_line": [(54.10, 7.85), (53.60, 8.35)],
    },
    # ── Off Thyborøn (Limfjord ingang, Denemarken) ────────────────────────────
    "thyboron_tss": {
        "lane_bearing":    0, "cross_bearing":  90,
        "zone_center":   (56.70, 8.20), "zone_radius_nm":  8,
        "cross_line": [(56.70, 7.80), (56.70, 8.60)],
    },
    # ── Great Belt / Storebælt ────────────────────────────────────────────────
    "great_belt_tss": {
        "lane_bearing":  180, "cross_bearing":  90,
        "zone_center":   (55.30, 11.00), "zone_radius_nm": 10,
        "cross_line": [(55.30, 10.60), (55.30, 11.40)],
    },
    # ── Little Belt / Lillebælt ───────────────────────────────────────────────
    "little_belt_tss": {
        "lane_bearing":  180, "cross_bearing":  90,
        "zone_center":   (55.50, 9.80), "zone_radius_nm":  8,
        "cross_line": [(55.50, 9.40), (55.50, 10.20)],
    },
    # ── South Falls / Off Lowestoft ─────────────────────────────────────────
    "south_falls": {
        "lane_bearing":  20,
        "cross_bearing": 110,
        "zone_center":   (52.10, 2.00),
        "zone_radius_nm":  8,
        "cross_line": [
            (52.40, 1.70),
            (51.80, 2.30),
        ],
    },
    # ── Off Lands End / Isles of Scilly ────────────────────────────────────
    "lands_end_tss": {
        "lane_bearing":  60,     # NE-SW Kanaal uitloop
        "cross_bearing": 150,
        "zone_center":   (49.90, -6.50),
        "zone_radius_nm": 12,
        "cross_line": [
            (50.20, -6.80),
            (49.60, -6.20),
        ],
    },
    # ── Ushant / Ouessant TSS (Atlantische ingang Kanaal) ───────────────────
    "ushant_tss": {
        "lane_bearing":  60,
        "cross_bearing": 150,
        "zone_center":   (48.50, -5.50),
        "zone_radius_nm": 15,
        "cross_line": [
            (48.90, -5.90),
            (48.10, -5.10),
        ],
    },
    # ── Casquets TSS (NW Normandie / Channel Islands) ───────────────────────
    "casquets_tss": {
        "lane_bearing":  75,
        "cross_bearing": 165,
        "zone_center":   (49.80, -2.50),
        "zone_radius_nm": 10,
        "cross_line": [
            (50.10, -2.70),
            (49.50, -2.30),
        ],
    },
    # ── Skagerrak (Noorwegen/Denemarken) ────────────────────────────────────
    "skagerrak_tss": {
        "lane_bearing":  270,
        "cross_bearing":   0,
        "zone_center":   (57.50, 9.00),
        "zone_radius_nm": 10,
        "cross_line": [
            (57.80, 9.00),
            (57.20, 9.00),
        ],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # NEDERLAND / NOORDZEE KUST (aanvullend)
    # ══════════════════════════════════════════════════════════════════════════
    "vlieland_tss": {
        "lane_bearing":  315, "cross_bearing":  45,
        "zone_center":   (53.30, 4.70), "zone_radius_nm":  9,
        "cross_line": [(53.55, 4.45), (53.05, 4.95)],
    },
    "vlieland_north_tss": {
        "lane_bearing":  270, "cross_bearing":   0,
        "zone_center":   (53.55, 4.60), "zone_radius_nm":  7,
        "cross_line": [(53.80, 4.60), (53.30, 4.60)],
    },
    "off_texel_tss": {
        "lane_bearing":  315, "cross_bearing":  45,
        "zone_center":   (53.00, 4.40), "zone_radius_nm":  8,
        "cross_line": [(53.25, 4.15), (52.75, 4.65)],
    },
    "maas_northwest_tss": {
        "lane_bearing":  315, "cross_bearing":  45,
        "zone_center":   (52.00, 3.70), "zone_radius_nm":  9,
        "cross_line": [(52.30, 3.40), (51.70, 4.00)],
    },
    "west_hinder_tss": {
        "lane_bearing":  225, "cross_bearing": 135,
        "zone_center":   (51.40, 2.30), "zone_radius_nm":  8,
        "cross_line": [(51.65, 2.05), (51.15, 2.55)],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # NOORWEGEN
    # ══════════════════════════════════════════════════════════════════════════
    "off_runde_tss": {
        "lane_bearing":    0, "cross_bearing":  90,
        "zone_center":   (62.40, 5.00), "zone_radius_nm": 10,
        "cross_line": [(62.40, 4.60), (62.40, 5.40)],
    },
    "off_stad_tss": {
        "lane_bearing":    0, "cross_bearing":  90,
        "zone_center":   (62.10, 4.70), "zone_radius_nm":  9,
        "cross_line": [(62.10, 4.30), (62.10, 5.10)],
    },
    "off_sotra_tss": {
        "lane_bearing":    0, "cross_bearing":  90,
        "zone_center":   (60.25, 4.60), "zone_radius_nm":  8,
        "cross_line": [(60.25, 4.20), (60.25, 5.00)],
    },
    "north_cape_tss": {
        "lane_bearing":  270, "cross_bearing":   0,
        "zone_center":   (71.20, 25.80), "zone_radius_nm": 12,
        "cross_line": [(71.50, 25.80), (70.90, 25.80)],
    },
    "rost_tss": {
        "lane_bearing":  330, "cross_bearing":  60,
        "zone_center":   (67.50, 11.50), "zone_radius_nm": 10,
        "cross_line": [(67.80, 11.00), (67.20, 12.00)],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # IJSLAND
    # ══════════════════════════════════════════════════════════════════════════
    "iceland_gardskagi_tss": {
        "lane_bearing":  270, "cross_bearing":   0,
        "zone_center":   (64.10, -22.80), "zone_radius_nm": 12,
        "cross_line": [(64.40, -22.80), (63.80, -22.80)],
    },
    "iceland_reykjanes_tss": {
        "lane_bearing":   90, "cross_bearing":   0,
        "zone_center":   (63.60, -24.00), "zone_radius_nm": 12,
        "cross_line": [(63.90, -24.00), (63.30, -24.00)],
    },
    "faxafloi_tss": {
        "lane_bearing":  180, "cross_bearing":  90,
        "zone_center":   (64.20, -23.20), "zone_radius_nm": 10,
        "cross_line": [(64.20, -23.60), (64.20, -22.80)],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # IERLAND / SCHOTLAND
    # ══════════════════════════════════════════════════════════════════════════
    "north_channel_tss": {
        "lane_bearing":  210, "cross_bearing": 120,
        "zone_center":   (55.20, -5.80), "zone_radius_nm": 10,
        "cross_line": [(55.50, -6.10), (54.90, -5.50)],
    },
    "malin_head_tss": {
        "lane_bearing":  250, "cross_bearing": 160,
        "zone_center":   (55.50, -7.50), "zone_radius_nm":  9,
        "cross_line": [(55.80, -7.70), (55.20, -7.30)],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # ATLANTISCHE KUST — PORTUGAL / SPANJE
    # ══════════════════════════════════════════════════════════════════════════
    "cape_finisterre_tss": {
        "lane_bearing":  180, "cross_bearing":  90,
        "zone_center":   (42.90, -9.50), "zone_radius_nm": 12,
        "cross_line": [(42.90, -9.90), (42.90, -9.10)],
    },
    "cape_st_vincent_tss": {
        "lane_bearing":  180, "cross_bearing":  90,
        "zone_center":   (36.80, -9.00), "zone_radius_nm": 12,
        "cross_line": [(36.80, -9.50), (36.80, -8.50)],
    },
    "gibraltar_tss": {
        "lane_bearing":   90, "cross_bearing":   0,
        "zone_center":   (35.95, -5.60), "zone_radius_nm": 10,
        "cross_line": [(36.20, -5.60), (35.70, -5.60)],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # OOSTZEE
    # ══════════════════════════════════════════════════════════════════════════
    "north_rugen_tss": {
        "lane_bearing":   90, "cross_bearing":   0,
        "zone_center":   (54.80, 13.50), "zone_radius_nm":  8,
        "cross_line": [(55.10, 13.50), (54.50, 13.50)],
    },
    "hoburgs_north_tss": {
        "lane_bearing":    0, "cross_bearing":  90,
        "zone_center":   (57.80, 19.20), "zone_radius_nm":  8,
        "cross_line": [(57.80, 18.80), (57.80, 19.60)],
    },
    "hoburgs_south_tss": {
        "lane_bearing":    0, "cross_bearing":  90,
        "zone_center":   (57.20, 19.10), "zone_radius_nm":  8,
        "cross_line": [(57.20, 18.70), (57.20, 19.50)],
    },
    "kattegat_north_tss": {
        "lane_bearing":  180, "cross_bearing":  90,
        "zone_center":   (57.40, 11.00), "zone_radius_nm":  9,
        "cross_line": [(57.40, 10.60), (57.40, 11.40)],
    },

    # ── Öresund / Sont (Kopenhagen) ───────────────────────────────────────────
    "oresund_tss": {
        "lane_bearing":  180, "cross_bearing":  90,
        "zone_center":   (55.80, 12.60), "zone_radius_nm":  8,
        "cross_line": [(55.80, 12.20), (55.80, 13.00)],
    },

    # ── Off Egersund (ZW Noorwegen, route van NL naar Stavanger) ─────────────
    "egersund_tss": {
        "lane_bearing":    0, "cross_bearing":  90,
        "zone_center":   (58.40,  5.50), "zone_radius_nm":  9,
        "cross_line": [(58.40, 5.10), (58.40, 5.90)],
    },

    # ── Off Utsira (toegang Stavanger/Bergen, druk scheepvaartgebied) ─────────
    "utsira_tss": {
        "lane_bearing":    0, "cross_bearing":  90,
        "zone_center":   (59.30,  4.80), "zone_radius_nm": 10,
        "cross_line": [(59.30, 4.40), (59.30, 5.20)],
    },

    # ── Fair Isle Channel (Schotland → Noorwegen / Noorwegen → Schotland) ────
    "fair_isle_tss": {
        "lane_bearing":   90, "cross_bearing":   0,
        "zone_center":   (59.50, -2.00), "zone_radius_nm":  9,
        "cross_line": [(59.80, -2.00), (59.20, -2.00)],
    },

    # ── Pentland Firth (N-Schotland, drukste TSS van de Britse eilanden) ──────
    "pentland_firth_tss": {
        "lane_bearing":  270, "cross_bearing":   0,
        "zone_center":   (58.70, -3.20), "zone_radius_nm": 10,
        "cross_line": [(59.00, -3.20), (58.40, -3.20)],
    },

    # ── Off Ekofisk (centrale Noordzee, olie- en scheepvaartgebied) ──────────
    "ekofisk_tss": {
        "lane_bearing":  315, "cross_bearing":  45,
        "zone_center":   (56.50,  3.20), "zone_radius_nm": 10,
        "cross_line": [(56.80, 2.90), (56.20, 3.50)],
    },

    # ── Off Egersund uitgebreid — Stavanger aanloop ───────────────────────────
    "stavanger_approach_tss": {
        "lane_bearing":    0, "cross_bearing":  90,
        "zone_center":   (58.90,  5.00), "zone_radius_nm": 12,
        "cross_line": [(58.90, 4.50), (58.90, 5.50)],
    },

    # ── Off Ålesund / Møre (tss midden-Noorwegen) ────────────────────────────
    "more_tss": {
        "lane_bearing":    0, "cross_bearing":  90,
        "zone_center":   (62.50,  4.80), "zone_radius_nm": 12,
        "cross_line": [(62.50, 4.30), (62.50, 5.30)],
    },

    # ── Centrale Noordzee / Forties oilfield (Edinburgh/Aberdeen route) ───────
    "central_north_sea_tss": {
        "lane_bearing":  315, "cross_bearing":  45,
        "zone_center":   (57.50,  1.00), "zone_radius_nm": 12,
        "cross_line": [(57.80, 0.70), (57.20, 1.30)],
    },

    # ── Off Skagen (Denemarken/Noorwegen, Göteborg aanloop) ──────────────────
    "skagen_tss": {
        "lane_bearing":  315, "cross_bearing":  45,
        "zone_center":   (57.80,  9.80), "zone_radius_nm": 10,
        "cross_line": [(58.10, 9.50), (57.50, 10.10)],
    },

    # ── Göteborg aanloop (Kattegat/Skagerrak overgang) ────────────────────────
    "goteborg_approach_tss": {
        "lane_bearing":  180, "cross_bearing":  90,
        "zone_center":   (57.60, 11.50), "zone_radius_nm": 10,
        "cross_line": [(57.60, 11.10), (57.60, 11.90)],
    },

    # ── Læsø Rende (Noord-Kattegat, smalle vaargeul DK) ─────────────────────
    "laeso_rende_tss": {
        "lane_bearing":  180, "cross_bearing":  90,
        "zone_center":   (57.30, 11.00), "zone_radius_nm":  8,
        "cross_line": [(57.30, 10.60), (57.30, 11.40)],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CARIBISCH GEBIED
    # ══════════════════════════════════════════════════════════════════════════
    "mona_passage_tss": {
        "lane_bearing":  180, "cross_bearing":  90,
        "zone_center":   (18.20, -67.80), "zone_radius_nm": 12,
        "cross_line": [(18.20, -68.20), (18.20, -67.40)],
    },
    "windward_passage_tss": {
        "lane_bearing":  180, "cross_bearing":  90,
        "zone_center":   (19.80, -74.10), "zone_radius_nm": 12,
        "cross_line": [(19.80, -74.50), (19.80, -73.70)],
    },
    "yucatan_channel_tss": {
        "lane_bearing":    0, "cross_bearing":  90,
        "zone_center":   (21.50, -85.50), "zone_radius_nm": 12,
        "cross_line": [(21.50, -85.90), (21.50, -85.10)],
    },
    "florida_strait_tss": {
        "lane_bearing":   45, "cross_bearing": 135,
        "zone_center":   (24.50, -80.50), "zone_radius_nm": 12,
        "cross_line": [(24.80, -80.80), (24.20, -80.20)],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # SPITSBERGEN / GROENLAND
    # ══════════════════════════════════════════════════════════════════════════
    "bear_island_tss": {
        "lane_bearing":    0, "cross_bearing":  90,
        "zone_center":   (74.50, 19.00), "zone_radius_nm": 12,
        "cross_line": [(74.50, 18.50), (74.50, 19.50)],
    },
    "denmark_strait_tss": {
        "lane_bearing":  180, "cross_bearing":  90,
        "zone_center":   (66.00, -27.00), "zone_radius_nm": 15,
        "cross_line": [(66.00, -27.60), (66.00, -26.40)],
    },
    "cape_farewell_tss": {
        "lane_bearing":  270, "cross_bearing":   0,
        "zone_center":   (59.50, -43.50), "zone_radius_nm": 12,
        "cross_line": [(59.80, -43.50), (59.20, -43.50)],
    },
}


def indicative_hits(route, schemes=None, zones=None):
    """Namen van tabelzones waar de route doorheen gaat en die NIET al door echte
    OSM-data gedekt zijn. Puur informatief: de route wordt hier niet door aangepast."""
    zones = TABLE_ZONES if zones is None else zones
    if not route:
        return []
    step = max(1, len(route) // 200)
    pts = route[::step] + [route[-1]]
    hits = []
    for name, z in zones.items():
        clat, clon = z["zone_center"]
        r = z["zone_radius_nm"]
        if min(haversine_nm(p[0], p[1], clat, clon) for p in pts) > r:
            continue
        covered = False
        for sc in schemes or []:
            b = sc.bbox
            if b[0] - 0.2 <= clat <= b[2] + 0.2 and b[1] - 0.3 <= clon <= b[3] + 0.3:
                covered = True
                break
        if not covered:
            hits.append({"name": name.replace("_tss", "").replace("_", " ").title(),
                         "lat": clat, "lon": clon, "r_nm": r})
    return hits
