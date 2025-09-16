import streamlit as st
import networkx as nx
import heapq
import tempfile
from pyvis.network import Network
from typing import List, Tuple, Dict, Any, Iterator

# ---------------------------
# Utility & MST step generators
# ---------------------------
class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if self.parent.get(x, x) != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent.get(x, x)

    def union(self, a, b) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank.get(ra, 0) < self.rank.get(rb, 0):
            self.parent[ra] = rb
        else:
            self.parent[rb] = ra
            if self.rank.get(ra, 0) == self.rank.get(rb, 0):
                self.rank[ra] = self.rank.get(ra, 0) + 1
        return True

def kruskal_steps_multigraph(nodes: List[str], edges: List[Tuple[str,str,float,int]]) -> Iterator[Dict[str,Any]]:
    uf = UnionFind()
    for n in nodes:
        uf.parent[n] = n
        uf.rank[n] = 0
    edges_sorted = sorted(edges, key=lambda x: (x[2], x[3]))
    mst = []
    for (u, v, w, k) in edges_sorted:
        yield {'edge': (u, v, w, k), 'action': 'consider', 'mst_edges': list(mst)}
        if uf.union(u, v):
            mst.append((u, v, w, k))
            yield {'edge': (u, v, w, k), 'action': 'add', 'mst_edges': list(mst)}
        else:
            yield {'edge': (u, v, w, k), 'action': 'reject', 'mst_edges': list(mst)}
    yield {'edge': None, 'action': 'done', 'mst_edges': list(mst)}

def prim_steps_multigraph(G: nx.MultiDiGraph, start_node: str=None) -> Iterator[Dict[str,Any]]:
    nodes = list(G.nodes())
    if not nodes:
        yield {'edge': None, 'action': 'done', 'mst_edges': []}
        return
    if start_node is None:
        start_node = nodes[0]
    visited = {start_node}
    mst = []
    heap = []
    for nbr, keydict in G[start_node].items():
        for k, data in G[start_node][nbr].items():
            heapq.heappush(heap, (data['weight'], start_node, nbr, k))
    while heap and len(visited) < len(nodes):
        w, u, v, k = heapq.heappop(heap)
        yield {'edge': (u, v, w, k), 'action': 'consider', 'mst_edges': list(mst)}
        if u in visited and v in visited:
            yield {'edge': (u, v, w, k), 'action': 'reject', 'mst_edges': list(mst)}
            continue
        newnode = v if v not in visited else u
        visited.add(newnode)
        mst.append((u, v, w, k))
        yield {'edge': (u, v, w, k), 'action': 'add', 'mst_edges': list(mst)}
        for nbr, keydict in G[newnode].items():
            for k2, data in keydict.items():
                if nbr not in visited:
                    heapq.heappush(heap, (data['weight'], newnode, nbr, k2))
    yield {'edge': None, 'action': 'done', 'mst_edges': list(mst)}

# ---------------------------
# Streamlit UI & Helpers
# ---------------------------
st.set_page_config(page_title="MST Visualizer (MultiDiGraph)", layout="wide")

st.title("Minimum Spanning Tree Visualizer — Web")

col1, col2 = st.columns([1,2])

with col1:
    st.header("Graph Input")
    mode = st.radio("Input method", ("Manual text", "Upload CSV"))
    nodes = []
    edges = []
    if mode == "Manual text":
        nodes_text = st.text_area("Nodes (comma separated)", value="A,B,C")
        nodes = [n.strip() for n in nodes_text.split(",") if n.strip()]
        st.markdown("Enter edges, one per line, in format: `u v weight`")
        edges_text = st.text_area("Edges (one per line)", value="A B 1\nB C 3\nC A 4\nB A 2")
        raw_lines = [ln.strip() for ln in edges_text.splitlines() if ln.strip()]
        for i,ln in enumerate(raw_lines):
            parts = ln.split()
            if len(parts) != 3:
                st.error(f"Edge line {i+1} malformed: '{ln}' (expected 3 tokens)")
            else:
                u,v,wstr = parts
                try:
                    w = float(wstr)
                    edges.append((u, v, w))  # keep both A→B and B→A
                except:
                    st.error(f"Edge line {i+1} invalid weight: '{wstr}'")
    else:
        up = st.file_uploader("Upload CSV with columns: u,v,weight", type=["csv","txt"])
        if up is not None:
            import pandas as pd
            try:
                df = pd.read_csv(up)
            except Exception:
                try:
                    df = pd.read_csv(up, header=None, names=['u','v','weight'])
                except Exception:
                    st.error("Could not read file as CSV.")
                    df = None
            if df is not None:
                if set(['u','v','weight']).issubset(df.columns):
                    for u,v,w in df[['u','v','weight']].itertuples(index=False):
                        edges.append((str(u).strip(), str(v).strip(), float(w)))
                        if str(u).strip() not in nodes: nodes.append(str(u).strip())
                        if str(v).strip() not in nodes: nodes.append(str(v).strip())
                elif df.shape[1] >= 3:
                    for row in df.itertuples(index=False):
                        u, v, w = row[0], row[1], row[2]
                        edges.append((str(u).strip(), str(v).strip(), float(w)))
                        if str(u).strip() not in nodes: nodes.append(str(u).strip())
                        if str(v).strip() not in nodes: nodes.append(str(v).strip())
                else:
                    st.error("CSV must have at least three columns: u,v,weight")

    st.markdown("---")
    st.header("Algorithm")
    algo = st.selectbox("Choose algorithm", ("Kruskal", "Prim"))
    prim_start = None
    if algo == "Prim":
        prim_start = st.selectbox("Start node for Prim", ["(auto)"] + nodes)
        if prim_start == "(auto)":
            prim_start = None

    show_steps = st.checkbox("Show step-by-step actions", value=False)

    st.markdown("---")
    if st.button("Build & Visualize"):
        if not nodes:
            st.error("Please enter nodes.")
            st.stop()
        if not edges:
            st.error("Please enter edges.")
            st.stop()

        # Use MultiDiGraph (directed)
        G = nx.MultiDiGraph()
        G.add_nodes_from(nodes)
        edges_with_keys = []
        key_counter = 0
        for (u,v,w) in edges:
            if u not in nodes or v not in nodes:
                st.error(f"Edge references unknown node: {u} or {v}")
                st.stop()
            G.add_edge(u, v, key=key_counter, weight=float(w))
            edges_with_keys.append((u, v, float(w), key_counter))
            key_counter += 1

        if algo == "Kruskal":
            steps_iter = kruskal_steps_multigraph(nodes, edges_with_keys)
        else:
            steps_iter = prim_steps_multigraph(G, start_node=prim_start)

        steps = list(steps_iter)
        final = [s for s in steps if s['action'] == 'done']
        mst_edges = []
        if final:
            mst_edges = final[-1]['mst_edges']

        total_cost = sum(e[2] for e in mst_edges)

        mst_edge_keys = set(e[3] for e in mst_edges)
        net = Network(height="650px", width="100%", directed=True)  # directed
        net.barnes_hut()
        for n in G.nodes():
            net.add_node(n, label=str(n), title=str(n))
        for u, v, k, data in G.edges(keys=True, data=True):
            w = data.get('weight', '')
            is_mst = (k in mst_edge_keys)
            color = "green" if is_mst else "#888888"
            net.add_edge(
                u, v,
                value=float(w),
                label=str(w),
                title=f"w={w} (key={k})",
                color=color,
                arrows="to",  # direction visible
                smooth={'type':'curvedCW', 'roundness': 0.2}
            )

        tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
        net.save_graph(tmp.name)
        st.session_state = getattr(st, "session_state", {})
        st.session_state["tmp_html"] = tmp.name
        st.success("Graph built — scroll right to view interactive visualization")

        st.session_state["steps"] = steps
        st.session_state["mst_edges"] = mst_edges
        st.session_state["total_cost"] = total_cost

with col2:
    st.header("Interactive Graph")
    if "tmp_html" in st.session_state:
        with open(st.session_state["tmp_html"], 'r', encoding='utf-8') as f:
            html = f.read()
        st.components.v1.html(html, height=700)
    else:
        st.info("Build a graph (left) to see the interactive visualization here.")

    st.markdown("---")
    st.header("MST Result")
    if "mst_edges" in st.session_state:
        if st.session_state["mst_edges"]:
            st.write("Edges in MST (u, v, weight, key):")
            for u,v,w,k in st.session_state["mst_edges"]:
                st.write(f"{u} -> {v}  (w={w})  [edge key={k}]")
            st.write("**Total MST cost:**", st.session_state["total_cost"])
        else:
            st.write("No MST edges (graph might be disconnected).")
    else:
        st.write("No MST computed yet. Build a graph from the left column.")

    st.markdown("---")
    st.header("Step-by-step actions")
    if "steps" in st.session_state and st.session_state["steps"]:
        if show_steps:
            for s in st.session_state["steps"]:
                if s['action'] == 'done':
                    st.write("DONE — final MST edges:", s['mst_edges'])
                else:
                    e = s['edge']
                    st.write(f"{s['action'].upper():8}  edge: ({e[0]} -> {e[1]}) w={e[2]} key={e[3]}   | MST now: {s['mst_edges']}")
        else:
            st.write("Toggle 'Show step-by-step actions' to view actions.")
    else:
        st.write("No steps to show.")
