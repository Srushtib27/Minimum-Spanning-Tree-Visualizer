# Minimum-Spanning-Tree-Visualizer-
A web-based interactive visualizer for Kruskal’s and Prim’s algorithms.
Built with Python, Streamlit, NetworkX, and PyVis, this tool allows you to:

1. Enter custom nodes and weighted edges (including parallel edges).
2. Visualize the full graph with weights on edges.
3. Run Kruskal’s or Prim’s algorithm to compute the MST.
4. Highlight MST edges in green while showing non-MST edges in black.

✨ Features

-Supports multiple edges between the same two nodes (via NetworkX MultiGraph).

-Edge weights displayed directly on the graph.

-Interactive visualization rendered in browser using PyVis.

-Choose between Kruskal or Prim algorithms.

-Easy-to-use Streamlit web interface.

🖼️ Demo Screenshot
<img width="946" height="444" alt="image" src="https://github.com/user-attachments/assets/95745d93-c37a-48c6-a791-1ea25b029c54" />

⚙️Installation

Clone this repository:

git clone https://github.com/Srushtib27/Minimum-Spanning-Tree-Visualizer.git
cd Minimum-Spanning-Tree-Visualizer


Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate


Install dependencies:

pip install streamlit networkx pyvis

▶️ Usage

Run the Streamlit app:

streamlit run app.py


Then open the URL shown in your terminal (usually http://localhost:8501) to use the MST Visualizer.

🧮 Algorithms Implemented

Kruskal’s Algorithm → Greedy, edge-sorting approach.

Prim’s Algorithm → Greedy, vertex-growing approach.

Both compute the Minimum Spanning Tree with minimum edge cost.

🚀 Future Improvements

Step-by-step animation of algorithm execution (edges highlighted one by one).

Export MST results as JSON/CSV.

Add support for directed graphs.

📜 License

This project is licensed under the MIT License – free to use, modify, and distribute.
