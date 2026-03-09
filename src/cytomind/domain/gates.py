import json
import re
from typing import Any, Iterable, Mapping
from dataclasses import dataclass, field, asdict
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

from .constants import GML_VERSION
from cytomind.utils import now_iso

__all__ = ["GatingStrategyRef", "GateNode"]

@dataclass
class GateNode:
    """
    Passive data record for a gate in a GatingStrategy.

    Stores gate configuration (type, hyperparams, dimensions, layer) and
    fitted parameters per sample. JSON-serializable for persistence.

    The node is a record - it doesn't perform fitting or application.
    Use the gate class specified in `gate_type` to create Gate instances
    for actual computation.

    Parameter Storage
    -----------------
    Supports both batch-level and sample-specific parameter storage:

    - **params**: Batch-level parameters (dict structure with 'hyperparams', 'params', 'diagnostics').
      Applied to all samples unless overridden by custom_gates.

    - **custom_gates[sample_id]**: Sample-specific parameter overrides (same dict structure).
      Takes precedence over params for that specific sample.

    Use Gate.from_node(node, sample_id) to reconstruct a Gate with the correct parameters:
    - from_node(node) → uses batch-level params
    - from_node(node, "sample_123") → uses custom_gates["sample_123"] or falls back to params
    """

    id: str                               # Unique node identifier
    gate_type: str                        # Gate class name (e.g., "RectangleGate")
    dimensions: list[str]                 # Dimension IDs this gate operates on
    layer: str = "xf"                     # Data layer: "raw", "comp", or "xf"
    name: str | None = None               # Human-readable name
    glm_type: str | None = None           # GatingML type (optional)
    parent_ids: list[str] = field(default_factory=list)  # Parent node IDs in hierarchy
    use_as_complement: bool = False       # Whether to use gate as complement
    params: dict[str, Any] = field(default_factory=dict)  # Batch-level parameters (structure: {hyperparams, params, diagnostics})
    custom_gates: dict[str, dict[str, Any]] = field(default_factory=dict)  # Sample-specific parameter overrides

    def __post_init__(self) -> None:
        pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
        if not re.match(pattern, self.id):
            raise ValueError(f"GateNode id '{self.id}' is not a valid identifier.")

        if self.name is None:
            self.name = self.id

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        return f"GateNode(id={self.id}, type={self.gate_type}, parents={self.parent_ids})"

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize GateNode to dictionary for JSON storage.

        Returns
        -------
        dict[str, Any]
            Serialized gate node configuration
        """

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GateNode":
        """
        Deserialize GateNode from dictionary.

        Parameters
        ----------
        data : Mapping[str, Any]
            Serialized gate node data

        Returns
        -------
        GateNode
            Deserialized gate node instance
        """

        # Backward/forward compatibility for incoming gate payloads:
        # - New shape: params={"hyperparams": ..., "params": ..., "diagnostics": ...}
        # - Legacy add_gate payload: hyperparams={...}
        params = data.get("params", {})
        if not params and "hyperparams" in data:
            params = {
                "hyperparams": dict(data.get("hyperparams", {})),
                "params": {},
                "diagnostics": {},
            }

        return GateNode(
            id=data["id"],
            gate_type=data["gate_type"],
            dimensions=data["dimensions"],
            layer=data.get("layer", "xf"),
            name=data.get("name"),
            glm_type=data.get("glm_type"),
            parent_ids=data.get("parent_ids", []),
            use_as_complement=data.get("use_as_complement", False),
            params=params,
            custom_gates=data.get("custom_gates", {}),
        )

    def get_params_for_sample(self, sample_id: str) -> dict[str, Any]:
        """Get merged parameters for a specific sample.

        Implements parameter precedence: sample-specific overrides fall back to batch-level params.

        Parameters
        ----------
        sample_id : str
            Sample identifier to look up parameters for

        Returns
        -------
        dict[str, Any]
            Merged parameter structure ({"hyperparams": {...}, "params": {...}, "diagnostics": {...}}).
            If sample_id is in custom_gates, returns that (with fallback).
            Otherwise returns batch-level params.

        Examples
        --------
        Get parameters (with precedence):
        >>> params = node.get_params_for_sample("sample_123")
        >>> # Returns custom_gates["sample_123"] if present, else params
        """
        if sample_id in self.custom_gates:
            return self.custom_gates[sample_id]
        else:
            return self.params

@dataclass
class GatingStrategyRef:
    """
    Reference to a gating strategy, which defines gates linked in a directed acyclic graph.

    A gating strategy consists of:
    - Gate definitions (to be added in future steps)
    - References to compensations and transformations used by gates
    - Sample associations
    - GatingML 2.0 specification compliance

    The strategy definition is stored as JSON and can be exported to GatingML XML format.
    """
    id: str                              # unique id (initialy same as name)
    name: str                            # human-readable name (allowed to change)
    batch_id: str                        # batch id to be analyzed
    path: str | None                     # path serialized gating strategy file
    created_at: str | None = None        # ISO timestamp
    description: str | None = None       # human-readable description
    glm_version: str = field(default=GML_VERSION)  # GatingML version
    _graph: nx.DiGraph | None = field(repr=False, hash=False, default=None)

    def __post_init__(self) -> None:
        # check if id is valid filename
        if any(c in self.id for c in r'\/:*?"<>|'):
            raise ValueError(f"GatingStrategyRef id '{self.id}' contains invalid filename characters")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GatingStrategyRef":
        """
        Create a GatingStrategyRef from a catalog record (JSON dict).

        Parameters
        ----------
        data : Mapping[str, Any]
            Dictionary of serialized the gating strategy data

        Returns
        -------
        GatingStrategyRef
            Deserialized GatingStrategyRef instance
        """
        for f in ["id", "name", "batch_id"]:
            if f not in data:
                raise KeyError(f"Missing required field '{f}' in GatingStrategyRef data")

        graph_data = data.get("graph")
        graph = json_graph.node_link_graph(graph_data) if graph_data else None

        ref = GatingStrategyRef(
            id=data["id"],
            name=data["name"],
            batch_id=data["batch_id"],
            path=data.get("path"),
            created_at=data.get("created_at"),
            description=data.get("description"),
            glm_version=data.get("glm_version", GML_VERSION),
            _graph=graph,
        )

        return ref

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize GatingStrategyRef to a dictionary for JSON storage.

        Returns
        -------
        dict[str, Any]
            Serialized dictionary representation
        """
        return {
            "id": self.id,
            "name": self.name,
            "batch_id": self.batch_id,
            "path": self.path,
            "created_at": self.created_at or now_iso(),
            "description": self.description,
            "glm_version": self.glm_version,
            "graph": json_graph.node_link_data(self._graph) if self._graph else None,
        }

    @property
    def graph(self) -> nx.DiGraph:
        """
        Get the GatingStrategy instance associated with this reference.

        Returns
        -------
        nx.DiGraph
            The gating strategy instance

        Raises
        ------
        ValueError
            If the gating strategy instance is not loaded
        """
        if isinstance(self._graph, nx.DiGraph):
            return self._graph

        if self._graph is not None:
            raise TypeError("Gating strategy instance is of invalid type.")

        if self.path is None:
            raise ValueError("Gating strategy path is not set and instance is not set.")

        path = Path(self.path)
        if not path.exists():
            raise FileNotFoundError(f"Gating strategy file not found: {self.path} and instance is not set.")

        try:
            graph_data = json.loads(path.read_text())["graph"]
        except KeyError:
            raise ValueError(f"Gating strategy file {self.path} does not contain graph data.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding gating strategy file {self.path}: {e}") from e

        try:
            self._graph = json_graph.node_link_graph(data=graph_data)
        except Exception as e:
            raise ValueError(f"Error loading gating strategy graph from file {self.path}: {e}") from e

        return self._graph # pyright: ignore[reportReturnType]

    def init_graph(self) -> None:
        """
        Initialize an empty gating strategy graph with a root node.
        """

        self._graph = nx.DiGraph()
        self._graph.add_node("root")

    def iter_nodes(self) -> Iterable[GateNode]:
        """
        Iterate over all GateNode instances in the gating strategy graph.

        Yields
        ------
        Iterable[GateNode]
            An iterator over GateNode instances
        """

        for node_id, node_data in self.graph.nodes(data=True):
            if node_id == "root":
                continue
            if node_data:
                node = GateNode.from_dict(node_data)
                node.parent_ids = list(self.graph.predecessors(node_id))
                yield node

    def iter_descendants(self, root: str = "root") -> Iterable[str]:
        """
        Iterate over descendants of a node in topological order.

        Parameters
        ----------
        root : str
            The node ID whose descendants should be iterated.

        Yields
        ------
        Iterable[str]
            Descendant node IDs in topological order.

        Raises
        ------
        KeyError
            If the root node does not exist in the gating strategy.
        """

        if not self.graph.has_node(root):
            raise KeyError(f"Gate node with id '{root}' not found in gating strategy.")

        descendants = nx.descendants(self.graph, root)
        if not descendants:
            return

        descendant_graph = nx.DiGraph(self.graph.subgraph(descendants))
        for node_id in nx.topological_sort(descendant_graph):
            yield node_id

    def get_node(self, node_id: str) -> GateNode:
        """
        Get a GateNode by its ID.

        Parameters
        ----------
        node_id : str
            The ID of the gate node to retrieve.

        Returns
        -------
        GateNode
            The GateNode instance with the specified ID.

        Raises
        ------
        KeyError
            If the node with the specified ID does not exist.
        """

        if not self.graph.has_node(node_id):
            raise KeyError(f"Gate node with id '{node_id}' not found in gating strategy.")

        node_data = self.graph.nodes[node_id]
        # Add id back since it was popped during add_node()
        node_data = {**node_data, "id": node_id}
        node = GateNode.from_dict(node_data)
        parent_ids = list(self.graph.predecessors(node_id))
        try:
            parent_ids.remove("root")
        except ValueError:
            pass
        node.parent_ids = parent_ids
        return node

    def add_node(self, gate_node: GateNode | dict[str, Any]) -> None:
        """
        Add a GateNode to the gating strategy graph.

        Parameters
        ----------
        gate_node : GateNode | dict[str, Any]
            The GateNode instance to add.
        """

        if isinstance(gate_node, GateNode):
            gate_node = gate_node.to_dict()

        # Make sure all parents exist
        parents = gate_node.pop("parent_ids") or ["root"]
        for parent_id in parents:
            if not self.graph.has_node(parent_id):
                raise KeyError(f"Parent gate node with id '{parent_id}' not found in gating strategy.")

        node_id = gate_node.pop("id")

        # Check if node already exists
        if self.graph.has_node(node_id):
            raise ValueError(f"Gate node with id '{node_id}' already exists in the gating strategy.")

        self.graph.add_node(node_id, **gate_node)

        # Add edges and check for cycles
        for parent_id in parents:
            self.graph.add_edge(parent_id, node_id)

        # Check if the new node creates a cycle by looking for a path back to any parent
        for parent_id in parents:
            if nx.has_path(self.graph, node_id, parent_id):
                self.graph.remove_node(node_id)
                raise ValueError(
                    f"Adding gate '{node_id}' with parents {parents} would create a cycle in the gating strategy. "
                    "The gating strategy must remain a directed acyclic graph (DAG)."
                )

    def has_path(self, from_node_id: str, to_node_id: str) -> bool:
        """
        Check if there is a path from one node to another in the gating strategy graph.

        Parameters
        ----------
        from_node_id : str
            The ID of the starting node.
        to_node_id : str
            The ID of the target node.

        Returns
        -------
        bool
            True if there is a path from from_node_id to to_node_id, False otherwise.
        """

        return nx.has_path(self.graph, from_node_id, to_node_id)