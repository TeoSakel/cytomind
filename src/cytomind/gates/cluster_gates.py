"""Cluster gates that fit multiple coupled clusters at once."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Mapping, Hashable, Sequence, Iterator, TYPE_CHECKING
import pickle

import anndata as ad
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import BisectingKMeans, KMeans, MeanShift
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

from cytomind.domain.constants import FloatArray

from . import GateRegistry
from .base import Gate
from cytomind.domain.gates import GateNode

if TYPE_CHECKING:
    from pandas import DataFrame
    from cytomind.domain.constants import BooleanArray, FloatArray, IntArray
else:
    DataFrame = object
    BooleanArray = object
    FloatArray = object
    IntArray = object


class ClusterGate(Gate):
    """Abstract gate family that fits multiple coupled clusters."""

    gate_type: str = "cluster"
    default_tunable: bool = True
    model_type: Any

    def __init__(
        self,
        gate_name: str,
        dimensions: Sequence[str],
        hyperparams: Mapping[str, Any] | None = None,
        use_as_complement: bool = False,
        tunable: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Generic Cluster Gate to generate multiple gates based on clustering

        Args:
            gate_name (str): name of parent gate
            dimensions (Sequence[str]): Dimensions to be used for clustering analysis
            use_as_complement (bool, optional): Whether to use the gate as a complement. Defaults to False.
            tunable (bool | None, optional): Whether the gate is tunable. Defaults to None.
            kwargs (Any): Additional keyword arguments for specific cluster gate implementations.

        Raises:
            ValueError: If dimensions are empty
            ValueError: If use_as_complement is True
        """
        if len(dimensions) < 1:
            raise ValueError("ClusterGate requires at least 1 dimension.")
        if use_as_complement:
            raise ValueError("Cluster gates do not support use_as_complement.")

        hyperparams = dict(hyperparams) if hyperparams is not None else {}
        hyperparams.update(kwargs)

        super().__init__(
            gate_name,
            dimensions,
            hyperparams=hyperparams,
            use_as_complement=False,
            tunable=tunable,
        )

        self.state: dict[str, Any] = {
            "model": None,   # The fitted clustering model (e.g., GaussianMixture instance)
            "labels": None,  # Cluster labels for each event seen during fitting
            "clusters": [],  # List of cluster metadata dicts with keys "id", "name", and "params"
        }

    @property
    def model(self) -> Any:
        return self.state.get("model")

    @property
    def clusters(self) -> list[dict[str, Any]]:
        return self.state.get("clusters", [])

    @property
    def n_clusters(self) -> int:
        if self.model is not None:
            if hasattr(self.model, "n_clusters"):
                return int(self.model.n_clusters)
            if hasattr(self.model, "n_components"):
                return int(self.model.n_components)
        return len(self.clusters)

    def validate_params(self, params: Mapping[str, Any]) -> bool:
        return isinstance(params, Mapping)

    def _cluster_id(self, cluster_index: int) -> str:
        if cluster_index < len(self.clusters):
            return str(self.clusters[cluster_index]["id"])
        return f"{self.gate_name}_cluster_{cluster_index}"

    def _cluster_name(self, cluster_index: int) -> str:
        if cluster_index < len(self.clusters):
            return str(self.clusters[cluster_index]["name"])
        return f"{self.gate_name} cluster {cluster_index}"

    def _cluster_params(self, cluster_index: int) -> dict[str, Any]:
        return self.state["clusters"][cluster_index]["params"]

    def _child_gate_type(self, cluster: Mapping[str, Any]) -> str:
        return "cluster_component"

    # ---- Learning Phase ---- #

    def _fit_gate(self, events_slice: DataFrame) -> None:
        data = np.asarray(events_slice.to_numpy(dtype=np.float32, copy=False), dtype=np.float32)
        if data.shape[0] == 0:
            raise ValueError("Cannot fit a cluster gate on an empty event table.")

        self.state["model"], self.state["labels"] = self._fit_cluster_model(data)
        self.state["clusters"] = list(self._extract_clusters_from_model(data))
        self.params = self._extract_model_params()
        self.diagnostics = self._extract_fit_diagnostics()

    def _fit_cluster_model(self, data: FloatArray) -> tuple[Any, IntArray]:
        model = self.model_type(**self.hyperparams)
        labels = model.fit_predict(data)
        return model, labels

    @abstractmethod
    def _extract_clusters_from_model(self, data: FloatArray) -> Iterator[dict[str, Any]]:
        pass

    def _extract_model_params(self) -> dict[str, Any]:
        return {}

    def _extract_fit_diagnostics(self) -> dict[str, Any]:
        return {}

    def _export_json_state_payload(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.state.items()
            if key not in self._binary_state_artifacts()
        } | {"tunable": self.tunable}

    def _binary_state_artifacts(self) -> dict[str, str]:
        return {"model": "model.pkl"}

    def export_state(self) -> dict[str, Any]:
        bundle = super().export_state()
        manifest = dict(bundle["manifest"])
        artifact_map = dict(manifest.get("artifacts", {}))
        artifacts = dict(bundle["artifacts"])

        for state_key, filename in self._binary_state_artifacts().items():
            obj = self.state.get(state_key)
            if obj is None:
                continue
            artifact_map[state_key] = filename
            artifacts[filename] = pickle.dumps(obj)

        manifest["serializer"] = "hybrid"
        manifest["artifacts"] = artifact_map
        return {
            "manifest": manifest,
            "artifacts": artifacts,
        }

    def import_state(
        self,
        manifest: Mapping[str, Any],
        artifacts: Mapping[str, Any],
    ) -> None:
        super().import_state(manifest, artifacts)
        artifact_map = manifest.get("artifacts", {})
        for state_key, filename in self._binary_state_artifacts().items():
            artifact_name = str(artifact_map.get(state_key, filename))
            raw = artifacts.get(state_key)
            if raw is None:
                raw = artifacts.get(artifact_name)
            if raw is None:
                continue
            if not isinstance(raw, (bytes, bytearray)):
                raise ValueError(f"Cluster state artifact '{artifact_name}' must be binary.")
            self.state[state_key] = pickle.loads(bytes(raw))

    def generate_offsprings(
        self,
        *,
        parent_node: GateNode,
        sample_overrides: Mapping[str, dict[str, Any]] | None = None,
    ) -> list[GateNode]:
        sample_overrides = sample_overrides or {}
        child_nodes: list[GateNode] = []
        try:
            clusters = self.clusters
        except ValueError:
            clusters = []
            for sample_params in sample_overrides.values():
                sample_clusters = sample_params.get("state", {}).get("clusters", [])
                if isinstance(sample_clusters, list) and sample_clusters:
                    clusters = sample_clusters
                    break

        for cluster in clusters:
            cluster_id = str(cluster["id"])
            custom_gates: dict[str, dict[str, Any]] = {}
            for sample_id, sample_params in sample_overrides.items():
                sample_clusters = sample_params.get("state", {}).get("clusters", [])
                for sample_cluster in sample_clusters:
                    if str(sample_cluster["id"]) == cluster_id:
                        custom_gates[sample_id] = {
                            "hyperparams": {},
                            "params": dict(sample_cluster["params"]),
                            "diagnostics": {},
                            "state": {},
                        }
                        break

            child_nodes.append(
                GateNode(
                    id=cluster_id,
                    gate_type=self._child_gate_type(cluster),
                    dimensions=list(self.dimensions),
                    layer=parent_node.layer,
                    name=str(cluster["name"]),
                    parent_ids=[parent_node.id],
                    use_as_complement=False,
                    params={
                        "hyperparams": {},
                        "params": dict(cluster["params"]),
                        "diagnostics": {},
                        "state": {},
                    },
                    custom_gates=custom_gates,
                )
            )

        return child_nodes

    # ---- Application Phase ---- #

    def _apply_gate(self, events_slice: DataFrame | None = None) -> dict[str, BooleanArray]:
        if events_slice is None and self.model is None:
            raise ValueError("Cannot apply an unfitted cluster gate without event data.")
        elif events_slice is None:
            labels = self.state["labels"]
        elif self.model is None:
            self._fit_gate(events_slice)
            labels = self.state["labels"]
        else:
            coords = events_slice[self.dimensions].to_numpy(dtype=np.float32, copy=False)
            labels = self.model.predict(coords)

        results: dict[str, BooleanArray] = {}
        union_mask = np.zeros(len(labels), dtype=np.bool_)
        for i, cluster in enumerate(self.clusters):
            cluster_mask = np.asarray(np.asarray(labels, dtype=np.int_) == i, dtype=np.bool_)
            results[str(cluster["id"])] = cluster_mask
            union_mask |= cluster_mask

        results[self.gate_name] = union_mask
        return results

    def score_clusters(self, data: FloatArray) -> dict[str, np.ndarray]:
        cluster_scores: FloatArray = self._score_clusters(data)
        scores: dict[str, FloatArray] = {}
        for cluster, cluster_score in enumerate(cluster_scores):
            max_other = np.max(np.delete(cluster_scores, cluster, axis=0), axis=0)
            score_rel = cluster_score - max_other
            if np.any(score_rel > 0):
                score_rel /=  np.max(score_rel)
            elif np.any(score_rel < 0):
                score_rel /= -np.min(score_rel)
            else:
                score_rel = np.zeros_like(score_rel, dtype=np.float32)
            scores[self._cluster_id(cluster)] = score_rel.astype(np.float32)
        return scores

    def _score_gate(self, events_slice: DataFrame) -> dict[str, FloatArray]:
        coords = events_slice[self.dimensions].to_numpy(dtype=np.float32, copy=False)
        cluster_scores = self.score_clusters(coords)
        if cluster_scores:
            union_scores = np.maximum.reduce([np.asarray(scores, dtype=np.float32) for scores in cluster_scores.values()])
        else:
            union_scores = np.zeros(len(coords), dtype=np.float32)
        return {
            self.gate_name: union_scores.astype(np.float32),
            **cluster_scores,
        }

    @abstractmethod
    def _score_clusters(self, data: FloatArray) -> FloatArray:
        pass


class CentroidClusterGate(ClusterGate):
    """Cluster gate family whose fitted state is fully described by centroids."""

    _param_names = ("cluster_centers_",)

    def _extract_clusters_from_model(self, data: FloatArray) -> Iterator[dict[str, Any]]:
        del data
        if self.model is None:
            raise RuntimeError("Model parameters cannot be extracted before the model is fitted.")
        for i, cluster_center in enumerate(self.model.cluster_centers_):
            yield {
                "id": self._cluster_id(i),
                "name": self._cluster_name(i),
                "params": {"cluster_centers_": np.asarray(cluster_center).tolist()},
            }

    def _extract_model_params(self) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model parameters cannot be extracted before the model is fitted.")
        return {
            "cluster_centers_": np.asarray(self.model.cluster_centers_).tolist(),
        }

    def _score_clusters(self, data: FloatArray) -> FloatArray:
        if self.model is None:
            raise RuntimeError("Cluster scores cannot be computed before the model is fitted.")
        centers = np.asarray(self.model.cluster_centers_, dtype=np.float32)
        coords = np.asarray(data, dtype=np.float32)
        deltas = coords[None, :, :] - centers[:, None, :]
        return np.asarray(-np.sum(deltas * deltas, axis=2), dtype=np.float32)


@GateRegistry.register("kmeans")
class KMeansClusterGate(CentroidClusterGate):
    """Cluster gate that fits KMeans centroids and emits all child records."""

    gate_type = "kmeans_cluster"
    model_type = KMeans

    def __init__(
        self,
        gate_name: str,
        dimensions: Sequence[str],
        hyperparams: Mapping[str, Any] | None = None,
        use_as_complement: bool = False,
        tunable: bool | None = None,
        n_clusters: int = 2,
        init: str | FloatArray = "k-means++",
        n_init: int | str = "auto",
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
        copy_x: bool = True,
        algorithm: str = "lloyd",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            gate_name=gate_name,
            dimensions=dimensions,
            hyperparams=hyperparams,
            use_as_complement=use_as_complement,
            tunable=tunable,
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            copy_x=copy_x,
            algorithm=algorithm,
        )

        if self.hyperparams["n_clusters"] < 1:
            raise ValueError("n_clusters must be at least 1.")
        if self.hyperparams["tol"] <= 0.0:
            raise ValueError("tol must be positive.")
        if self.hyperparams["max_iter"] < 1:
            raise ValueError("max_iter must be >= 1.")
        valid_algorithms = {"lloyd", "elkan"}
        if self.hyperparams["algorithm"] not in valid_algorithms:
            raise ValueError(f"algorithm must be one of {valid_algorithms}.")

    def _extract_fit_diagnostics(self) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model diagnostics cannot be extracted before the model is fitted.")
        return {
            "converged": bool(int(self.model.n_iter_) < int(self.model.max_iter)),
            "n_iter": int(self.model.n_iter_),
            "neg_inertia": -float(self.model.inertia_) / len(self.model.labels_),
        }


@GateRegistry.register("bisecting_kmeans")
class BisectingKMeansClusterGate(CentroidClusterGate):
    """Cluster gate that fits BisectingKMeans centroids and emits all child records."""

    gate_type = "bisecting_kmeans_cluster"
    model_type = BisectingKMeans

    def __init__(
        self,
        gate_name: str,
        dimensions: Sequence[str],
        hyperparams: Mapping[str, Any] | None = None,
        use_as_complement: bool = False,
        tunable: bool | None = None,
        n_clusters: int = 2,
        init: str | FloatArray = "random",
        n_init: int = 1,
        random_state: int | None = None,
        max_iter: int = 300,
        tol: float = 1e-4,
        copy_x: bool = True,
        algorithm: str = "lloyd",
        bisecting_strategy: str = "biggest_inertia",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            gate_name=gate_name,
            dimensions=dimensions,
            hyperparams=hyperparams,
            use_as_complement=use_as_complement,
            tunable=tunable,
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            random_state=random_state,
            max_iter=max_iter,
            tol=tol,
            copy_x=copy_x,
            algorithm=algorithm,
            bisecting_strategy=bisecting_strategy,
        )

        if self.hyperparams["n_clusters"] < 1:
            raise ValueError("n_clusters must be at least 1.")
        if self.hyperparams["n_init"] < 1:
            raise ValueError("n_init must be at least 1.")
        if self.hyperparams["tol"] <= 0.0:
            raise ValueError("tol must be positive.")
        if self.hyperparams["max_iter"] < 1:
            raise ValueError("max_iter must be >= 1.")
        valid_algorithms = {"lloyd", "elkan"}
        if self.hyperparams["algorithm"] not in valid_algorithms:
            raise ValueError(f"algorithm must be one of {valid_algorithms}.")
        valid_strategies = {"biggest_inertia", "largest_cluster"}
        if self.hyperparams["bisecting_strategy"] not in valid_strategies:
            raise ValueError(f"bisecting_strategy must be one of {valid_strategies}.")

    def _extract_fit_diagnostics(self) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model diagnostics cannot be extracted before the model is fitted.")
        return {
            "converged": None,
            "neg_inertia": -float(self.model.inertia_) / len(self.model.labels_),
        }


@GateRegistry.register("mean_shift")
class MeanShiftClusterGate(CentroidClusterGate):
    """Cluster gate that fits MeanShift centroids and emits all child records."""

    gate_type = "mean_shift_cluster"
    model_type = MeanShift

    def __init__(
        self,
        gate_name: str,
        dimensions: Sequence[str],
        hyperparams: Mapping[str, Any] | None = None,
        use_as_complement: bool = False,
        tunable: bool | None = None,
        bandwidth: float | None = None,
        seeds: FloatArray | None = None,
        bin_seeding: bool = False,
        min_bin_freq: int = 1,
        cluster_all: bool = True,
        n_jobs: int | None = None,
        max_iter: int = 300,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            gate_name=gate_name,
            dimensions=dimensions,
            hyperparams=hyperparams,
            use_as_complement=use_as_complement,
            tunable=tunable,
            bandwidth=bandwidth,
            seeds=seeds,
            bin_seeding=bin_seeding,
            min_bin_freq=min_bin_freq,
            cluster_all=cluster_all,
            n_jobs=n_jobs,
            max_iter=max_iter,
        )

        if self.hyperparams["bandwidth"] is not None and self.hyperparams["bandwidth"] <= 0.0:
            raise ValueError("bandwidth must be positive when provided.")
        if self.hyperparams["min_bin_freq"] < 1:
            raise ValueError("min_bin_freq must be at least 1.")
        if self.hyperparams["max_iter"] < 1:
            raise ValueError("max_iter must be >= 1.")

    def _extract_fit_diagnostics(self) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model diagnostics cannot be extracted before the model is fitted.")
        return {
            "converged": bool(int(self.model.n_iter_) < int(self.model.max_iter)),
            "n_iter": int(self.model.n_iter_),
        }


@GateRegistry.register("gaussian_mixture")
class GaussianMixtureGate(ClusterGate):
    """Cluster gate that fits a Gaussian mixture and emits all child records."""

    gate_type = "gaussian_mixture_cluster"
    model_type = GaussianMixture

    def __init__(
        self,
        gate_name: str,
        dimensions: Sequence[str],
        hyperparams: Mapping[str, Any] | None = None,
        use_as_complement: bool = False,
        tunable: bool | None = None,
        n_components: int = 1,
        init_params: str = "kmeans",
        covariance_type: str = "full",
        reg_covar: float = 1e-6,
        tol: float = 1e-3,
        max_iter: int = 100,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> None:

        super().__init__(
            gate_name=gate_name,
            dimensions=dimensions,
            hyperparams=hyperparams,
            use_as_complement=use_as_complement,
            tunable=tunable,
            n_components=n_components,
            init_params=init_params,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
        )

        if self.hyperparams["n_components"] < 1:
            raise ValueError("n_components must be at least 1.")
        valid_cov_types = {"full", "tied", "diag", "spherical"}
        if self.hyperparams["covariance_type"] not in valid_cov_types:
            raise ValueError(f"covariance_type must be one of {valid_cov_types}.")
        if self.hyperparams["tol"] <= 0.0:
            raise ValueError("tol must be positive.")
        if self.hyperparams["reg_covar"] < 0.0:
            raise ValueError("reg_covar must be non-negative.")
        if self.hyperparams["max_iter"] < 1:
            raise ValueError("max_iter must be >= 1.")
        valid_init_params = {"kmeans", "random", "k-means++", "random_from_data"}
        if self.hyperparams["init_params"] not in valid_init_params:
            raise ValueError(f"init_params must be one of {valid_init_params}.")

        self._param_names = ("weights_", "means_", "covariances_", "precisions_cholesky_")

    def _child_gate_type(self, cluster: Mapping[str, Any]) -> str:
        return "gaussian_component"

    def _extract_clusters_from_model(self, data: FloatArray) -> Iterator[dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("Model parameters cannot be extracted before the model is fitted.")
        for i, cluster_params in enumerate(zip(*self.model._get_parameters())):
            yield {
                "id": self._cluster_id(i),
                "name": self._cluster_name(i),
                "params": dict(zip(self._param_names, map(lambda x: x.tolist(), cluster_params))),
            }

    def _extract_model_params(self) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model parameters cannot be extracted before the model is fitted.")
        return {
            param: getattr(self.model, param).tolist() if hasattr(self.model, param) else None
            for param in self._param_names
        }

    def _extract_fit_diagnostics(self) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model diagnostics cannot be extracted before the model is fitted.")
        return {
            "converged": bool(self.model.converged_),
            "n_iter": int(self.model.n_iter_),
            "lower_bounds": [-float(lb) for lb in self.model.lower_bounds_]
        }

    def _score_clusters(self, data: FloatArray) -> FloatArray:
        if self.model is None:
            raise RuntimeError("Cluster scores cannot be computed before the model is fitted.")
        return np.log(self.model.predict_proba(data)).T


@GateRegistry.register("bayesian_gaussian_mixture")
class BayesianGaussianMixtureClusterGate(GaussianMixtureGate):
    """Cluster gate that fits a Bayesian Gaussian mixture and emits all child records."""

    gate_type = "bayesian_gaussian_mixture_cluster"
    model_type = BayesianGaussianMixture

    def __init__(
        self,
        gate_name: str,
        dimensions: Sequence[str],
        hyperparams: Mapping[str, Any] | None = None,
        use_as_complement: bool = False,
        tunable: bool | None = None,
        n_components: int = 1,
        init_params: str = "kmeans",
        covariance_type: str = "full",
        reg_covar: float = 1e-6,
        tol: float = 1e-3,
        max_iter: int = 100,
        weight_concentration_prior_type: str = 'dirichlet_process',
        weight_concentration_prior: float | None = None,
        mean_precision_prior: float | None = None,
        mean_prior: FloatArray | None = None,
        degrees_of_freedom_prior: float | None = None,
        covariance_prior: FloatArray | None = None,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            gate_name=gate_name,
            dimensions=dimensions,
            hyperparams=hyperparams,
            use_as_complement=use_as_complement,
            tunable=tunable,
            n_components=n_components,
            init_params=init_params,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
            weight_concentration_prior_type=weight_concentration_prior_type,
            weight_concentration_prior=weight_concentration_prior,
            mean_precision_prior=mean_precision_prior,
            mean_prior=mean_prior,
            degrees_of_freedom_prior=degrees_of_freedom_prior,
            covariance_prior=covariance_prior,
        )
        valid_concentration_prior_types = {"dirichlet_process", "dirichlet_distribution"}
        if self.hyperparams["weight_concentration_prior_type"] not in valid_concentration_prior_types:
            raise ValueError(f"weight_concentration_prior_type must be one of {valid_concentration_prior_types}.")

        self._param_names = (
            "weight_concentration_", "weights_",
            "mean_precision_", "means_",
            "degrees_of_freedom_", "covariances_", "precisions_cholesky_"
        )

GateRegistry.register("gaussian_mixture_cluster")(GaussianMixtureGate)
GateRegistry.register("bayesian_gaussian_mixture_cluster")(BayesianGaussianMixtureClusterGate)
GateRegistry.register("kmeans_cluster")(KMeansClusterGate)
GateRegistry.register("bisecting_kmeans_cluster")(BisectingKMeansClusterGate)
GateRegistry.register("mean_shift_cluster")(MeanShiftClusterGate)


class ClusterComponentGate(Gate):
    """Passive child gate reconstructed from a fitted cluster parent."""

    gate_type = "cluster_component"
    default_tunable = False

    def __init__(
        self,
        gate_name: str,
        dimensions: Sequence[str],
        hyperparams: Mapping[str, Any] | None = None,
        use_as_complement: bool = False,
        tunable: bool | None = None,
        **kwargs: Any,
    ) -> None:
        if use_as_complement:
            raise ValueError("Cluster component gates do not support use_as_complement.")

        super().__init__(
            gate_name=gate_name,
            dimensions=dimensions,
            hyperparams=hyperparams,
            use_as_complement=False,
            tunable=tunable,
            **kwargs,
        )

    def validate_params(self, params: Mapping[str, Any]) -> bool:
        return isinstance(params, Mapping)

    def _fit_gate(self, events_slice: DataFrame) -> None:
        del events_slice
        raise RuntimeError(f"{self.__class__.__name__} cannot be fit independently.")

    def _apply_gate(self, events_slice: DataFrame) -> dict[str, BooleanArray]:
        del events_slice
        raise RuntimeError(f"{self.__class__.__name__} cannot be applied independently.")


@GateRegistry.register("gaussian_component")
class GaussianComponentGate(ClusterComponentGate):
    """Child gate representing a single Gaussian component from a GaussianMixtureGate."""

    gate_type = "gaussian_component"
    default_tunable = False

    def __init__(
        self,
        gate_name: str,
        dimensions: Sequence[str],
        hyperparams: Mapping[str, Any] | None = None,
        use_as_complement: bool = False,
        tunable: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(gate_name, dimensions, hyperparams, use_as_complement, tunable, **kwargs)

    def validate_params(self, params: Mapping[str, Any]) -> bool:
        return (
            super().validate_params(params)
            and "means_" in params
            and "covariances_" in params
        )

    def _score_gate(self, events_slice: DataFrame) -> dict[str, FloatArray]:
        means = np.asarray(self.params.get("means_"))
        cov = np.asarray(self.params.get("covariances_"))
        if means is None or cov is None:
            raise ValueError("Gaussian component parameters are missing for scoring.")
        if means.ndim == 0 or cov.ndim == 0:
            raise ValueError("Gaussian component parameters are malformed for scoring.")
        n_dims = len(self.dimensions)
        if n_dims == 1:
            means = means.reshape(1, 1) if means.ndim == 0 else means.reshape(1, -1)
        elif means.ndim == 1:
            means = means.reshape(1, -1)
        if n_dims != means.shape[1]:
            raise ValueError("Dimensionality of data does not match Gaussian component parameters.")
        from scipy.stats import multivariate_normal, norm

        coords = events_slice[self.dimensions].to_numpy(dtype=np.float32, copy=False)
        if n_dims > 1:
            scores: FloatArray = multivariate_normal.logpdf(coords, mean=np.ravel(means), cov=cov) # pyright: ignore[reportArgumentType]
        else:
            scores: FloatArray = norm.logpdf(
                coords[:, 0],
                loc=float(np.ravel(means)[0]),
                scale=float(np.sqrt(np.ravel(cov)[0])),
            )
        scores -= np.min(scores)  # Shift scores to have a minimum of 0
        scores /= np.max(scores)  # Normalize scores to have a maximum of 1
        return {self.gate_name: scores.astype(np.float32)}
