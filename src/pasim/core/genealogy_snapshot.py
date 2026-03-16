from dataclasses import dataclass, field
from typing import List, Tuple

from pasim.core.state import Material, Region, Script


@dataclass
class GenealogyNode:
    """Represents a single node in the genealogy graph with its demographic metadata."""

    instance_id: str
    witness_id: str
    manuscript_id: str
    birth_tick: int
    death_tick: int
    parent_ids: List[str]
    region: Region
    material: Material
    script: Script
    reputation: int
    location: Tuple[float, float]
    pa_intervention_regimes: List[str] = field(default_factory=list)


@dataclass
class GenealogySnapshot:
    """A complete, serialisable snapshot of the demographic simulation."""

    nodes: List[GenealogyNode] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Converts the snapshot to a JSON-serialisable dictionary."""
        return {
            "nodes": [
                {
                    "instance_id": node.instance_id,
                    "witness_id": node.witness_id,
                    "manuscript_id": node.manuscript_id,
                    "birth_tick": node.birth_tick,
                    "death_tick": node.death_tick,
                    "parent_ids": node.parent_ids,
                    "region": node.region.value,
                    "material": node.material.value,
                    "script": node.script.value,
                    "reputation": node.reputation,
                    "location": node.location,
                    "pa_intervention_regimes": node.pa_intervention_regimes,
                }
                for node in self.nodes
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GenealogySnapshot":
        """Creates a snapshot from a dictionary."""
        return cls(
            nodes=[
                GenealogyNode(
                    instance_id=node["instance_id"],
                    witness_id=node["witness_id"],
                    manuscript_id=node["manuscript_id"],
                    birth_tick=node["birth_tick"],
                    death_tick=node["death_tick"],
                    parent_ids=node["parent_ids"],
                    region=Region(node["region"]),
                    material=Material(node["material"]),
                    script=Script(node["script"]),
                    reputation=node["reputation"],
                    location=tuple(node["location"]),  # type: ignore
                    pa_intervention_regimes=node.get("pa_intervention_regimes", []),
                )
                for node in data["nodes"]
            ]
        )
