from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


class Material(Enum):
    """
    Represents the material on which a manuscript is written.
    """

    PARCHMENT = "parchment"
    PAPYRUS = "papyrus"
    PAPER = "paper"


class Region(Enum):
    """
    Represents the geographical region where a manuscript is located or originated.
    """

    ASIA_MINOR = "Asia Minor"
    EGYPT = "Egypt"
    LEVANT = "Levant"


class Script(Enum):
    """
    Represents the script style used in a textual witness.
    """

    UNCIAL = "uncial"
    MINUSCULE = "minuscule"


class DeathReason(Enum):
    """
    Represents the explicit cause of a manuscript's death.
    """

    NATURAL = "natural"
    PERSECUTION = "persecution"
    CULTURAL_REPLACEMENT = "cultural_replacement"


@dataclass
class Manuscript:
    """
    Represents a physical manuscript within the simulation.

    A Manuscript has an identity, defined temporal bounds within the simulation's
    tick-based timeline, and mutable attributes describing its physical
    characteristics and geographical location.
    """

    manuscript_id: str
    """
    A unique identifier for the manuscript. This ID is immutable and serves
    to distinguish this manuscript from all others in the simulation.
    """

    birth_tick: int
    """
    The simulation tick at which this manuscript is 'born' or first appears.
    This marks the beginning of its existence in the simulation.
    """

    death_tick: int
    """
    The simulation tick at which this manuscript 'dies' or ceases to exist.
    This marks the end of its existence in the simulation.
    """

    material: Material = field()
    """
    The material on which the manuscript is written (e.g., parchment, papyrus, paper).
    This attribute cannot change over the lifespan of the manuscript.
    """

    region: Region = field()
    """
    The geographical region where the manuscript is currently located or originated.
    This attribute can change over time as manuscripts are moved or copied in
    different regions.
    """

    location: Tuple[float, float] = field()
    """
    The precise planar (x, y) coordinates representing the manuscript's
    geographical location. This attribute can change over time.
    """

    death_reason: Optional[DeathReason] = field(default=None)
    """
    The explicit reason for the manuscript's death, if applicable.
    This helps distinguish between natural lifecycle end and external events.
    """


@dataclass
class Witness:
    """
    Represents a textual witness, which is a specific copy of a text
    associated with a Manuscript.

    A Witness has its own identity and can carry arbitrary metadata, but it
    does not have independent temporal or spatial existence; its presence
    is tied to the Manuscript it belongs to via `manuscript_id`.
    """

    witness_id: str
    """
    A unique identifier for this textual witness. This ID distinguishes
    this particular copy of the text.
    """

    manuscript_id: str
    """
    The identifier of the `Manuscript` to which this witness belongs.
    This provides the link to the physical entity.
    """

    script: Script = field()
    """
    The script style used in this witness (e.g., uncial, minuscule).
    This attribute is immutable for the lifespan of the witness.
    """

    metadata: Dict[str, Any] = field(default_factory=dict)
    """

    An optional dictionary for storing arbitrary metadata associated with
    this specific witness. This can include information not captured
    by the core fields. Defaults to an empty dictionary.
    """


class ManuscriptRegistry:
    """
    Maintains a collection of all manuscripts in the simulation.

    This registry acts as an authoritative, in-memory store for Manuscript objects,
    indexed by their unique `manuscript_id`. It ensures that each manuscript
    is represented only once and provides a central point of access for querying
    what manuscripts exist within the simulation state.
    """

    def __init__(self) -> None:
        self._manuscripts: Dict[str, Manuscript] = {}

    def add(self, manuscript: Manuscript) -> None:
        """Adds a manuscript to the registry, enforcing ID uniqueness."""
        if manuscript.manuscript_id in self._manuscripts:
            raise KeyError(f"Duplicate manuscript_id: {manuscript.manuscript_id}")
        self._manuscripts[manuscript.manuscript_id] = manuscript

    def get(self, manuscript_id: str) -> Manuscript:
        """Retrieves a manuscript by its ID."""
        return self._manuscripts[manuscript_id]

    def __contains__(self, manuscript_id: str) -> bool:
        """Checks if a manuscript ID exists in the registry."""
        return manuscript_id in self._manuscripts

    def __len__(self) -> int:
        """Returns the total number of manuscripts in the registry."""
        return len(self._manuscripts)

    def items(self) -> Iterable[tuple[str, Manuscript]]:
        """Returns a view of the manuscript items (id, object)."""
        return self._manuscripts.items()


class WitnessRegistry:
    """
    Maintains a collection of all witnesses in the simulation.

    This registry stores Witness objects, indexed by `witness_id`, and ensures
    that every witness is valid by checking that its associated `manuscript_id`
    refers to a manuscript that exists in the ManuscriptRegistry.
    """

    def __init__(self, manuscript_registry: ManuscriptRegistry) -> None:
        self._witnesses: Dict[str, Witness] = {}
        self._manuscript_registry = manuscript_registry

    def add(self, witness: Witness) -> None:
        """
        Adds a witness, ensuring its ID is unique and its manuscript exists.
        """
        if witness.witness_id in self._witnesses:
            raise KeyError(f"Duplicate witness_id: {witness.witness_id}")
        if witness.manuscript_id not in self._manuscript_registry:
            raise ValueError(f"Witness {witness.witness_id} references non-existent manuscript_id: {witness.manuscript_id}")
        self._witnesses[witness.witness_id] = witness

    def get(self, witness_id: str) -> Witness:
        """Retrieves a witness by its ID."""
        return self._witnesses[witness_id]

    def __contains__(self, witness_id: str) -> bool:
        """Checks if a witness ID exists in the registry."""
        return witness_id in self._witnesses

    def __len__(self) -> int:
        """Returns the total number of witnesses."""
        return len(self._witnesses)

    def items(self) -> Iterable[tuple[str, Witness]]:
        """Returns a view of the witness items (id, object)."""
        return self._witnesses.items()


@dataclass
class StateRegistry:
    """
    A container for the simulation's core identity registries.

    This class holds the authoritative registries for all manuscripts and
    witnesses, providing a single, consistent snapshot of what entities
    exist in the simulation. It has no simulation logic itself but serves
    as the foundational context for the simulation model.
    """

    manuscripts: ManuscriptRegistry = field(default_factory=ManuscriptRegistry)
    witnesses: WitnessRegistry = field(init=False)
    instance_texts: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.witnesses = WitnessRegistry(self.manuscripts)
