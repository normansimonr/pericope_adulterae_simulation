from enum import Enum
from dataclasses import dataclass, field
from typing import Tuple, Any, Dict

class Material(Enum):
    """
    Represents the material on which a manuscript is written.
    """
    PARCHMENT = "parchment"
    PAPURUS = "papyrus"
    PAPER = "paper"

class Area(Enum):
    """
    Represents the geographical area where a manuscript is located or originated.
    """
    ASIA_MINOR = "AsiaMinor"
    EGYPT = "Egypt"
    LEVANT = "Levant"

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

    area: Area = field()
    """
    The geographical area where the manuscript is currently located or originated.
    This attribute can change over time as manuscripts are moved or copied in
    different regions.
    """

    location: Tuple[float, float] = field()
    """
    The precise planar (x, y) coordinates representing the manuscript's
    geographical location. This attribute can change over time.
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

    metadata: Dict[str, Any] = field(default_factory=dict)
    """
    An optional dictionary for storing arbitrary metadata associated with
    this specific witness. This can include information not captured
    by the core fields. Defaults to an empty dictionary.
    """
