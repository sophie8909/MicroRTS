from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import copy


@dataclass
class Individual:
    """
    Individual representation for static prompt evolution.

    The genome is represented as:
    - components: named prompt segments
    - component_order: the rendering order of the final static prompt

    Important:
    - Only the static prompt is evolved.
    - The dynamic game state is injected later during evaluation.
    """

    components: Dict[str, str]
    component_order: List[str]

    # Cached rendered prompt text.
    prompt_text: Optional[str] = None

    # Multi-objective values. Larger is always better.
    objectives: Optional[List[float]] = None

    # Evolution bookkeeping.
    evaluated: bool = False
    rank: Optional[int] = None
    crowding_distance: float = 0.0

    # Arbitrary metadata for logging and analysis.
    metadata: Dict[str, Any] = field(default_factory=dict)

    def render_prompt(self) -> str:
        """
        Render the static prompt from the ordered components.
        Empty or missing components are skipped.
        """
        # print("Rendering prompt from components:")
        # for key in self.component_order:
        #     if key in self.components and self.components[key].strip():
        #         print(f"  - {key}: {self.components[key]}")
        self.prompt_text = "\n\n".join(
            self.components[key]
            for key in self.component_order
            if key in self.components and self.components[key].strip()
        )
        # print(f"Rendered prompt text:\n{self.prompt_text}\n")
        return self.prompt_text

    def copy(self) -> "Individual":
        """
        Deep-copy the individual so that variation operators do not mutate in-place.
        """
        return Individual(
            components=copy.deepcopy(self.components),
            component_order=self.component_order[:],
            prompt_text=self.prompt_text,
            objectives=copy.deepcopy(self.objectives),
            evaluated=self.evaluated,
            rank=self.rank,
            crowding_distance=self.crowding_distance,
            metadata=copy.deepcopy(self.metadata),
        )
    


    