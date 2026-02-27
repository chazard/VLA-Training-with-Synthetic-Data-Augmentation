# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Abstract base for env configs that support feasibility checks and settable parameters for search (e.g. CEM)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

# Type for settable range parameters: event term -> list of (key_sequence, range_tuple).
# Order of keys and list order are fixed so callers can pass amended ranges back consistently.
SettableParamsT = dict[str, list[tuple[list[str], tuple[float, float]]]]


class SearchableEnvironmentCfg(ABC):
    """
    Abstract base for env configs that support domain search: settable distribution
    parameters, active event terms, feasibility checks, and introspection of
    settable parameters (e.g. for CEM or other search over domain randomization).
    """

    @abstractmethod
    def set_distribution_parameters(self, params: SettableParamsT) -> None:
        """Set parameter ranges from the same structure as get_settable_parameters_for_active_events.

        Args:
            params: Map from event term name to list of (key_sequence, range_tuple).
                Same format and ordering as returned by get_settable_parameters_for_active_events.
        """
        ...

    @abstractmethod
    def set_active_event_terms(self, term_names: list[str]) -> None:
        """Change which event terms are active and rebuild the events object.

        Args:
            term_names: List of event term names to activate.
        """
        ...

    @abstractmethod
    def get_default_event_ranges(self) -> dict[str, Any]:
        """Return a copy of the default event parameter ranges for this env.
        Used by domain samplers so they can apply to any env that exposes this interface."""
        ...

    @abstractmethod
    def get_default_active_event_terms(self) -> list[str]:
        """Return the default list of active event term names for this env.
        Used by domain samplers so they can apply to any env that exposes this interface."""
        ...

    @abstractmethod
    def check_feasibility(
        self,
        distribution_parameters: SettableParamsT,
    ) -> bool:
        """Check feasibility constraints for the given distribution parameters.

        Args:
            distribution_parameters: Candidate set to check (same structure
                as returned by get_settable_parameters_for_active_events).

        Returns:
            True if the configuration is feasible, False otherwise.
        """
        ...

    @abstractmethod
    def get_settable_parameters_for_active_events(self) -> SettableParamsT:
        """Return current range parameters for each active event term in a stable order.

        Iterates active event terms and, for each, uses the term's key sequences into
        the default event ranges to resolve (key_sequence, range_tuple) pairs. Order of
        event terms and of key sequences per term is fixed so amended ranges can be
        passed back (e.g. to set_distribution_parameters or CEM) with matching ordering.

        Returns:
            Map from event term name to list of (key_sequence, range_tuple) in consistent
            order (same as ALL_CHANGEABLE_EVENT_TERMS / active term order).
        """
        ...
