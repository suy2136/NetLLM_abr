"""Speculative inference helpers for adaptive bitrate streaming."""

from plm_special.speculative.mpc_draft import MPCDraftRollout, RobustMPCDraftGenerator
from plm_special.speculative.acceptance import (
    AcceptancePlan,
    ObservationValidation,
    build_acceptance_plan,
    validate_speculative_observation,
)

__all__ = [
    'AcceptancePlan', 'MPCDraftRollout', 'ObservationValidation',
    'RobustMPCDraftGenerator', 'build_acceptance_plan',
    'validate_speculative_observation',
]
