"""
Central parameter configuration for the İzmir Water Health Factor (HF).

The Health Factor is a safety-oriented transformation of the Weighted
Arithmetic Water Quality Index (WAWQI):

    HF = clip(100 - WAWQI, 0, 100)

Only parameters with a positive numeric standard (Sn > 0) participate in
WAWQI unit-weight calculation. Parameters whose standard is zero are handled
separately as binary fail-fast parameters; therefore no division by zero is
performed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ParameterKind = Literal[
    "numeric",
    "range",
    "zero_standard",
    "acceptable",
    "descriptive",
]


@dataclass(frozen=True)
class ParameterSpec:
    """Configuration for one canonical water-quality parameter."""

    name: str
    aliases: tuple[str, ...]
    unit: str | None
    standard: float | tuple[float, float] | None
    ideal: float | None
    kind: ParameterKind
    category: str
    fail_fast: bool = False
    include_in_wawqi: bool = False

    @property
    def weight_standard(self) -> float | None:
        """Positive Sn used in Wn = K / Sn.

        For pH, the manuscript defines a permissible range of 6.5–9.5 and an
        ideal value of 7.0. The upper permissible bound (9.5) is used for the
        unit-weight calculation, matching the previous WAWQI implementation.
        """

        if not self.include_in_wawqi:
            return None
        if isinstance(self.standard, tuple):
            return float(self.standard[1])
        if isinstance(self.standard, (int, float)) and self.standard > 0:
            return float(self.standard)
        return None


# Thresholds required by the manuscript/reviewer request.
HF_GOOD_MIN = 85.0
HF_CAUTION_MIN = 60.0
HF_MIN = 0.0
HF_MAX = 100.0
FAIL_FAST_HF = 0.0


PARAMETERS: dict[str, ParameterSpec] = {
    "E.coli": ParameterSpec(
        name="E.coli",
        aliases=("E.Coli", "E. coli", "E.COL", "ECOL", "E-COLI", "ECOLI"),
        unit="Sayı/100 ml",
        standard=0.0,
        ideal=0.0,
        kind="zero_standard",
        category="Microbiological",
        fail_fast=True,
    ),
    "Koliform Bakteri": ParameterSpec(
        name="Koliform Bakteri",
        aliases=("KOLIF", "Koliform", "Coliform Bacteria", "Coliform Bac."),
        unit="Sayı/100 ml",
        standard=0.0,
        ideal=0.0,
        kind="zero_standard",
        category="Microbiological",
        fail_fast=True,
    ),
    "C.Perfringens": ParameterSpec(
        name="C.Perfringens",
        aliases=(
            "C.perfringens",
            "C. perfringens",
            "C.PER",
            "C PERFRINGENS",
        ),
        unit="Sayı/100 ml",
        standard=0.0,
        ideal=0.0,
        kind="zero_standard",
        category="Microbiological",
        fail_fast=True,
    ),
    "Arsenik": ParameterSpec(
        name="Arsenik",
        aliases=("As", "Arsenic"),
        unit="μg/L",
        standard=10.0,
        ideal=0.0,
        kind="numeric",
        category="Toxic",
        fail_fast=True,
        include_in_wawqi=True,
    ),
    "Nitrit": ParameterSpec(
        name="Nitrit",
        aliases=("NO2", "NO₂", "Nitrite", "Nitrit (NO₂)"),
        unit="mg/L",
        standard=0.5,
        ideal=0.0,
        kind="numeric",
        category="Toxic",
        fail_fast=True,
        include_in_wawqi=True,
    ),
    "Amonyum": ParameterSpec(
        name="Amonyum",
        aliases=("NH4+", "NH₄⁺", "Ammonium", "Amonyum (NH₄⁺)"),
        unit="mg/L",
        standard=0.5,
        ideal=0.0,
        kind="numeric",
        category="Industrial",
        include_in_wawqi=True,
    ),
    "Alüminyum": ParameterSpec(
        name="Alüminyum",
        aliases=("Al", "Aluminum", "Aluminium"),
        unit="μg/L",
        standard=200.0,
        ideal=0.0,
        kind="numeric",
        category="Industrial",
        include_in_wawqi=True,
    ),
    "Demir": ParameterSpec(
        name="Demir",
        aliases=("Fe", "Iron"),
        unit="μg/L",
        standard=200.0,
        ideal=0.0,
        kind="numeric",
        category="Industrial",
        include_in_wawqi=True,
    ),
    "Klorür": ParameterSpec(
        name="Klorür",
        aliases=("Cl-", "Chloride", "Klorur"),
        unit="mg/L",
        standard=250.0,
        ideal=0.0,
        kind="numeric",
        category="Industrial",
        include_in_wawqi=True,
    ),
    "İletkenlik": ParameterSpec(
        name="İletkenlik",
        aliases=("ILETK", "Conductivity", "EC"),
        unit="μS/cm",
        standard=2500.0,
        ideal=0.0,
        kind="numeric",
        category="Industrial",
        include_in_wawqi=True,
    ),
    "Oksitlenebilirlik": ParameterSpec(
        name="Oksitlenebilirlik",
        aliases=("OKSIT", "Oxidizability", "Oxidability"),
        unit="mg/L O2",
        standard=5.0,
        ideal=0.0,
        kind="numeric",
        category="Industrial",
        include_in_wawqi=True,
    ),
    "pH": ParameterSpec(
        name="pH",
        aliases=("PH", "ph"),
        unit=None,
        standard=(6.5, 9.5),
        ideal=7.0,
        kind="range",
        category="Physical",
        include_in_wawqi=True,
    ),
    # These parameters are preserved and scored when an explicit textual
    # acceptability result exists, but they are not included in WAWQI because
    # the source data do not provide a positive numeric Sn suitable for K/Sn.
    "Bulanıklık": ParameterSpec(
        name="Bulanıklık",
        aliases=("BULAN", "Turbidity"),
        unit=None,
        standard=None,
        ideal=None,
        kind="acceptable",
        category="Physical/Aesthetic",
    ),
    "Tat": ParameterSpec(
        name="Tat",
        aliases=("TAT", "Taste"),
        unit=None,
        standard=None,
        ideal=None,
        kind="acceptable",
        category="Aesthetic",
    ),
    "Koku": ParameterSpec(
        name="Koku",
        aliases=("KOKU", "Odor", "Odour"),
        unit=None,
        standard=None,
        ideal=None,
        kind="acceptable",
        category="Aesthetic",
    ),
    "Renk": ParameterSpec(
        name="Renk",
        aliases=("RENK", "Color", "Colour"),
        unit=None,
        standard=None,
        ideal=None,
        kind="acceptable",
        category="Aesthetic",
    ),
    "Toplam Sertlik": ParameterSpec(
        name="Toplam Sertlik",
        aliases=("TOPLA", "Total Hardness"),
        unit="Fr",
        standard=None,
        ideal=None,
        kind="descriptive",
        category="Descriptive",
    ),
    "Tuzluluk": ParameterSpec(
        name="Tuzluluk",
        aliases=("TUZLU", "Salinity"),
        unit="%0",
        standard=None,
        ideal=None,
        kind="descriptive",
        category="Descriptive",
    ),
}


def _alias_key(value: str) -> str:
    return "".join(str(value).strip().casefold().replace("ı", "i").split())


_ALIAS_TO_CANONICAL: dict[str, str] = {}
for canonical_name, spec in PARAMETERS.items():
    for alias in (canonical_name, *spec.aliases):
        _ALIAS_TO_CANONICAL[_alias_key(alias)] = canonical_name


def canonical_parameter_name(value: object) -> str:
    """Map raw/legacy parameter names to the canonical names above."""

    if value is None:
        return ""
    text = str(value).strip()
    return _ALIAS_TO_CANONICAL.get(_alias_key(text), text)


def calculate_unit_weights() -> tuple[float, dict[str, float]]:
    """Calculate WAWQI proportionality constant K and unit weights Wn."""

    standards = {
        name: spec.weight_standard
        for name, spec in PARAMETERS.items()
        if spec.weight_standard is not None
    }
    if not standards:
        return 0.0, {}

    k = 1.0 / sum(1.0 / sn for sn in standards.values())
    weights = {name: k / sn for name, sn in standards.items()}
    return k, weights


WAWQI_K, UNIT_WEIGHTS = calculate_unit_weights()
WAWQI_PARAMETER_NAMES = tuple(UNIT_WEIGHTS.keys())
ZERO_STANDARD_PARAMETERS = tuple(
    name for name, spec in PARAMETERS.items() if spec.kind == "zero_standard"
)
FAIL_FAST_PARAMETERS = tuple(
    name for name, spec in PARAMETERS.items() if spec.fail_fast
)