"""Burden-referenced efficacy assessment and allocation of independent equal-cost actions."""
from __future__ import annotations
__version__ = '0.1.0'
from dataclasses import dataclass
from math import isfinite, fsum
from numbers import Real
from typing import Mapping, Sequence

class InputError(ValueError):
    """Inputs do not meet the method's explicit data or feasibility requirements."""

def _number(value: Real, label: str, *, nonnegative: bool=False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or (not isfinite(value)):
        raise InputError(f'{label} must be a finite number')
    value = float(value)
    if nonnegative and value < 0:
        raise InputError(f'{label} must be nonnegative')
    return value

def absolute_benefit(burden: Real, fractional_response: Real) -> float:
    """Return M * E. Negative E represents an adverse effect; no silent clipping."""
    result = _number(burden, 'burden', nonnegative=True) * _number(fractional_response, 'fractional_response')
    return _number(result, 'absolute benefit')

def relative_efficacy(benefit: Real, burden: Real) -> float | None:
    """Return S / M as a fraction, or None for a zero denominator."""
    s = _number(benefit, 'benefit')
    m = _number(burden, 'burden', nonnegative=True)
    return _number(s / m, 'relative efficacy') if m else None

@dataclass(frozen=True)
class Outcome:
    """One problem's burden and incremental benefit at one spatial unit.

    Both quantities must use the same endpoint, time interval and physical unit.
    Equity weights affect the allocation objective, never reported physical totals.
    """
    burden: float
    benefit: float
    equity_weight: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, 'burden', _number(self.burden, 'burden', nonnegative=True))
        object.__setattr__(self, 'benefit', _number(self.benefit, 'benefit'))
        object.__setattr__(self, 'equity_weight', _number(self.equity_weight, 'equity_weight', nonnegative=True))

    @classmethod
    def from_fraction(cls, burden: float, fractional_response: float, equity_weight: float=1.0) -> Outcome:
        """Convert a local fractional response to an absolute incremental benefit."""
        return cls(burden, absolute_benefit(burden, fractional_response), equity_weight)

@dataclass(frozen=True)
class Candidate:
    """One independent, equal-cost action at a distinct spatial unit.

    Exact score ties use ascending tie_order, then ascending string id. Explicit
    numeric tie_order makes ordering reproducible for labels such as '2' and '10'.
    """
    id: str
    outcomes: Mapping[str, Outcome]
    tie_order: int = 0

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id.strip():
            raise InputError('candidate id must be a nonempty string')
        if type(self.tie_order) is not int:
            raise InputError('tie_order must be an integer')
        if not self.outcomes or any((not isinstance(k, str) or not k.strip() for k in self.outcomes)):
            raise InputError('outcomes must have nonempty domain names')
        if any((not isinstance(o, Outcome) for o in self.outcomes.values())):
            raise InputError('each outcome must be an Outcome')
        object.__setattr__(self, 'outcomes', dict(self.outcomes))

@dataclass(frozen=True)
class Problem:
    """Comparable outcomes for all candidates, plus any non-candidate burden.

    Spatial units must not overlap. background_burden represents burden outside
    the candidate units and contributes once to the whole-area denominator.
    """
    candidates: Sequence[Candidate]
    units: Mapping[str, str]
    background_burden: Mapping[str, float] | None = None

    def __post_init__(self):
        candidates = tuple(self.candidates)
        if not candidates or any((not isinstance(c, Candidate) for c in candidates)):
            raise InputError('provide at least one Candidate')
        if len({c.id for c in candidates}) != len(candidates):
            raise InputError('candidate ids must be unique')
        units = dict(self.units)
        if not units or any((not isinstance(k, str) or not k.strip() or (not isinstance(v, str)) or (not v.strip()) for k, v in units.items())):
            raise InputError('units must specify a nonempty unit for every domain')
        if any((set(c.outcomes) != set(units) for c in candidates)):
            raise InputError('every candidate must explicitly contain every domain')
        background = dict(self.background_burden or {})
        if set(background) - set(units):
            raise InputError('background_burden contains an unknown domain')
        background = {d: _number(background.get(d, 0), f'background_burden[{d}]', nonnegative=True) for d in units}
        object.__setattr__(self, 'candidates', candidates)
        object.__setattr__(self, 'units', units)
        object.__setattr__(self, 'background_burden', background)

    @property
    def domains(self) -> tuple[str, ...]:
        return tuple(self.units)

    def total_burden(self, domain: str) -> float:
        _domain(self, domain)
        return fsum([self.background_burden[domain], *(c.outcomes[domain].burden for c in self.candidates)])

def _domain(problem: Problem, domain: str) -> None:
    if domain not in problem.units:
        raise InputError(f'unknown domain: {domain}')

def benefit_scores(problem: Problem, domain: str, *, equity: bool=True) -> dict[str, float]:
    """Within-problem priorities: q_i * S_i, or S_i when equity=False."""
    _domain(problem, domain)
    return {c.id: _number(c.outcomes[domain].benefit * (c.outcomes[domain].equity_weight if equity else 1), 'score') for c in problem.candidates}

def fraction_scores(problem: Problem, domain: str) -> dict[str, float]:
    """Local S_i/M_i comparator. Undefined fractions raise rather than rank as zero."""
    _domain(problem, domain)
    scores = {}
    for c in problem.candidates:
        o = c.outcomes[domain]
        value = relative_efficacy(o.benefit, o.burden)
        if value is None:
            raise InputError(f'local fraction is undefined for zero burden: {c.id}/{domain}')
        scores[c.id] = value
    return scores

def composite_scores(problem: Problem, weights: Mapping[str, float]) -> dict[str, float]:
    """Comparator: sum_k w_k S_ik / max_j(S_jk).

    Weights must be nonnegative and sum to one (within 1e-10). Extrema are computed
    over the full input candidate set before feasibility filtering. Zero-valued
    layers contribute zero; a nonzero layer whose maximum is <= 0 is rejected,
    because candidate-maximum scaling would be undefined or reverse its meaning.
    No burden or equity multiplier is applied here: S already contains M * E.
    """
    if not weights or set(weights) - set(problem.domains):
        raise InputError('provide weights for one or more known domains')
    weights = {d: _number(w, f'weight[{d}]', nonnegative=True) for d, w in weights.items()}
    if abs(fsum(weights.values()) - 1) > 1e-10:
        raise InputError('composite weights must sum to one')
    result = {c.id: 0.0 for c in problem.candidates}
    for d, w in weights.items():
        if w == 0:
            continue
        values = [c.outcomes[d].benefit for c in problem.candidates]
        maximum = max(values)
        if all((v == 0 for v in values)):
            continue
        if maximum <= 0:
            raise InputError(f'candidate-maximum normalization requires a positive maximum: {d}')
        for c in problem.candidates:
            result[c.id] += w * c.outcomes[d].benefit / maximum
    return {k: _number(v, 'composite score') for k, v in result.items()}

def select(problem: Problem, scores: Mapping[str, float], budget: int, *, minimum_benefits: Mapping[str, float] | None=None, fill_budget: bool=True) -> tuple[str, ...]:
    """Rank independent equal-cost actions under a cardinality budget.

    A minimum is a per-action constraint in the named domain's physical units;
    {'air': 0} excludes actions that worsen air. Exactly budget actions are
    selected by default (including nonpositive scores). fill_budget=False chooses
    only strictly positive scores, up to budget. Feasibility ties are never random.
    """
    if type(budget) is not int or not 0 <= budget <= len(problem.candidates):
        raise InputError('budget must be an integer between zero and the candidate count')
    if type(fill_budget) is not bool:
        raise InputError('fill_budget must be a boolean')
    if set(scores) != {c.id for c in problem.candidates}:
        raise InputError('scores must contain exactly the candidate ids')
    scores = {k: _number(v, f'score[{k}]') for k, v in scores.items()}
    minimum = dict(minimum_benefits or {})
    if set(minimum) - set(problem.domains):
        raise InputError('minimum_benefits contains an unknown domain')
    minimum = {d: _number(v, f'minimum_benefit[{d}]') for d, v in minimum.items()}
    feasible = [c for c in problem.candidates if all((c.outcomes[d].benefit >= m for d, m in minimum.items()))]
    if fill_budget and len(feasible) < budget:
        raise InputError('too few feasible candidates to fill the requested budget')
    ranked = sorted(feasible, key=lambda c: (-scores[c.id], c.tie_order, c.id))
    if not fill_budget:
        ranked = [c for c in ranked if scores[c.id] > 0]
    return tuple((c.id for c in ranked[:budget]))

def evaluate(problem: Problem, selected: Sequence[str]) -> dict:
    """Report physical outcomes, S/M and spatial residuals, never mixed units.

    Benefits exceeding local burdens are reported without clipping. Residuals
    use max(0, M_i-S_i); excess beyond a burden is reported separately. Negative
    benefits increase residual burden. A zero total burden yields a null fraction.
    """
    chosen = tuple(selected)
    ids = {c.id for c in problem.candidates}
    if len(set(chosen)) != len(chosen) or set(chosen) - ids:
        raise InputError('selected ids must be unique and belong to the problem')
    chosen_set = set(chosen)
    result = {'selected_ids': list(chosen), 'selected_count': len(chosen), 'domains': {}}
    for d in problem.domains:
        increments = [(c, c.outcomes[d].benefit if c.id in chosen_set else 0.0) for c in problem.candidates]
        benefit = fsum((s for _, s in increments))
        burden = problem.total_burden(d)
        local = {c.id: max(0.0, c.outcomes[d].burden - s) for c, s in increments}
        residual = fsum([problem.background_burden[d], *local.values()])
        result['domains'][d] = {'unit': problem.units[d], 'total_burden': burden, 'benefit': benefit, 'fraction_of_total_burden': relative_efficacy(benefit, burden), 'residual_burden': residual, 'local_residuals': local, 'benefit_beyond_local_burden': fsum((max(0.0, s - c.outcomes[d].burden) for c, s in increments)), 'equity_weighted_objective': fsum((c.outcomes[d].equity_weight * s for c, s in increments))}
    return result

def role_potentials(problem: Problem, budget: int, *, minimum_benefits: Mapping[str, float] | None=None, fill_budget: bool=True) -> dict:
    """Independent physical-benefit optimum for each domain at a common budget.

    These separate counterfactual portfolios support role assessment. They cannot
    be combined as if simultaneously achievable, and no roles are assigned here.
    Equity weights are deliberately omitted from these physical maxima.
    """
    result = {}
    for d in problem.domains:
        chosen = select(problem, benefit_scores(problem, d, equity=False), budget, minimum_benefits=minimum_benefits, fill_budget=fill_budget)
        result[d] = {'selected_ids': list(chosen), **evaluate(problem, chosen)['domains'][d]}
    return result

def compare_portfolios(problem: Problem, reference: Sequence[str], alternative: Sequence[str]) -> dict:
    """Report overlap and domain-specific changes relative to the reference.

    Overlap fraction uses the reference portfolio size. Jaccard uses the union.
    Relative gain is null for nonpositive reference benefit; absolute differences
    remain available. Percentage-point differences equal 100 times share changes.
    """
    ref, alt = (evaluate(problem, reference), evaluate(problem, alternative))
    a, b = (set(reference), set(alternative))
    overlap = len(a & b)
    domains = {}
    for d in problem.domains:
        r, s = (ref['domains'][d], alt['domains'][d])
        share_r, share_s = (r['fraction_of_total_burden'], s['fraction_of_total_burden'])
        domains[d] = {'unit': problem.units[d], 'reference_benefit': r['benefit'], 'alternative_benefit': s['benefit'], 'benefit_difference': s['benefit'] - r['benefit'], 'relative_gain_percent': 100 * (s['benefit'] / r['benefit'] - 1) if r['benefit'] > 0 else None, 'burden_share_change_percentage_points': 100 * (share_s - share_r) if share_s is not None and share_r is not None else None}
    return {'shared_count': overlap, 'reference_count': len(a), 'alternative_count': len(b), 'overlap_fraction_of_reference': overlap / len(a) if a else None, 'jaccard_overlap': overlap / len(a | b) if a | b else None, 'domains': domains}
import csv
import json
from pathlib import Path

def _keys(obj, allowed, label):
    if not isinstance(obj, dict):
        raise InputError(f'{label} must be an object')
    unknown = set(obj) - set(allowed)
    if unknown:
        raise InputError(f"unknown {label} fields: {', '.join(sorted(unknown))}")

def problem_from_dict(data: dict) -> Problem:
    """Load format_version=1 data with an explicit benefit OR response per outcome."""
    _keys(data, {'format_version', 'units', 'background_burden', 'candidates'}, 'problem')
    if type(data.get('format_version')) is not int or data['format_version'] != 1:
        raise InputError('format_version must be 1')
    if not isinstance(data.get('candidates'), list):
        raise InputError('candidates must be an array')
    units = data.get('units')
    if not isinstance(units, dict):
        raise InputError('units must be an object')
    background = data.get('background_burden', {})
    if not isinstance(background, dict):
        raise InputError('background_burden must be an object')
    candidates = []
    for index, record in enumerate(data['candidates']):
        _keys(record, {'id', 'outcomes', 'tie_order'}, f'candidate {index}')
        if not isinstance(record.get('outcomes'), dict):
            raise InputError(f'candidate {index} outcomes must be an object')
        outcomes = {}
        for domain, values in record['outcomes'].items():
            _keys(values, {'burden', 'benefit', 'fractional_response', 'equity_weight'}, 'outcome')
            if 'burden' not in values or ('benefit' in values) == ('fractional_response' in values):
                raise InputError('each outcome needs burden and exactly one of benefit/fractional_response')
            weight = values.get('equity_weight', 1)
            outcomes[domain] = Outcome(values['burden'], values['benefit'], weight) if 'benefit' in values else Outcome.from_fraction(values['burden'], values['fractional_response'], weight)
        candidates.append(Candidate(record.get('id'), outcomes, record.get('tie_order', index)))
    return Problem(candidates, units, background)

def problem_to_dict(problem: Problem) -> dict:
    """Export explicit benefits, units and tie orders without discarding precision."""
    return {'format_version': 1, 'units': dict(problem.units), 'background_burden': dict(problem.background_burden), 'candidates': [{'id': c.id, 'tie_order': c.tie_order, 'outcomes': {d: {'burden': o.burden, 'benefit': o.benefit, 'equity_weight': o.equity_weight} for d, o in c.outcomes.items()}} for c in problem.candidates]}

def load_csv(path: str | Path, units: dict, background_burden: dict | None=None) -> Problem:
    """Read one row per candidate-domain; blank response/benefit is never assumed zero."""
    if not isinstance(units, dict) or (background_burden is not None and (not isinstance(background_burden, dict))):
        raise InputError('units and background_burden must be objects')
    records = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'unit_id', 'domain', 'burden'}
        allowed = required | {'benefit', 'fractional_response', 'equity_weight', 'tie_order'}
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields) or not required <= set(fields) or set(fields) - allowed:
            raise InputError('CSV headers must include unit_id, domain, burden and only documented fields')
        for row_number, row in enumerate(reader, 2):
            if None in row:
                raise InputError(f'CSV row {row_number} has too many values')
            row = {k: (v or '').strip() for k, v in row.items()}
            ident, domain = (row['unit_id'], row['domain'])
            if not ident or not domain:
                raise InputError(f'CSV row {row_number} needs unit_id and domain')
            record = records.setdefault(ident, {'id': ident, 'outcomes': {}, 'tie_order': len(records)})
            if domain in record['outcomes']:
                raise InputError(f'duplicate candidate-domain at CSV row {row_number}')
            try:
                tie = int(row['tie_order']) if row.get('tie_order') else record['tie_order']
                if record['outcomes'] and tie != record['tie_order']:
                    raise InputError(f'inconsistent tie_order at row {row_number}')
                record['tie_order'] = tie
                values = {'burden': float(row['burden'])}
                for key in ('benefit', 'fractional_response', 'equity_weight'):
                    if row.get(key):
                        values[key] = float(row[key])
            except ValueError as exc:
                raise InputError(f'invalid numeric value at CSV row {row_number}: {exc}') from exc
            record['outcomes'][domain] = values
    return problem_from_dict({'format_version': 1, 'units': units, 'background_burden': background_burden or {}, 'candidates': list(records.values())})

def load_problem(path: str | Path) -> Problem:
    """Read JSON; data_file may instead point to a tidy CSV relative to that JSON."""
    path = Path(path)
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if isinstance(data, dict) and 'data_file' in data:
        _keys(data, {'format_version', 'units', 'background_burden', 'data_file'}, 'problem')
        if type(data.get('format_version')) is not int or data['format_version'] != 1:
            raise InputError('format_version must be 1')
        if not isinstance(data['data_file'], str) or not data['data_file'].strip():
            raise InputError('data_file must be a CSV path')
        return load_csv(path.parent / data['data_file'], data.get('units'), data.get('background_burden'))
    return problem_from_dict(data)
import argparse
import json
import sys
from pathlib import Path

def _assignments(values):
    result = {}
    for item in values or []:
        key, sep, value = item.partition('=')
        if not sep or not key.strip() or key.strip() in result:
            raise InputError(f'expected a unique DOMAIN=VALUE argument: {item}')
        result[key.strip()] = float(value)
    return result

def main(argv=None):
    parser = argparse.ArgumentParser(description='Assess burden-referenced efficacy and allocate independent equal-cost actions.')
    parser.add_argument('--input', required=True, help='JSON data or JSON metadata referring to a CSV')
    parser.add_argument('--budget', required=True, type=int, help='Number of equal-cost actions')
    parser.add_argument('--primary', required=True, help='User-assigned primary problem/domain')
    parser.add_argument('--weight', action='append', metavar='DOMAIN=VALUE', help='Repeat for candidate-maximum composite comparison; sum must equal one')
    parser.add_argument('--minimum-benefit', action='append', metavar='DOMAIN=VALUE', help='Repeat for per-action constraints, e.g. air=0')
    parser.add_argument('--compare-local-fractions', action='store_true', help='Also compare local S/M ranking in the primary domain')
    parser.add_argument('--leave-budget-unspent', action='store_true', help='Select only strictly positive scores, up to budget')
    parser.add_argument('--output', help='Output JSON path; default is stdout')
    parser.add_argument('--version', action='version', version=__version__)
    args = parser.parse_args(argv)
    try:
        problem = load_problem(args.input)
        weights = _assignments(args.weight)
        minimum = _assignments(args.minimum_benefit)
        options = {'minimum_benefits': minimum, 'fill_budget': not args.leave_budget_unspent}
        primary = select(problem, benefit_scores(problem, args.primary), args.budget, **options)
        result = {'software': 'efficacy-allocation', 'version': __version__, 'settings': {'budget_actions': args.budget, 'primary_domain': args.primary, 'composite_weights': weights, **options}, 'role_potentials': role_potentials(problem, args.budget, **options), 'primary_allocation': evaluate(problem, primary)}
        if weights:
            comparison = select(problem, composite_scores(problem, weights), args.budget, **options)
            result['composite_allocation'] = evaluate(problem, comparison)
            result['primary_vs_composite'] = compare_portfolios(problem, comparison, primary)
        if args.compare_local_fractions:
            fractional = select(problem, fraction_scores(problem, args.primary), args.budget, **options)
            result['local_fraction_allocation'] = evaluate(problem, fractional)
            result['primary_vs_local_fraction'] = compare_portfolios(problem, fractional, primary)
        payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(payload, encoding='utf-8')
        else:
            sys.stdout.write(payload)
    except (InputError, OSError, ValueError, TypeError, OverflowError) as exc:
        parser.exit(2, f'error: {exc}\n')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
