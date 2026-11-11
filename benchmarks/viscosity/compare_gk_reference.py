#!/usr/bin/env python3
"""State-aware reference diagnostics for a completed hexane GK campaign.

Retains every declared setting and joint replica bootstrap draw. This is not
a same-pressure experimental comparison or a viscosity acceptance decision.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import experimental_reference as ref


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(fluid, density, estimates, temperatures, samples, sampled_temperatures):
    estimates, temperatures, samples, sampled_temperatures = [np.asarray(v, float)
        for v in (estimates, temperatures, samples, sampled_temperatures)]
    if (estimates.ndim != 1 or temperatures.shape != estimates.shape or
            samples.ndim != 2 or samples.shape != sampled_temperatures.shape or
            samples.shape[1:] != estimates.shape or samples.shape[0] < 100 or
            estimates.size == 0 or
            any(not np.all(np.isfinite(v)) for v in (estimates, temperatures, samples, sampled_temperatures))):
        raise ValueError('finite matched settings and joint bootstrap arrays required')
    # Repeated replica draws have few unique temperatures; preserve their pairing.
    unique, inverse = np.unique(np.concatenate((temperatures, sampled_temperatures.ravel())), return_inverse=True)
    values = np.array([ref.reference(fluid, float(t), density)['viscosity_pa_s']*1000 for t in unique])
    evaluated = values[inverse]
    references = evaluated[:len(estimates)]
    sampled_references = evaluated[len(estimates):].reshape(samples.shape)
    ratios = samples/sampled_references
    result = dict(reference_mpa_s=references, ratio=estimates/references,
        conditional_ratio_95_interval=np.quantile(ratios, [.025, .975], axis=0),
        conditional_temperature_95_interval_k=np.quantile(sampled_temperatures, [.025, .975], axis=0),
        nonpositive_draws=np.sum(samples <= 0, axis=0))
    return result, sampled_references, ratios


def load_campaign(campaign, analysis):
    campaign, analysis = Path(campaign).resolve(), Path(analysis).resolve()
    plan = json.loads((campaign/'plan.json').read_text())
    execution = json.loads((campaign/'execution.json').read_text())
    readiness = json.loads((campaign/'analysis-readiness.json').read_text())
    summary = json.loads((analysis/'summary.json').read_text())
    plan_hash = digest(campaign/'plan.json')
    if (plan['contract'] != 'sponge.hexane.gk-replicas.v1' or
            execution['stage'] != 'completed_sampling_pending_analysis' or
            execution['completed'] != plan['replicas'] or (campaign/'failure.json').exists() or
            any(item['plan_sha256'] != plan_hash for item in (execution, readiness, summary)) or
            summary['status'] != 'conditional_finite_window_diagnostics_not_transport_acceptance' or
            summary['formal_viscosity_acceptance'] is not False or
            summary['analysis_sha256'] != readiness['source_hashes']['benchmarks/viscosity/hexane_gk_analysis.py']):
        raise ValueError('completed frozen conditional analysis required')
    for name, expected in readiness['source_hashes'].items():
        path = (campaign/'analysis-sources'/name).resolve()
        if not path.is_relative_to(campaign/'analysis-sources') or digest(path) != expected:
            raise ValueError('analysis source identity')
    for name in ('benchmarks/viscosity/hexane_model.py', 'benchmarks/viscosity/water_model.py',
                 'benchmarks/viscosity/lj_model.py', 'scripts/analyze_green_kubo.py'):
        if (readiness['source_hashes'].get(name) != plan['source_hashes'].get(name) or
                name not in plan['source_hashes'] or
                summary['dependencies'].get(name) != plan['source_hashes'][name]):
            raise ValueError('analysis dependency differs from production plan')
    settings = plan['analysis']
    layout = [(f'start-{start:g}-decimation-{dec}-demean-{int(demean)}', window)
        for start in settings['start_ps'] for dec in settings['decimation']
        for demean in settings['demean'] for window in settings['windows_ps']]
    replicas = summary['replicas']
    if (len(replicas) != 4 or len(set(plan['replicas'])) != 4 or
            [r['replica'] for r in replicas] != plan['replicas']):
        raise ValueError('replica identity/order')
    for replica in replicas:
        if digest(campaign/'runs'/replica['replica']/'summary.json') != replica['run_summary_sha256']:
            raise ValueError('run summary identity')
    for rows in [summary['aggregate']] + [r['estimates'] for r in replicas]:
        if [(r['setting'], r['window_ps']) for r in rows] != layout:
            raise ValueError('all declared settings required in original order')
    numerator = np.array([[r['numerator_mpa_s_k'] for r in replica['estimates']] for replica in replicas])
    temperature = np.array([[r['temperature_k'] for r in replica['estimates']] for replica in replicas])
    if not np.all(np.isfinite(numerator)) or not np.all(np.isfinite(temperature)) or np.any(temperature <= 0):
        raise ValueError('invalid replica estimates')
    expected_indices = np.random.default_rng(settings['replica_bootstrap_seed']).integers(
        len(replicas), size=(settings['replica_bootstrap_draws'], len(replicas)))
    with np.load(analysis/'bootstrap.npz', allow_pickle=False) as bootstrap:
        indices, samples = bootstrap['indices'], bootstrap['viscosity_mpa_s']
    np.testing.assert_array_equal(indices, expected_indices)
    sampled_temperature = temperature[indices].mean(axis=1)
    np.testing.assert_allclose(samples, numerator[indices].mean(axis=1)/sampled_temperature, rtol=1e-13, atol=0)
    estimates = numerator.mean(axis=0)/temperature.mean(axis=0)
    np.testing.assert_allclose(estimates, [r['viscosity_mpa_s'] for r in summary['aggregate']], rtol=1e-13, atol=0)
    np.testing.assert_allclose(temperature.mean(axis=0), [r['temperature_k'] for r in summary['aggregate']], rtol=1e-13, atol=0)
    np.testing.assert_allclose(np.quantile(samples, [.025, .975], axis=0).T,
        [r['conditional_replica_95_interval_mpa_s'] for r in summary['aggregate']], rtol=1e-13, atol=0)
    return summary, estimates, temperature.mean(axis=0), samples, sampled_temperature, indices


def run(campaign, analysis, output):
    campaign, analysis, output = map(lambda p: Path(p).resolve(), (campaign, analysis, output))
    if output.exists():
        raise ValueError('new output directory required')
    summary, estimates, temperature, samples, sampled_temperature, indices = load_campaign(campaign, analysis)
    density = summary['physical_model']['actual_density_kg_m3']
    diagnostic, sampled_reference, ratios = compare('n_hexane', density, estimates, temperature, samples, sampled_temperature)
    rows = []
    for i, original in enumerate(summary['aggregate']):
        rows.append(dict(setting=original['setting'], window_ps=original['window_ps'],
            viscosity_mpa_s=float(estimates[i]), temperature_k=float(temperature[i]),
            reference=ref.reference('n_hexane', float(temperature[i]), density),
            md_to_reference_ratio=float(diagnostic['ratio'][i]),
            conditional_ratio_95_interval=diagnostic['conditional_ratio_95_interval'][:, i].tolist(),
            conditional_temperature_95_interval_k=diagnostic['conditional_temperature_95_interval_k'][:, i].tolist(),
            nonpositive_viscosity_bootstrap_draws=int(diagnostic['nonpositive_draws'][i])))
    result = dict(status='conditional_T_rho_reference_diagnostic_not_experimental_acceptance',
        formal_viscosity_acceptance=False, phase_and_pressure_validated=False,
        reference_uncertainty_propagated=False, density_kg_m3=density, settings=rows,
        source_hashes={str(p): digest(p) for p in (campaign/'plan.json', campaign/'analysis-readiness.json',
            analysis/'summary.json', analysis/'bootstrap.npz', Path(__file__), Path(ref.__file__))},
        limitations=['Four whole-replica resamples retain temperature/viscosity covariance and all settings.',
            'Negative finite-window integrals and bootstrap draws are retained without selection.',
            'Pooled unequal-temperature NVE replicas do not define a single equilibrium state.',
            'Reference is evaluated at pooled measured T and fixed model density; no EOS or pressure/phase matching.',
            'Intervals include only conditional replica sampling, not reference, model, equilibration or finite-window errors.',
            'No direct ratio to a discrete experimental measurement at a different temperature.'])
    output.mkdir(parents=True)
    np.savez(output/'joint-bootstrap.npz', indices=indices, temperature_k=sampled_temperature,
        viscosity_mpa_s=samples, reference_mpa_s=sampled_reference, md_to_reference_ratio=ratios)
    result['joint_bootstrap_sha256'] = digest(output/'joint-bootstrap.npz')
    (output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign', 'analysis', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    report = run(args.campaign, args.analysis, args.output)
    print(json.dumps(dict(status=report['status'], settings=len(report['settings'])), indent=2))
